import pathlib
import subprocess
from typing import List
from typing import Tuple
from typing import Union


def main(
    *,
    dataset_name: str,
    datasets_path: Union[str, pathlib.Path],
    dataset_img_dir: str,
    output_path: Union[str, pathlib.Path],
    n_processes: int,
    dry_run: bool,
    cache_enabled: bool,
    cache_shared: bool,
    use_fixed_polygons: bool,
    initial_scale: float,
    creators_group: str,
    create_zip: bool,
    overwrite: bool,
):
    from pprint import pprint

    import dlc.tools.db
    import dlc.frames.factory
    import dlc.frames.creators.data

    if use_fixed_polygons:
        initial_scale_s = f"{initial_scale:.1f}".replace(".", "_")
        frames_name = "{}-{}-{}".format(
            dataset_name,
            creators_group,
            initial_scale_s,
        )
    else:
        frames_name = f"{dataset_name}-{creators_group}"

    frames_path = pathlib.Path(output_path).joinpath(frames_name)
    if not frames_path.exists():
        frames_path.mkdir(parents=True)

    tiles_path = pathlib.Path(
        "{}/{}/tiles.geojson".format(
            datasets_path,
            dataset_name,
        )
    )
    areas_path = pathlib.Path(
        "{}/{}/areas.geojson".format(
            datasets_path,
            dataset_name,
        )
    )
    polygons_path = pathlib.Path(
        "{}/{}/polygons.geojson".format(
            datasets_path,
            dataset_name,
        )
    )

    predicted_polygons_path = pathlib.Path(
        "{}/{}/predicted-polygons.geojson".format(
            datasets_path,
            dataset_name,
        )
    )

    initial_scale_s = f"{initial_scale:.1f}".replace(".", "_")
    fixed_polygons_path = pathlib.Path(
        "{}/{}/fixed-polygons-{}.geojson".format(
            datasets_path,
            dataset_name,
            initial_scale_s,
        )
    )

    print("Loading tiles...")
    tiles = dlc.tools.db.get_tiles_with_areas(tiles_path, areas_path)

    print("Loading areas...")
    areas = dlc.tools.db.get_areas_with_tiles(areas_path, tiles_path)

    if use_fixed_polygons:
        print("Loading fixed polygons...")
        print(f"Initial scale: {initial_scale:.1f}")
        if not fixed_polygons_path.exists():
            polygons = dlc.tools.db.get_polygons_with_tile_and_area(
                polygons_path,
                areas_path,
                tiles_path,
            )
            polygons = dlc.tools.db.fix_overlapped_polygons(
                polygons,
                initial_scale=initial_scale,
            )
        else:
            polygons = dlc.tools.db.get_polygons_with_tile_and_area(
                fixed_polygons_path,
                areas_path,
                tiles_path,
            )
    else:
        print("Loading polygons...")
        polygons = dlc.tools.db.get_polygons_with_tile_and_area(
            polygons_path,
            areas_path,
            tiles_path,
        )

    predicted_polygons = None
    if predicted_polygons_path.exists():
        print("Found predicted polygons file.")
        predicted_polygons = dlc.tools.db.get_polygons_with_tile_and_area(
            predicted_polygons_path,
            areas_path,
            tiles_path,
        )

    dataset_path = pathlib.Path(datasets_path).joinpath(dataset_name)

    data = dlc.frames.creators.data.CoreFrameDataSource(
        dataset_name,
        dataset_path.joinpath(dataset_img_dir),
        tiles,
        areas,
        polygons,
    )

    predicted_data = None
    if predicted_polygons is not None:
        predicted_data = dlc.frames.creators.data.CoreFrameDataSource(
            dataset_name,
            dataset_path.joinpath(dataset_img_dir),
            tiles,
            areas,
            predicted_polygons,
        )

    print("Creating factory...")
    print(f"Group: {creators_group}")

    creator_names: Tuple[str, ...] = ()
    if creators_group in ("segmentation", "density"):
        creator_names += (
            "image",
            "props",
            "model",
            "segmentation-mask",
            "segmentation-boundary-weights",
            "desired-weights",
            "outlier-weights",
        )
    if creators_group in ("density",):
        creator_names += (
            "gaussian-density",
            "th-gaussian-density",
            "dm-gaussian-density",
            "uniform-density",
            "edt-density",
        )

    if dataset_name == "rwanda":
        filter_sizes = [19, 15]
    elif dataset_name == "sahara-sahel":
        filter_sizes = [11, 13, 9]
    else:
        msg = f"Invalid value for data: {dataset_name}"
        raise ValueError(msg)

    gaussian_options: List[dlc.frames.factory.GaussianOptions] = []
    for filter_size in filter_sizes:
        gaussian_options.append(
            dict(
                filter_size=filter_size,
                sigma=5.0,
                centroid_type="energy",
                filter_target="centroid",
            )
        )
        gaussian_options.append(
            dict(
                filter_size=filter_size,
                sigma=5.0,
                centroid_type="standard",
                filter_target="centroid",
            )
        )
    gaussian_options.append(
        dict(
            filter_size=3,
            sigma=5.0,
            centroid_type="energy",
            filter_target="polygon",
        ),
    )

    th_gaussian_options: List[dlc.frames.factory.THGaussianOptions] = [
        dict(
            sigma=5.0,
            thresh_z_score=0.1,
            centroid_type="energy",
            filter_target="centroid",
        ),
        dict(
            sigma=5.0,
            thresh_z_score=None,
            centroid_type="energy",
            filter_target="centroid",
        ),
    ]

    dm_gaussian_options: List[dlc.frames.factory.DMGaussianOptions] = []
    for filter_size in filter_sizes:
        dm_gaussian_options.append(dict(sigma=5.0, filter_size=filter_size))

    factory = dlc.frames.factory.create_and_configure_factory(
        data,
        frames_path,
        creator_names=creator_names,
        gaussian_options=gaussian_options,
        th_gaussian_options=th_gaussian_options,
        dm_gaussian_options=dm_gaussian_options,
        predicted_data=predicted_data,
    )
    print("Factory creator keys:")
    pprint(factory.keys)

    print("Loading raster data...")
    data.load_raster_data(n_processes=n_processes)

    print("Running creator jobs...")
    result = factory.run_jobs(
        (tiles, areas),
        dry_run=dry_run,
        job_slice=None,
        n_processes=n_processes,
        save_keys=None,
        cache_enabled=cache_enabled,
        cache_shared=cache_shared,
        overwrite=overwrite,
    )
    # frames = dlc.tools.db.add_n_polygons(result.frames, polygons)
    output_path = pathlib.Path(output_path)
    result.frames.to_file(
        frames_path.joinpath("frames.geojson"),
        driver="GeoJSON",
    )

    if not dry_run and create_zip:
        zip_path = frames_path.parent.joinpath(
            f"frames-{frames_name}.zip".format(frames_name)
        )
        if zip_path.exists():
            zip_path.unlink()

        print(f"Creating zip archive in {zip_path}...")
        command = [
            "zip",
            "-j",
            "-r",
            str(zip_path),
            str(frames_path),
        ]
        subprocess.run(command)

    print("Finished!")


if __name__ == "__main__":
    import argparse
    import pathlib
    import shutil

    import jinja2

    parser = argparse.ArgumentParser(
        description="Create a database from raw images and given annotations."
    )

    parser.add_argument(
        "--dataset-name",
        dest="dataset_name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--datasets-path",
        dest="datasets_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--img-dir",
        dest="dataset_img_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output-path",
        dest="output_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--creators-group",
        dest="creators_group",
        type=str,
        default="density",
    )
    parser.add_argument(
        "--n-processes",
        dest="n_processes",
        type=int,
        required=False,
        default=12,
    )
    parser.add_argument(
        "--cache-enabled",
        dest="cache_enabled",
        action="store_true",
    )
    parser.set_defaults(cache_enabled=False)
    parser.add_argument(
        "--cache-shared",
        dest="cache_shared",
        action="store_true",
    )
    parser.set_defaults(cache_shared=False)
    parser.add_argument(
        "--fixed-polygons",
        dest="use_fixed_polygons",
        action="store_true",
    )
    parser.add_argument(
        "--no-fixed-polygons",
        dest="use_fixed_polygons",
        action="store_false",
    )
    parser.set_defaults(use_fixed_polygons=True)
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
    )
    parser.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
    )
    parser.set_defaults(overwrite=True)
    parser.add_argument(
        "--initial-scale",
        dest="initial_scale",
        type=float,
        required=False,
        default=0.9,
    )
    parser.add_argument(
        "--zip",
        dest="create_zip",
        action="store_true",
    )
    parser.add_argument(
        "--no-zip",
        dest="create_zip",
        action="store_false",
    )
    parser.set_defaults(create_zip=True)
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
    )
    parser.set_defaults(dry_run=False)
    parser.add_argument(
        "--slurm",
        dest="run_with_slurm",
        action="store_true",
    )
    parser.set_defaults(run_with_slurm=False)
    parser.add_argument(
        "--slurm-job-name",
        dest="slurm_job_name",
        type=str,
        required=False,
        default="create-frames",
    )
    parser.add_argument(
        "--slurm-partition",
        dest="slurm_partition",
        type=str,
        required=False,
        default="image1",
    )
    parser.add_argument(
        "--slurm-memory",
        dest="slurm_memory",
        type=str,
        required=False,
        default="24000M",
    )
    parser.add_argument(
        "--slurm-time-limit",
        dest="slurm_time_limit",
        type=str,
        required=False,
        default="3:00:00",
    )
    parser.add_argument(
        "--slurm-email",
        dest="slurm_email",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--conda-env",
        dest="conda_env",
        type=str,
        required=False,
        default="dlc",
    )

    args = parser.parse_args()

    if not args.run_with_slurm:
        # Run locally in the current machine.
        main(
            dataset_name=args.dataset_name,
            datasets_path=args.datasets_path,
            dataset_img_dir=args.dataset_img_dir,
            output_path=args.output_path,
            creators_group=args.creators_group,
            n_processes=args.n_processes,
            dry_run=args.dry_run,
            cache_enabled=args.cache_enabled,
            cache_shared=args.cache_shared,
            use_fixed_polygons=args.use_fixed_polygons,
            initial_scale=args.initial_scale,
            create_zip=args.create_zip,
            overwrite=args.overwrite,
        )
    else:
        # Submit a batch job to the cluster.
        if shutil.which("sbatch") is not None:
            print("[1/2] Creating custom Slurm script...")
            env = jinja2.Environment(
                loader=jinja2.FileSystemLoader(pathlib.Path(__file__).parent),
                trim_blocks=True,
                lstrip_blocks=True,
            )
            script_template = env.get_template("create_frames.sh.template")
            script = script_template.render(
                dataset_name=args.dataset_name,
                datasets_path=args.datasets_path,
                dataset_img_dir=args.dataset_img_dir,
                output_path=args.output_path,
                creators_group=args.creators_group,
                n_processes=args.n_processes,
                dry_run=args.dry_run,
                cache_enabled=args.cache_enabled,
                cache_shared=args.cache_shared,
                use_fixed_polygons=args.use_fixed_polygons,
                initial_scale=args.initial_scale,
                create_zip=args.create_zip,
                overwrite=args.overwrite,
                conda_env=args.conda_env,
            )
            if not args.dry_run:
                print("[2/2] Spawning Slurm job...")
                command: List[str] = list()
                command.append("sbatch")
                command.append(
                    "--job-name={}".format(args.slurm_job_name),
                )
                command.append("-p")
                command.append(args.slurm_partition)
                command.append("--ntasks=1")
                command.append(f"--cpus-per-task={args.n_processes}")
                command.append(f"--mem={args.slurm_memory}")
                command.append(f"--time={args.slurm_time_limit}")
                if args.slurm_email is not None:
                    command.append("--mail-type=END,FAIL")
                    command.append(f"--mail-user={args.slurm_email}")
                subprocess.run(command, input=script.lstrip(), text=True)
            else:
                print("[2/2] Printing created script (dry run)...")
                print(script.lstrip())
        else:
            raise Exception("sbatch executable not detected.")
