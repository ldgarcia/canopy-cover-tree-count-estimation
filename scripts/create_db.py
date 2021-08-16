def main(
    img_path: str,
    img_search_pattern: str,
    areas_path: str,
    polygons_path: str,
    output_path: str,
):
    import pathlib
    import dlc.tools.db

    print("[1/4] Creating tiles metadata...")
    tiles = dlc.tools.db.get_objects_from_images(
        pathlib.Path(img_path),
        img_search_pattern,
    )

    print("[2/4] Updating database...")
    database = dlc.tools.db.load_database(
        tiles=tiles,
        areas=areas_path,
        polygons=polygons_path,
        outliers=True,
    )
    print("[3/4] Fixing overlaps in polygons...")
    print("Initial scale = 1.0")
    fixed_polygons_10 = dlc.tools.db.fix_overlapped_polygons(
        database.polygons.query("is_orphan == False"),
        initial_scale=1.0,
    )
    print("Initial scale = 0.9")
    fixed_polygons_09 = dlc.tools.db.fix_overlapped_polygons(
        database.polygons.query("is_orphan == False"),
        initial_scale=0.9,
    )

    print("[4/4] Saving databases as files...")
    output_path_pl = pathlib.Path(output_path)
    if not output_path_pl.exists():
        output_path_pl.mkdir(parents=True)

    database.to_files(output_path_pl)
    fixed_polygons_10.to_file(
        output_path_pl.joinpath("fixed-polygons-1_0.geojson"),
        driver="GeoJSON",
    )
    fixed_polygons_09.to_file(
        output_path_pl.joinpath("fixed-polygons-0_9.geojson"),
        driver="GeoJSON",
    )

    print("Finished! Summary:")
    print(f"Total tiles: {len(database.tiles)}")
    print(f"Total areas: {len(database.areas)}")
    print(f"Total polygons (before fix): {len(database.polygons)}")
    print(f"Total polygons (after fix, scale 1.0): {len(fixed_polygons_10)}")
    print(f"Total polygons (after fix, scale 0.9): {len(fixed_polygons_09)}")


if __name__ == "__main__":
    import argparse
    import pathlib
    import shutil
    import subprocess
    from typing import List

    import jinja2

    parser = argparse.ArgumentParser(
        description="Create a database from raw images and given annotations."
    )

    parser.add_argument(
        "--img-path",
        dest="img_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--pattern",
        dest="img_search_pattern",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--areas-path",
        dest="areas_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--polygons-path",
        dest="polygons_path",
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
        "--slurm",
        dest="run_with_slurm",
        action="store_true",
    )
    parser.add_argument(
        "--slurm-job-name",
        dest="slurm_job_name",
        type=str,
        required=False,
        default="create-db",
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
        default="4000M",
    )
    parser.add_argument(
        "--slurm-time-limit",
        dest="slurm_time_limit",
        type=str,
        required=False,
        default="2:00:00",
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
            img_path=args.img_path,
            img_search_pattern=args.img_search_pattern,
            areas_path=args.areas_path,
            polygons_path=args.polygons_path,
            output_path=args.output_path,
        )
    else:
        # Submit a batch job to the cluster.
        if shutil.which("sbatch") is not None:
            print("[1/2] Creating custom Slurm script...")
            env = jinja2.Environment(
                loader=jinja2.FileSystemLoader(pathlib.Path(__file__).parent),
            )
            script_template = env.get_template("create_db.sh.template")
            script = script_template.render(
                img_path=args.img_path,
                img_search_pattern=args.img_search_pattern,
                areas_path=args.areas_path,
                polygons_path=args.polygons_path,
                output_path=args.output_path,
                conda_env=args.conda_env,
            )
            print("[2/2] Spawning Slurm job...")
            command: List[str] = list()
            command.append("sbatch")
            command.append(
                "--job-name={}".format(args.slurm_job_name),
            )
            command.append("-p")
            command.append(args.slurm_partition)
            command.append("--ntasks=1")
            command.append("--cpus-per-task=2")
            command.append(f"--mem={args.slurm_memory}")
            command.append(f"--time={args.slurm_time_limit}")
            if args.slurm_email is not None:
                command.append("--mail-type=END,FAIL")
                command.append(f"--mail-user={args.slurm_email}")
            subprocess.run(command, input=script.lstrip(), text=True)
        else:
            raise Exception("sbatch executable not detected.")
