#!/usr/bin/env python3
import argparse
import multiprocessing
import os
import pathlib
import subprocess
from functools import partial
from typing import List
from typing import Optional
from typing import Union


def stack_image(
    input_paths: List[pathlib.Path],
    *,
    output_path: pathlib.Path,
    overwrite: bool = False,
):
    prefix_list = []

    suffix = ""
    for path in input_paths:
        splitted_name = path.name.split("_")
        prefix_list.append(splitted_name[0])
        if suffix == "":
            suffix = "_".join(splitted_name[1:])
    prefix = "_".join(prefix_list)

    output_vrt_path = output_path.joinpath(f"{prefix}_{suffix}.vrt")
    if not suffix.endswith(".tif"):
        output_tif_path = str(output_path.joinpath(f"{prefix}_{suffix}.tif"))
    else:
        output_tif_path = str(output_path.joinpath(f"{prefix}_{suffix}"))

    # TODO: Currently assumes 1 band per image
    vrt_command = ["gdalbuildvrt", "-q", "-separate", str(output_vrt_path)] + [
        str(path) for path in input_paths
    ]

    tif_command = [
        "gdal_translate",
        "-q",
        "-of",
        "GTiff",
        "-co",
        "BIGTIFF=YES",
        "-co",
        "tiled=yes",
        "-co",
        "blockxsize=256",
        "-co",
        "blockysize=256",
        "-co",
        "compress=LZW",
        "-co",
        "interleave=band",
        "-r",
        "bilinear",
        str(output_vrt_path),
        output_tif_path,
    ]

    result_vrt = subprocess.run(vrt_command)
    result_tif = subprocess.run(tif_command)
    output_vrt_path.unlink()

    return result_vrt, result_tif


def stack_images(
    *,
    input_path: Union[pathlib.Path, str],
    search_pattern: str,
    output_path: Union[pathlib.Path, str],
    verbose: bool = True,
    dry_run: bool = False,
    job_slice: Optional[slice] = None,
    n_processes: Optional[int] = None,
    overwrite: bool = False,
) -> None:
    input_path = pathlib.Path(input_path)
    output_path = pathlib.Path(output_path)
    groups = {}
    for path in input_path.rglob(search_pattern):
        splitted_name = path.name.split("_")
        suffix = "_".join(splitted_name[1:])
        if suffix not in groups:
            groups[suffix] = []
        groups[suffix].append(path)
    # Filter-out groups with only one element
    jobs = list(filter(lambda x: len(x) > 1, groups.values()))

    n_jobs = len(jobs)
    if job_slice is None:
        job_slice = slice(0, len(jobs))
    slice_jobs = jobs[job_slice]
    n_slice_jobs = len(slice_jobs)

    if n_processes is None:
        n_processes = os.cpu_count()
    n_processes = min(n_processes, n_slice_jobs, os.cpu_count())

    if verbose:
        print(f"Found {len(groups)} groups of files to stack")
        print(
            "Will process {}/{} groups using {} CPUs and {}.".format(
                n_slice_jobs,
                n_jobs,
                n_processes,
                job_slice,
            )
        )

    results = []
    if not dry_run:
        with multiprocessing.Pool(processes=n_processes) as pool:
            results = pool.map(
                partial(
                    stack_image,
                    overwrite=overwrite,
                    output_path=output_path,
                ),
                slice_jobs,
            )
    if verbose:
        print("Finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stack images that share a common suffix."
    )
    parser.add_argument(
        "--input_path",
        dest="input_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_path",
        dest="output_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--pattern",
        dest="search_pattern",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-v",
        dest="verbose",
        action="store_true",
    )
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
    )
    parser.add_argument(
        "-slice_start", dest="slice_start", type=int, required=False, default=None
    )
    parser.add_argument(
        "-slice_end", dest="slice_end", type=int, required=False, default=None
    )
    parser.add_argument(
        "-n_processes", dest="n_processes", type=int, required=False, default=1
    )
    args = parser.parse_args()
    job_slice = None
    if args.slice_start is not None and args.slice_end is not None:
        job_slice = slice(args.slice_start, args.slice_end)

    stack_images(
        input_path=args.input_path,
        search_pattern=args.search_pattern,
        output_path=args.output_path,
        verbose=args.verbose,
        dry_run=args.dry_run,
        job_slice=job_slice,
        n_processes=args.n_processes,
    )
