#!/usr/bin/env python3
import argparse
import datetime
import importlib
import os
import pathlib
import shutil
import subprocess
import sys
from typing import List

import jinja2


def _time_delta_to_slurm_str(delta):
    seconds = delta.total_seconds()
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    hours = int(minutes // 60)
    minutes = int(minutes % 60)
    days = int(hours // 24)
    hours = int(hours % 24)
    return f"{days}-{hours:0=2d}:{minutes:0=2d}:{seconds:0=2d}"


def main():
    parser = argparse.ArgumentParser(description="Train or evaluate a model.")
    parser.add_argument(
        "--config",
        default=[],
        nargs="+",
        dest="config_names",
        help="Configuration file(s)",
    )
    parser.add_argument(
        "--train",
        default=None,
        type=int,
        dest="training_epochs",
        help="Number of epochs to train",
    )
    parser.add_argument(
        "--retrain",
        type=int,
        default=None,
        dest="retraining_epochs",
        help="Number of epochs to retrain",
    )
    parser.add_argument(
        "--retrain-iters",
        type=int,
        default=1,
        dest="retraining_iterations",
        help="Number of iterations to run the retrain process.",
    )
    parser.add_argument(
        "--eval",
        default=False,
        dest="run_evaluator",
        action="store_true",
    )
    parser.add_argument(
        "--dry-run",
        default=False,
        dest="dry_run",
        action="store_true",
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        dest="seed_idx",
        help="The index indicating which of the seeds to use",
    )
    parser.add_argument(
        "--cv",
        default=None,
        type=int,
        dest="cross_validation_split_index",
        help="[Optional] The index of the cross-validation split to use.",
    )
    parser.add_argument(
        "--erda",
        default=False,
        dest="use_erda",
        action="store_true",
    )
    parser.add_argument(
        "--datadir",
        default=None,
        type=str,
        dest="data_directory",
        help="Prefix for dataset and model directories.",
    )
    parser.add_argument(
        "--projectdir",
        default=None,
        type=str,
        dest="project_directory",
        help="Project directory.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=None,
        dest="model_name_suffix",
        help="Model name suffix",
    )
    parser.add_argument(
        "--sanity-check",
        default=False,
        dest="is_sanity_check",
        action="store_true",
    )
    parser.add_argument(
        "--slurm-minutes",
        type=int,
        default=None,
        dest="slurm_minutes",
        help="Optional time limit in minutes for slurm job",
    )
    parser.add_argument(
        "--profile",
        default=False,
        dest="run_profiler",
        action="store_true",
    )
    parser.add_argument(
        "--tensorboard",
        default=False,
        dest="use_tensorboard",
        action="store_true",
    )
    parser.add_argument(
        "--redirect_output_fn",
        type=str,
        default=None,
        dest="redirect_output_fn",
        help="Optional. Name of file to which output is redirected.",
    )
    parser.add_argument(
        "--gpu_number",
        default=None,
        type=int,
        dest="gpu_number",
        help="Optional. Which GPU to use.",
    )
    parser.add_argument(
        "--models-dir",
        default=None,
        dest="alt_models_directory",
    )

    args = parser.parse_args()

    if args.training_epochs is not None and args.training_epochs < 1:
        raise ValueError("Invalid retraining epochs")

    if args.retraining_epochs is not None and args.retraining_epochs < 1:
        raise ValueError("Invalid retraining epochs")

    is_training = False
    is_retraining = False
    epochs = 1
    if args.training_epochs is not None:
        epochs = args.training_epochs
        is_training = True
    if args.retraining_epochs is not None:
        epochs = args.retraining_epochs
        is_retraining = True

    project_directory = os.environ.get("DLC_PROJECT_DIRECTORY")
    if project_directory is None:
        project_directory = args.project_directory
    if project_directory is None:
        project_directory = str(pathlib.Path(__file__).parent.parent)
    project_directory = pathlib.Path(project_directory).absolute()
    if not project_directory.exists():
        raise ValueError(f"Invalid project directory: {project_directory}")
    else:
        print(f"Using project directory:\n{project_directory}")

    project_directory_str = str(project_directory)
    if project_directory_str not in sys.path:
        sys.path.append(project_directory_str)

    data_directory = os.environ.get("DLC_DATA_DIRECTORY")
    if data_directory is None:
        data_directory = args.data_directory
    if data_directory is None:
        raise ValueError("No data directory provided.")
    data_directory = pathlib.Path(data_directory).absolute()
    if not data_directory.exists():
        raise ValueError(f"Invalid data directory: {data_directory}")
    else:
        print(f"Using data directory:\n{data_directory}")

    config_mod = importlib.import_module("dlc.tools.config")
    config_paths = []
    for config_name in args.config_names:
        config_path = project_directory.joinpath(f"config/{config_name}.yml")
        config_paths.append(str(config_path))
    config = config_mod.read_config(config_paths)

    within_colab = None
    try:
        import colab

        within_colab = True
        print("Running within Colab")
    except ModuleNotFoundError:
        within_colab = False

    python_path_additions = None
    conda_prefix = os.environ.get("CONDA_PREFIX", None)
    conda_env_name = "dlc"
    if "python" in config:
        paths = config["python"]["path_additions"]
        python_path_additions = ":".join(paths)
        if "conda" in config["python"]:
            conda_prefix = config["python"]["conda"].get(
                "prefix",
                conda_prefix,
            )
            conda_env_name = config["python"]["conda"].get(
                "env_name",
                conda_env_name,
            )
    if conda_prefix is not None:
        print("Using conda prefix: ", conda_prefix)
        print("Using conda env: ", conda_env_name)

    run_plot = "plots" in config["settings"]

    model_name_suffix = None
    if args.model_name_suffix is not None:
        model_name_suffix = args.model_name_suffix
    elif "model_name_suffix" in config:
        model_name_suffix = "-".join(config["model_name_suffix"])
    if args.is_sanity_check:
        if model_name_suffix is not None:
            model_name_suffix = f"{model_name_suffix}-sanity"
        else:
            model_name_suffix = "sanity"
    if args.cross_validation_split_index is not None:
        k = args.cross_validation_split_index
        model_name_suffix = f"{model_name_suffix}-cv-{k}"
    print("Model name suffix: ", model_name_suffix)

    seeds = config["settings"]["seeds"]
    num_seeds = len(seeds)
    if args.seed_idx > num_seeds - 1:
        raise ValueError("Invalid seed index")
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(pathlib.Path(__file__).parent),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    script_template = env.get_template("train_eval.sh.template")
    script = script_template.render(
        seed=seeds[args.seed_idx],
        splitter_seed=config["settings"]["split_seed"],
        run_evaluator=args.run_evaluator,
        epochs=epochs,
        is_training=is_training,
        is_retraining=is_retraining,
        retraining_iterations=args.retraining_iterations,
        config_paths=",".join(config_paths),
        data_directory=str(data_directory),
        project_directory=project_directory,
        model_name_suffix=model_name_suffix,
        use_erda=args.use_erda,
        python_path_additions=python_path_additions,
        conda_prefix=conda_prefix,
        conda_env_name=conda_env_name,
        run_plot=run_plot,
        within_colab=within_colab,
        is_sanity_check=args.is_sanity_check,
        run_profiler=args.run_profiler,
        use_tensorboard=args.use_tensorboard,
        cross_validation_split_index=args.cross_validation_split_index,
        redirect_output_fn=args.redirect_output_fn,
        gpu_number=args.gpu_number,
        alt_models_directory=args.alt_models_directory,
    )

    if shutil.which("sbatch") is not None:
        command: List[str] = list()
        command.append("sbatch")
        command.append(
            "--job-name={}".format(config["settings"]["model"]),
        )
        command.append("-p")
        command.append(config["slurm"]["partition"])
        if "gres" in config["slurm"]:
            command.append("--gres={}".format(config["slurm"]["gres"]))
        command.append("--ntasks=1")
        command.append(
            "--cpus-per-task={}".format(config["slurm"]["cpus_per_task"]),
        )

        memory_mb = 1024 * config["slurm"]["memory_gb"]
        command.append(f"--mem={memory_mb}M")

        delta = datetime.timedelta(minutes=15)
        if args.slurm_minutes is not None:
            delta = datetime.timedelta(minutes=args.slurm_minutes)
        elif (
            "time_limit_minutes" in config["slurm"]
            and config["slurm"]["time_limit_minutes"] is not None
        ):
            x = config["slurm"]["time_limit_minutes"]
            delta = datetime.timedelta(minutes=x)
        elif (
            "approx_ms_per_epoch_step" in config["slurm"]
            and config["slurm"]["approx_ms_per_epoch_step"] is not None
        ):
            # estimated overhead per epoch for plotting and checkpointing
            overhead_ms_per_epoch = 24000
            x = config["slurm"]["approx_ms_per_epoch_step"]
            x *= config["settings"].get("training_steps", 480)
            x += overhead_ms_per_epoch
            x *= epochs
            x += 600000  # +10 min
            delta = datetime.timedelta(milliseconds=x)
        else:
            print("No slurm time information was provided, limit set to 15 min")
        time_limit = _time_delta_to_slurm_str(delta)
        print(f"Slurm time limit: {time_limit}")
        command.append(f"--time={time_limit}")

        command.append("--exclude=a00610,a00757")

        email = config["slurm"].get("email")
        if email is not None:
            command.append("--mail-type=END,FAIL")
            command.append(f"--mail-user={email}")

        if not args.dry_run:
            print("Spawning job subprocess using Slurm...")
            subprocess.run(command, input=script.lstrip(), text=True)
        else:
            print("Slurm command:")
            print("\n".join(command))
            print("Script:")
            print(script)
    else:
        if not args.dry_run:
            print("Spawning job subprocess...")
            with subprocess.Popen(
                script,
                shell=True,
                executable="/bin/bash",
            ) as p:
                try:
                    p.wait()
                except Exception:
                    p.kill()
                    p.wait()
        else:
            print("Script: ")
            print(script)


if __name__ == "__main__":
    main()
