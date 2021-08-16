# Project setup

## Update the Conda environment

```bash
mamba env update -n dlc -f environment.yml
```

## Environment variables
Some configuration values can be provided in environment variables.
They can be exported in the `~/.bashrc` file.

The ERDA configuration can be provided using:
```bash
export DLC_ERDA_KEY="${HOME}/.ssh/id_rsa"
export DLC_ERDA_USER="<your username here>"
export DLC_ERDA_DIR="./counting_project"
export DLC_ERDA_MOUNT="${HOME}/ERDA/mnt"
```

The parent directory that contains the frames directories can be provided using:
```bash
export DLC_DATA_DIRECTORY="${HOME}"
```
Otherwise, it needs to be passed as an argument when running the script.

## Slurm quick reference
See the Cluster's Wiki for more info.: https://diku-dk.github.io/wiki/slurm-cluster

### Run jobs
```bash
sbatch <script_path>
```

### Check our jobs
```bash
squeue | grep `whoami`
```

### Cancel our jobs
```bash
scancel <jobid>
```

## Scripts

### Model training and evaluation

This script allows to train and/or evaluate any model defined in the project.

#### Executable
```bash
./scripts/train_eval.py
```

#### Examples
For model examples, please see the README file on each config subdirectory.

Only plots the history:
```bash
./scripts/train_eval.py --config slurm/cpu ds/sahel density/unet2020 density/gt_uniform density/plots  --datadir $HOME --seed 0
```

Examples for Google Colab are provided in the ColabVM notebook.

#### Parameters
| Flag | Description | Mandatory | Default | Example |
|-|-|-|-|-|
| `--config` | Relative paths (without extensions) of configuration files. | Yes | N/A | `--config ds/sahel cover0`<br>Notes:<br>- Files must be stored in `scripts/config`.<br>- Configurations are composed in the specified order.<br>- Configurations with clashing keys are overrided in the specified order, respecting the nested structure. |
| `--train` | Number of epochs to train a new model for. | No | `None` | `--train 100` |
| `--eval` | Evaluate a new or existing model. | No | `False` |  |
| `--seed` | The index of the seed (0 to 9). | No | `0` | `--seed 2`<br>Notes:<br>- Valid values are in `[0-9]` |
| `--erda` | Mount ERDA. | No | `False` |  |
| `--datadir` | Prefix for dataset and model directories. | Conditionally | `None` | `--datadir $HOME`<br>Notes:<br>- It is required if `$DLC_DATA_DIRECTORY` is not set. |
| `--retrain` | Number of epochs to retrain an existing model for. | No | `None` | `--retrain 10` |
| `--suffix` | Model name suffix | No | `""` | `--suffix elu`<br>Notes:<br>- If provided, overrides the composable suffixes from the configuration files. |
| `--dry-run` | Print the commands that would be executed. | No | `False` |  |

### Creating a database

To create a tiles database from a folder of images:

```batch
python3 ${DLC_PROJECT_DIRECTORY}/scripts/create_db.py \
   --img-path="path-that-contains-the-raw-images" \
   --pattern="*.tif" \
   --areas-path="rectangles.gpkg" \
   --polygons-path="polygons.gpkg" \
   --output-path="output-path" \
   --slurm
```

### Creating frames

```batch
python3 ${DLC_PROJECT_DIRECTORY}/scripts/create_frames.py \
   --dataset-name="sahara-sahel" \
   --datasets-path="data/datasets" \
   --img-dir="StackedImages" \
   --output-path="data/datasets/frames" \
   --creators-group="density" \
   --slurm
```
