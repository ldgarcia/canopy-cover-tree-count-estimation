#!/usr/bin/env bash
#SBATCH -p image1
#SBATCH --job-name=SyncModels
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1000M
#SBATCH --time 2:00:00

# Script to sync model files from the cluster into Google Drive.
# Assumes that `rclone` has been configured properly.

echo "Activating conda environment"
source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda activate dlc

rclone -P copy ${DLC_DATA_DIRECTORY}/models/ gdrive:models/

echo "Finishing job"
conda deactivate
