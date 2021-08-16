#!/usr/bin/env bash

#SBATCH -p image1
#SBATCH --job-name=StackImages
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12000M
#SBATCH --time 20:00:00

echo "Mounting ERDA directory"

source ${DLC_PROJECT_DIR}/scripts/mount_erda.sh

ls -l ${DLC_ERDA_MOUNT}

source ${HOME}/miniconda3/etc/profile.d/conda.sh

conda activate dlc

echo "Running job"

#/usr/bin/time -v \
python3 ${DLC_PROJECT_DIR}/scripts/stack_images.py \
	--input_path "${DLC_ERDA_MOUNT}/datasets/sahel/Images" \
       	--output_path "${DLC_ERDA_MOUNT}/datasets/sahel/NewStackedImages" \
	--pattern "*.tif" \
        -slice_start 0 -slice_end 112 \
	-n_processes 4 \
        -v \
       	> ${HOME}/stack_images.log 2>&1

echo "Finished job"

conda deactivate

source ${DLC_PROJECT_DIR}/scripts/unmount_erda.sh
