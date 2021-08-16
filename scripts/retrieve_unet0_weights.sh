#!/usr/bin/env bash

source ${HOME}/miniconda3/etc/profile.d/conda.sh

conda activate retrieve_unet0_weights

export model_dir="${HOME}/Code/KU/Thesis/models"
export PYTHONPATH="${PWD}/..:${PYTHONPATH}"
python retrieve_unet0_weights.py

conda deactivate
