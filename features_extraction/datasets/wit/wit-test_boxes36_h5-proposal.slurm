#!/bin/bash
#SBATCH --ntasks=1 --cpus-per-task=4 --mem=2GB
#SBATCH -p gpu --gres=gpu:titanrtx:1
#SBATCH --time=24:00:00
#SBATCH --job-name="wit"

source scl_source enable devtoolset-7
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/pmh864/libs/anaconda3/lib

bash wit-test_boxes36_h5-proposal.sh
