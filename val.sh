#!/bin/sh
#
#SBATCH --job-name=val
#SBATCH --output=./log/val_out.txt  # output file
#SBATCH -e ./log/val_res.err        # File to which STDERR will be written
#SBATCH --partition=1080ti-long # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=02-01:00:00         # Runtime in D-HH:MM
#SBATCH --mem-per-cpu=4096    # Memory in MB per cpu allocated
#SBATCH --mem=20G

python validate.py
