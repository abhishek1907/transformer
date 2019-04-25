#!/bin/sh
#
#SBATCH --job-name=transformer
#SBATCH --output=./log/out.txt  # output file
#SBATCH -e ./log/res.err        # File to which STDERR will be written
#SBATCH --partition=m40-short # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=00-01:00:00         # Runtime in D-HH:MM
#SBATCH --mem-per-cpu=4096    # Memory in MB per cpu allocated
#SBATCH --mem=20G

python validate.py
