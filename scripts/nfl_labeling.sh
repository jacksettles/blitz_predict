#!/bin/bash
#SBATCH --partition=batch
#SBATCH --job-name=nfl_labels
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1-
#SBATCH --mem=64G
#SBATCH --output=nfl_labels.txt

#SBATCH --mail-type=END,BEGIN,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jts75596@uga.edu  # Where to send mail (change username@uga.edu to your email address)

ml Python/3.11.3-GCCcore-12.3.0

cd football/blitzing

source fbenv/bin/activate

python label_engineering.py