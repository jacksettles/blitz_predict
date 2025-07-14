#!/bin/bash
#SBATCH --partition=gpu_p
#SBATCH --job-name=nfl_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=2
#SBATCH --time=3-
#SBATCH --mem=128G
#SBATCH --output=nfl_train_new_labels.txt

#SBATCH --mail-type=END,BEGIN,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jts75596@uga.edu  # Where to send mail (change username@uga.edu to your email address)

ml Python/3.11.3-GCCcore-12.3.0

cd football/blitzing

source fbenv/bin/activate

torchrun --standalone --nproc_per_node=gpu train_model.py --trainfile /scratch/jts75596/fb/processed_data/train_2.pt --valfile /scratch/jts75596/fb/processed_data/val_2.pt --save_path sched_free_model3 --n_layer 4 --d_model 207 --lr 5e-5 --output_size 3 --num_epochs 10 --schedule_free 1