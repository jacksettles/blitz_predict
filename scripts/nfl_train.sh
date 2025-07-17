#!/bin/bash
#SBATCH --partition=gpu_p
#SBATCH --job-name=blitz_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=2
#SBATCH --time=3-
#SBATCH --mem=128G
#SBATCH --output=blitz_train.txt

#SBATCH --mail-type=END,BEGIN,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=your-email@something.com  # Where to send mail (change username@uga.edu to your email address)

ml Python/3.11.3-GCCcore-12.3.0


source fbenv/bin/activate

torchrun --standalone --nproc_per_node=gpu ./src/train_model.py --trainfile ./data/processed_data/train.pt --valfile ./data/processed_data/val.pt --save_path test_model --n_layer 4 --d_model 207 --lr 5e-5 --output_size 3 --num_epochs 10 --schedule_free 1