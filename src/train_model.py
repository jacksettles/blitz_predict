import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from schedulefree import AdamWScheduleFree

import argparse
import time
import os
import sys
import numpy as np
from typing import Optional
from tqdm import tqdm

from mamba_models import ModelArgs, Mamba
from nfl_data import NFLDataset

parser = argparse.ArgumentParser()

parser.add_argument("--seed", default=3455, type=int, help="Seed for setting randomness")
parser.add_argument("--trainfile", default="", type=str, help="Path to training data")
parser.add_argument("--valfile", default="", type=str, help="Path to val data")
parser.add_argument("--save_path", type=str, help="Name of the file to save the model to without file extensions")

parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
parser.add_argument("--d_model", type=int, default=207, help="Hidden/embedding dimensions")
parser.add_argument("--n_layer", type=int, default=6, help="Number of residual blocks")
parser.add_argument("--d_state", type=int, default=16, help="")
parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train for")
parser.add_argument("--output_size", type=int, default=3, help="Number of classes. 3 for us: 0 = not blitzing, 1 = blitzing, 2 = on offense or the football")
parser.add_argument("--schedule_free", type=int, default=1, help="0 if you want to use an optimizer with a cosine scheduler, 1 if you want to use schedulefree from Meta FAIR.")
parser.add_argument("--break_value", type=int, default=5, help="Break training if model scores a worse eval metric this many times.")


def ddp_setup():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_process_group(backend="nccl")


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_data: DataLoader,
        val_data: DataLoader,
        criterion: nn.CrossEntropyLoss,
        optimizer: optim.Optimizer,
        snapshot_path: str,
        num_epochs: int,
        output_size: int,
        break_value: int,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.device = torch.device(f'cuda:{self.gpu_id}')
        self.model = model.to(self.device)
        self.train_data = train_data
        self.val_data = val_data
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        if os.path.exists(f"saved_models/{self.snapshot_path}.pt"):
            print(f"Rank {self.gpu_id} loading snapshot")
            self._load_snapshot(f"saved_models/{self.snapshot_path}.pt")

        self.model = DDP(self.model, device_ids=[self.gpu_id])
        self.num_epochs = num_epochs
        self.output_size = output_size
        self.break_value = break_value
        
    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, weights_only=False, map_location=loc)
        self.model = snapshot["MODEL"].to(self.device)
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")
        
        
    def _save_snapshot(self, epoch, args):
        snapshot = {
            "ARGS": args.__dict__,
            "MODEL": self.model.module.to('cpu'),
            "EPOCHS_RUN": epoch,
        }
        save_dir = "../saved_models"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_file = f"{save_dir}/{self.snapshot_path}.pt"
        torch.save(snapshot, save_file)
        print(f"Epoch {epoch} | Training snapshot saved at {save_file}")
        
    def _run_batch(self, batch):
        # batch shape [B, L, D] - targets shape [B, 23]
        features, targets = batch[0].to(self.device), batch[1].to(self.device).long()

        total_preds = targets.size(0) * targets.size(1) * targets.size(2)
        
        outputs = self.model(features)
        reshaped_outputs = outputs.view(-1, self.output_size)
        reshaped_targets = targets.view(-1)
        loss = self.criterion(reshaped_outputs, reshaped_targets)
        loss.backward()
        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()
        return loss, total_preds
        
        
    def _run_epoch(self, epoch):
        self.train_data.sampler.set_epoch(epoch)
        
        train_epoch_loss = torch.tensor(0.0, device=self.device)
        train_epoch_preds = 0
        for batch in tqdm(self.train_data, desc="Training", total=len(self.train_data)):
            batch_loss, batch_num_preds = self._run_batch(batch)
            train_epoch_loss += batch_loss
            train_epoch_preds += batch_num_preds
        avg_train_loss = train_epoch_loss.item() / train_epoch_preds
        return avg_train_loss
            
    
    def _run_eval(self):
        self.model.eval()
        if self.scheduler is None:
            # This means you are using the schedulefree AdamW optimizer, which has to be set to eval mode
            self.optimizer.eval()
        total_val_ce_loss = torch.tensor(0.0, device=self.device)
        total_preds = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_data, desc="Eval", total=len(self.val_data)):
                # batch shape [B, L, D] - targets shape [B, L, 23]
                features, targets = batch[0].to(self.device), batch[1].to(self.device).long()
                
                outputs = self.model(features)
                reshaped_outputs = outputs.view(-1, self.output_size)
                reshaped_targets = targets.view(-1)
                
                loss = self.criterion(reshaped_outputs, reshaped_targets)
                
                total_val_ce_loss += loss
                total_preds += targets.size(0) * targets.size(1) * targets.size(2)
                
        avg_ce_loss = total_val_ce_loss.item() / total_preds
        return avg_ce_loss
        
        
    def _log_progress(self, epoch=None, avg_train_loss=None, avg_val_ce=None, lr=None, file_name=None):
        # Ensure the directory exists
        directory = "../progress_outputs/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = os.path.join(directory, f"{file_name}_progress_log.txt")

        with open(file_path, "a") as f:
            progress = f"Epoch: {epoch}, Train Avg. CE Loss: {avg_train_loss}, Val. Avg. CE Loss: {avg_val_ce}, Learning rate: {lr}\n"
            f.write(progress)
        
                
    def train(self, args):
        # Initial eval to get baselines
        best_ce_loss = self._run_eval()
        
        for epoch in range(self.num_epochs):
            self.model.train()
            if self.scheduler is None:
                self.optimizer.train()
            avg_train_loss = self._run_epoch(epoch)
            
            # Eval after each epoch
            avg_ce_loss = self._run_eval()
            if self.scheduler is not None:
                current_lr = self.scheduler.get_last_lr()[0]
            else:
                current_lr = self.optimizer.param_groups[0]['scheduled_lr']
            
            if self.gpu_id == 0:
                self._log_progress(epoch=(epoch+1),
                                   avg_train_loss=avg_train_loss,
                                   avg_val_ce=avg_ce_loss,
                                   lr=current_lr,
                                   file_name=self.snapshot_path)
                
                if avg_ce_loss < best_ce_loss:
                    best_ce_loss = avg_ce_loss
                    self._save_snapshot(epoch, args)
                    self.model.module.to(self.device)
        
        
def prepare_dataloader(dataset: Dataset, batch_size: int=1, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=False,
        shuffle=shuffle,
#         collate_fn=single_collate,
        num_workers=0,
        sampler=DistributedSampler(dataset)
    )
        
    
def lr_lambda(current_step: int, warmup_steps):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return 1.0

                
def make_scheduler(optimizer, args, epoch_length):
    warmup_steps = int(epoch_length*0.05)
    warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: lr_lambda(step, warmup_steps))
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=epoch_length*8, T_mult=2, eta_min=1e-9)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer,
                                                schedulers=[warmup_scheduler, cosine_scheduler],
                                                milestones=[warmup_steps])
    return scheduler
    
    
def load_train_objects(args):
    model_args = ModelArgs(d_model=args.d_model,
                           n_layer=args.n_layer,
                           vocab_size=args.output_size,
                           d_state=args.d_state,
                           pad_vocab_size_multiple=3 # 3 classes, so don't pad the 'vocab size' aka the classifier output
                          )
    
    model = Mamba(model_args)
    
    train_data = NFLDataset(args.trainfile)
    val_data = NFLDataset(args.valfile)
    train_data = prepare_dataloader(train_data)
    val_data = prepare_dataloader(val_data)
    
    num_batches = len(train_data)
    
    criterion = nn.CrossEntropyLoss(reduction="sum")
    
    if args.schedule_free == 0:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        scheduler = make_scheduler(optimizer, args, num_batches)
    else:
        warmup = 0.05*num_batches
        decay = 0.5
        optimizer = AdamWScheduleFree(model.parameters(), lr=args.lr, warmup_steps=warmup, weight_decay=decay)
        scheduler = None
    
    return model, train_data, val_data, criterion, optimizer, scheduler
        
def main(args):
    ddp_setup()
        
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    model, train_data, val_data, criterion, optimizer, scheduler = load_train_objects(args)
    
    trainer = Trainer(
        model = model,
        train_data = train_data,
        val_data = val_data,
        criterion = criterion,
        optimizer = optimizer,
        scheduler = scheduler,
        snapshot_path = args.save_path,
        num_epochs = args.num_epochs,
        output_size = args.output_size,
        break_value = args.break_value
    )
    trainer.train(args)
    
    destroy_process_group()
    
    
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)