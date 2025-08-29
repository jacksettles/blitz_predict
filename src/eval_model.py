import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformer_models import BlitzFormer
from mamba_models import Mamba
from nfl_data import NFLDataset
import argparse
import sys
from tqdm import tqdm
import copy
import thop
from thop import profile, clever_format
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser()

parser.add_argument("--seed", type=int, default=3435, help="Seed for random numbers - good for reproducibility.")
parser.add_argument("--model", type=str, default="saved_models/transformer_v1.pt", help="Path to the model you want to evaluate.")
parser.add_argument("--val_data", type=str, default="data/processed_data/transformer/test.pt", help="Path to validation dataset.")
parser.add_argument("--eval_type", type=str, default="all", choices=["all", "snap"], help="All for all time steps, snap for just at snap")

if torch.cuda.is_available():
    device = "cuda"
    print("Cuda is available. Using GPU.")
else:
    device = "cpu"
    print("Cuda is not available. Using CPU.")


def load_model(path):
    """
    This function loads the model.
    
    Args:
        path (str): Path to the saved model you wish to evaluate.
        
    Returns:
        nn.Module: The saved nn.Module that is your model. 
                   May have saved it as a dictionary with the key 'MODEL'.
    """
    loc = "cuda:0"
    snapshot = torch.load(path, map_location=loc)

    model = BlitzFormer()
    model.load_state_dict(snapshot["MODEL_STATE"])
    return model.to(device)


def load_data(path):
    """
    This function loads the data you want to use to evaluate your model on.
    
    Args:
        path (str): Path to the dataset
        
    Returns:
        DataLoader: A PyTorch DataLoader object that allows you to 
                    iterate over the sequences in the dataset.
    """
    data = NFLDataset(data_path=path)
    dl = prepare_dataloader(data)
    return dl
    
    
def collate_fn(batch):
    xs, metadatas, ys = zip(*batch)
    metadata = torch.stack(metadatas, dim=0)
    lens = torch.tensor([x.size(0) for x in xs], dtype=torch.long)
    T_max = int(lens.max())

    xs = pad_sequence(xs, batch_first=True, padding_value=0.0)   # [B, T_max, D]
    ys = pad_sequence(ys, batch_first=True, padding_value=0.0)   # [B, T_max, 22]

    # time mask: True for real steps
    mask = torch.arange(T_max).unsqueeze(0) < lens.unsqueeze(1)  # [B, T_max] (bool)
    return xs, metadata, ys, mask, lens

    
def prepare_dataloader(dataset: Dataset, batch_size: int=1, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=False,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=1
    )
    

def evaluate(model: nn.Module, data: DataLoader, model_name: str, snap: bool=False):
    total_val_ce_loss = torch.tensor(0.0, device=device)
    total_preds = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    all_labels = []
    all_probs  = []
    all_preds  = []

    with torch.no_grad():
        for batch in tqdm(data, desc="TEST_SET", total=len(data)):
            
            features = batch[0].to(device)
            B, S, H = features.shape
            n_frames = int(S/22)
            play_level_features = batch[1].to(device)
            targets = batch[2].to(device).long()
            targets = targets.reshape(features.size(0),-1, 22) # comment out for mamba
            time_mask = batch[3].to(device)
            time_mask = time_mask.reshape(features.size(0),-1, 22) # comment out for mamba

            outputs = model(features, play_level_features) # [B, S, 22, 2]
            if snap:
                outputs = outputs[:, -1, :11, :]
                targets = targets[:, -1, :11]
                valid = time_mask[:, -1, :11]
                flag = "_AT_SNAP_" # for output chart name
            else:
                outputs = outputs[:, :, :11, :] # only the defense, so [B, S, 11, 2]
                targets = targets[:, :, :11]
                valid = time_mask[:, :, :11]
                flag = "_"
                
            reshaped_outputs = outputs.reshape(-1, 2)
            reshaped_targets = targets.reshape(-1)
            flat_mask = valid.reshape(-1)

            loss = criterion(reshaped_outputs[flat_mask], reshaped_targets[flat_mask])

            total_val_ce_loss += loss
            total_preds += flat_mask.sum().item()

            preds = reshaped_outputs.argmax(dim=1)
            probs = torch.softmax(reshaped_outputs, dim=1)[:, 1]

            all_labels.append(reshaped_targets.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)
    y_pred = np.concatenate(all_preds)

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    avg_ce_loss = total_val_ce_loss.item() / total_preds
    metrics_text = (
        f"Accuracy:  {acc:.2f}\n"
        f"Precision: {precision:.2f}\n"
        f"Recall:    {recall:.2f}\n"
        f"F1 Score:  {f1:.2f}\n"
        f"AUC:       {auc:.2f}\n"
        f"Avg Loss:  {avg_ce_loss:.4f}"
    )

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    plt.plot(fpr, tpr, label=f"Test set AUC = {auc:.2f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()

    # Place text box
    plt.gca().text(
        0.6, 0.2, metrics_text, transform=plt.gca().transAxes,
        fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    save_dir = f"./roc_curves/{model_name}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(f"{save_dir}/TEST_SET{flag}roc_curve.png")
    plt.clf()


def main(args):
    print("Loading model!")
    model = load_model(args.model)
    print("Loading data!")
    data = load_data(args.val_data)
    
    test_batch = next(iter(data))
    test_seq = test_batch[0].to(device)
    test_metadata = test_batch[1].to(device)
    
    # Profile the model to count MACs
    print("Profiling model first!")
    model_for_prof = copy.deepcopy(model).eval().to(device)
    macs, params = profile(model_for_prof, inputs=(test_seq, test_metadata))
    macs, params = clever_format([macs, params], "%.3f")
    print(f"MACs: {macs}, Parameters: {params}\n\n")
    
    model_name = args.model.split('/')[-1].split('.')[0]
    print("Running eval!")
    if args.eval_type == "all":
        evaluate(model, data, model_name)
    elif args.eval_type == "snap":
        evaluate(model, data, model_name, snap=True)
    
if __name__ == "__main__":
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    main(args)