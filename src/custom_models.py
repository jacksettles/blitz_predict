import torch
import torch.nn as nn


class BlitzLSTM(nn.Module):
    def __init__(self, hidden_dim=512, num_layers=1, num_classes=3, feats_per_agent=9):
        super(BlitzLSTM, self).__init__()
        self.num_classes = num_classes
        
        self.lstm = nn.LSTM(input_size = 23 * feats_per_agent, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 23 * num_classes)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, seq_length, 23*feats_per_agent)
        """
        B, L, N = x.shape      # B=batch, L=seq_len, N=23*feats_per_agent
                
        lstm_out, _ = self.lstm(x)  # [B, L, hidden_dim]
        out = self.fc(lstm_out)     # [B, L, 69]
        out = out.view(B, L, 23, self.num_classes)  # [B, L, 23, 3] — logits per agent
        return out