import torch
import torch.nn as nn

class PlayMetadataMLP(nn.Module):
    def __init__(self, input_dim=7, output_dim=1024, gelu=True, dropout=0.1):
        super().__init__()
        if gelu:
            self.act_fn = nn.GELU()
        else:
            self.act_fn = nn.ReLU()
        
        self.dropout = dropout
        self.mlp = torch.nn.Sequential(
            nn.Linear(input_dim, 3072),
            self.act_fn,
            nn.Dropout(p=self.dropout),
            nn.Linear(3072, output_dim),
            nn.LayerNorm(output_dim)
        )
        self.output_dim = output_dim
        
    def forward(self, meta):
        out = self.mlp(meta)
        return out
    
    
class BlitzFormer(nn.Module):
    def __init__(self, num_layers=4, num_heads=4, embedding_dim=512, mlp_dim=256, num_classes=2, gelu=True, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.mlp_dim = mlp_dim
        self.num_classes = num_classes
        self.dropout = dropout
        self.gelu = gelu
        
        self.meta_encoder = PlayMetadataMLP(output_dim=embedding_dim, gelu=self.gelu, dropout=self.dropout)
        self.up_projection = nn.Linear(9, embedding_dim)
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=self.embedding_dim,
                                       nhead=self.num_heads,
                                       dim_feedforward=self.mlp_dim,
                                       dropout=self.dropout,
                                       activation='gelu',    # this gelu is for the MLP inside the transformer encoder layer
                                       norm_first=True,
                                       batch_first=True),
            num_layers=self.num_layers,
            norm=self.layer_norm
        )
        self.blitz_head = nn.Linear(2*embedding_dim, num_classes)
        
    def generate_block_causal_mask(self, num_frames, num_players=22):
        """Returns a causal mask in the shape of [seq_len, seq_len] = [T*22, T*22]"""
        seq_len = num_frames * num_players

        mask = torch.full((seq_len, seq_len), float('-inf'))

        for t in range(num_frames):
            row_start = t * num_players
            row_end = (t+1) * num_players
            for tp in range(t+1):
                col_start = tp * num_players
                col_end = (tp+1) * num_players
                mask[row_start:row_end, col_start:col_end] = 0.0

        return mask # shape: [seq_len, seq_len]
        
    def forward(self, x, play_features):
        """
        Using the same datasets made for the Mamba model, 
        x is our input sequence of spatio-temporal player
        tracking data. It comes in the shape of [B, S, H],
        where H is 22*9. 9 is the number of tracking features
        used per player, with 22 players. We'll need to reshape
        this some first.
        """
        B, S, H = x.shape    
        num_frames = int(S/22)
        
        x = self.up_projection(x)
        
        attn_mask = self.generate_block_causal_mask(num_frames).to(x.device)
        x = self.encoder(x, mask=attn_mask, is_causal=False)  # it is causal, but not traditional-causal, so mark as false
        
        encoded_play_features = self.meta_encoder(play_features)
        encoded_play_features = encoded_play_features.expand(-1, S, -1)
        x = torch.cat([x, encoded_play_features], dim=-1)
        logits = self.blitz_head(x)
        return logits.reshape(B, num_frames, 22, 2)