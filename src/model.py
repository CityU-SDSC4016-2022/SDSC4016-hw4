import torch
import torch.nn as nn
# import torch.nn.functional as F
from conformer import ConformerBlock


class Classifier(nn.Module):
    def __init__(self, d_model: int, n_spks: int, dim_head: int, heads: int, ff_mult: int, conv_exp: int, conv_k_size: int, dropout: float = 0):
        super().__init__()
        # Project the dimension of features from that of input into d_model.
        self.prenet = nn.Linear(40, d_model)

        # Change Transformer to Conformer.
        # https://arxiv.org/abs/2005.08100

        # Conformer layer
        self.conformer = ConformerBlock(
            dim=d_model,
            dim_head=dim_head,
            heads=heads,
            ff_mult=ff_mult,
            conv_expansion_factor=conv_exp,
            conv_kernel_size=conv_k_size,
            attn_dropout=dropout,
            ff_dropout=dropout,
            conv_dropout=dropout
        )

        # Transformer layer
        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, dim_feedforward=256, nhead=1)
        # self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

        # Project the the dimension of features from d_model into speaker nums.
        self.pred_layer = nn.Sequential(
            # nn.Linear(d_model, d_model),
            # nn.ReLU(),
            nn.Linear(d_model, n_spks),
        )

    def forward(self, mels: torch.Tensor) -> torch.Tensor:
        # out: (batch size, length, d_model)
        out = self.prenet(mels)
        # out: (length, batch size, d_model)
        out = out.permute(1, 0, 2)
        # The encoder layer expect features in the shape of (length, batch size, d_model).
        # out = self.encoder_layer(out)
        # self.encoder(out)
        # out: (batch size, length, d_model)
        out = self.conformer(out)
        out = out.transpose(0, 1)
        # mean pooling
        stats = out.mean(dim=1)

        # out: (batch, n_spks)
        out = self.pred_layer(stats)
        return out
