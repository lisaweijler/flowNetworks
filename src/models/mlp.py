import torch
import torch.nn as nn
from einops import rearrange
from src.models.base_model import BaseModel
from src.models.modules import PMA


class MLP(BaseModel):

    def __init__(
        self,
        dim_input,
        dim_hidden,
        hidden_layers,
        dim_output,
        batch_norm: bool = False,
        skip_con: bool = False,
    ):
        super().__init__()

        mlp_layers = [nn.Linear(dim_input, dim_hidden), nn.GELU()]
        if batch_norm:
            mlp_layers.append(nn.BatchNorm1d(dim_hidden))
        for _ in range(0, hidden_layers):
            mlp_layers.extend([nn.Linear(dim_hidden, dim_hidden), nn.GELU()])
            if batch_norm:
                mlp_layers.append(nn.BatchNorm1d(dim_hidden))
        mlp_layers.append(nn.Linear(dim_hidden, dim_output))
        self.mlp = nn.Sequential(*mlp_layers)

        self.skip_con = skip_con

    def forward(self, x):
        bs = None
        if len(x.shape) == 3:
            bs = x.shape[0]
            x = rearrange(x, "b n c -> (b n) c")
        output = self.mlp(x)
        if self.skip_con:
            output += x
        if bs is not None:
            output = rearrange(output, "(b n) c -> b n c", b=bs)
        return output


class LinLayer(BaseModel):

    def __init__(self, dim_input, dim_output, use_bias: bool = True):
        super().__init__()

        self.mlp = nn.Linear(dim_input, dim_output, bias=use_bias)

    def forward(self, x):
        output = self.mlp(x)

        return output


class MLPGlobal(BaseModel):

    def __init__(
        self,
        dim_input,
        dim_hidden,
        hidden_layers,
        dim_output,
        skip_con: bool = False,
        batch_norm: bool = True,
        agg: str = "max",
        fuse: str = "concat",
    ):
        super().__init__()

        self.agg = agg
        self.fuse = fuse
        mlp_layers = [nn.Linear(dim_input, dim_hidden), nn.GELU()]
        if batch_norm:
            mlp_layers.append(nn.BatchNorm1d(dim_hidden))
        for _ in range(0, hidden_layers):
            mlp_layers.extend([nn.Linear(dim_hidden, dim_hidden), nn.GELU()])
            if batch_norm:
                mlp_layers.append(nn.BatchNorm1d(dim_hidden))

        if self.agg == "pma":
            self.agg_pma = PMA(dim=dim_hidden, num_heads=1, num_seeds=1, ln=True)
        if self.fuse == "concat":
            self.dec = nn.Linear(dim_hidden * 2, dim_output)
        elif self.fuse == "add":
            self.dec = nn.Linear(dim_hidden, dim_output)
        else:
            raise ValueError(f"Invalid value for fuse: {self.fuse}")
        self.mlp = nn.Sequential(*mlp_layers)

        self.skip_con = skip_con

    def forward(self, x):
        bs = None
        if len(x.shape) == 3:
            bs = x.shape[0]
            n_events = x.shape[1]
            x = rearrange(x, "b n c -> (b n) c")
        enc_x = self.mlp(x)
        if bs is not None:
            enc_x = rearrange(enc_x, "(b n) c -> b n c", b=bs)

        if self.agg == "max":
            global_x, _ = torch.max(enc_x, dim=1, keepdim=True)
        elif self.agg == "mean":
            global_x = torch.mean(enc_x, dim=1, keepdim=True)
        elif self.agg == "pma":
            global_x = self.agg_pma(enc_x)

        global_x = global_x.repeat(1, enc_x.shape[1], 1)  # use expand instead?

        if self.skip_con:
            enc_x = enc_x + x

        if self.fuse == "concat":

            enc_x = torch.cat((enc_x, global_x), dim=-1)
        elif self.fuse == "add":
            enc_x = enc_x + global_x

        out = self.dec(enc_x)

        return out
