from src.models.modules import ISAB, FPSSAB, MABNoAtt
import torch.nn as nn
import torch
from torch_geometric.nn import fps


from src.models.base_model import BaseModel


class SetTransformerNoAtt(BaseModel):

    def __init__(
        self,
        dim_input,
        dim_hidden,  # dim_hidden must be divisible by num_heads i.e. dim_hidden%num_heads = 0
        num_heads,
        hidden_layers,
        layer_norm,
        dim_output,
    ):
        super(SetTransformerNoAtt, self).__init__()

        # one ISAB = 2 MAB
        enc_layers = [
            MABNoAtt(dim_input, dim_input, dim_hidden, num_heads, ln=layer_norm),
            MABNoAtt(dim_hidden, dim_hidden, dim_hidden, num_heads, ln=layer_norm),
        ]
        for _ in range(0, hidden_layers):
            enc_layers.extend(
                [
                    MABNoAtt(
                        dim_hidden, dim_hidden, dim_hidden, num_heads, ln=layer_norm
                    ),
                    MABNoAtt(
                        dim_hidden, dim_hidden, dim_hidden, num_heads, ln=layer_norm
                    ),
                ]
            )
        self.enc = nn.Sequential(*enc_layers)

        dec_layers = [nn.Linear(dim_hidden, dim_output)]
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):

        enc_out = self.enc(x)

        return self.dec(enc_out)


class SetTransformer(BaseModel):
    """
    Set transformer as described in https://arxiv.org/abs/1810.00825
    """

    def __init__(
        self,
        dim_input,
        dim_hidden,  # dim_hidden must be divisible by num_heads i.e. dim_hidden%num_heads = 0
        num_heads,
        num_inds,
        hidden_layers,
        layer_norm,
        dim_output,
    ):
        super(SetTransformer, self).__init__()

        enc_layers = [ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=layer_norm)]
        for _ in range(0, hidden_layers):
            enc_layers.append(
                ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=layer_norm)
            )
        self.enc = nn.Sequential(*enc_layers)

        dec_layers = [nn.Linear(dim_hidden, dim_output)]
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):

        enc_out = self.enc(x)

        return self.dec(enc_out)


class FPSTransformer(BaseModel):

    def __init__(
        self,
        dim_input,
        dim_hidden,  # dim_hidden must be divisible by num_heads i.e. dim_hidden%num_heads = 0
        num_heads,
        fps_ratio,
        layer_norm,
        dim_output,
    ):
        super(FPSTransformer, self).__init__()

        self.fps_ratio = fps_ratio
        self.enc_layer1 = FPSSAB(dim_input, dim_hidden, num_heads, ln=layer_norm)
        self.enc_layer2 = FPSSAB(dim_hidden, dim_hidden, num_heads, ln=layer_norm)
        self.enc_layer3 = FPSSAB(dim_hidden, dim_hidden, num_heads, ln=layer_norm)
        self.enc_layer4 = FPSSAB(dim_hidden, dim_hidden, num_heads, ln=layer_norm)

        dec_layers = [nn.Linear(dim_hidden, dim_output)]
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        # nn.sequential cannot handle multiple inputs -> hardcode layerss
        batch_size = x.shape[0]
        n_events = x.shape[1]
        n_marker = x.shape[2]

        b = torch.tensor(range(batch_size), device=x.device)
        batch_lbl = torch.flatten(b.repeat((n_events, 1)).transpose(0, 1))
        fps_idx = fps(
            x.reshape(batch_size * n_events, n_marker),
            batch=batch_lbl,
            ratio=self.fps_ratio,
            batch_size=batch_size,
        )

        fps_n_points = int(fps_idx.shape[0] / batch_size)

        # correct index since retrun index are for full range -> need to correct to batches
        batch_lbl_sampled = (
            torch.flatten(b.repeat((fps_n_points, 1)).transpose(0, 1)) * n_events
        )
        fps_idx -= batch_lbl_sampled
        # apply model
        fps_idx = fps_idx.reshape(batch_size, fps_n_points).to(torch.int64)

        enc_out = self.enc_layer1(x, fps_idx)
        enc_out = self.enc_layer2(enc_out, fps_idx)
        enc_out = self.enc_layer3(enc_out, fps_idx)
        enc_out = self.enc_layer4(enc_out, fps_idx)

        return self.dec(enc_out)
