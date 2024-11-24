import torch.nn as nn
from .no_softmax_att_modules import SABnoSoftmaxNonNeg
from .base_model import BaseModel


class SetReLUFormer(BaseModel):

    def __init__(
        self,
        dim_input,
        dim_hidden,  # dim_hidden must be divisible by num_heads i.e. dim_hidden%num_heads = 0
        num_heads,
        hidden_layers,
        layer_norm,
        dim_output,
    ):
        super(SetReLUFormer, self).__init__()

        enc_layers = [
            SABnoSoftmaxNonNeg(dim_input, dim_hidden, num_heads, ln=layer_norm)
        ]
        for _ in range(0, hidden_layers):
            enc_layers.append(
                SABnoSoftmaxNonNeg(dim_hidden, dim_hidden, num_heads, ln=layer_norm)
            )
        self.enc = nn.Sequential(*enc_layers)

        dec_layers = [nn.Linear(dim_hidden, dim_output)]
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        enc_out = self.enc(x)

        return self.dec(enc_out)
