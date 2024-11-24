import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# -----------------------------------------non negativity with relu -----------------------------------------------------
class MABnoSoftmaxNonNeg(nn.Module):
    """
    linear complexity based on this paper: https://arxiv.org/abs/2202.08791
    non-neg attention weights but without the cos-based reweighting

    Multihead attention Block (MAB). Performs multihead attention with a residual connection followed by
    a row-wise feedworward layer with a residual connection:

        MAB(X,Y) = LayerNorm(H(X,Y) + rFF(H(X,Y)))

    where

        H(X,Y) = LayerNorm(X + Multihead(X, Y, Y))

    for matrices X, Y. The three arguments for Multihead stand for Query, Value and Key matrices.
    Setting X = Y i.e. MAB(X, X) would result in the type of multi-headed self attention that was used in the original
    transformer paper 'attention is all you need'.
    Furthermore in the original transformer paper a type of positional encoding is used which is not present here.
    """

    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MABnoSoftmaxNonNeg, self).__init__()

        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)

        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def _compute_forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), dim=0)
        K_ = torch.cat(K.split(dim_split, 2), dim=0)
        V_ = torch.cat(V.split(dim_split, 2), dim=0)

        Q_pos = F.relu(Q_)
        K_pos = F.relu(K_)
        A_fake = K_pos.transpose(1, 2).bmm(V_)  # dxd

        row_norm = (
            Q_pos.bmm((torch.sum(K_pos.transpose(1, 2), 2, keepdim=True))) + 0.00001
        )  # nxd x dx1 = nx1 norm constant for each row, add eps to avoid div by zero
        O = Q_pos.bmm(A_fake) / math.sqrt(self.dim_V)
        O = torch.div(O, row_norm)
        O = torch.cat((Q_ + O).split(Q.size(0), 0), 2)

        # A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        # O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, "ln0", None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, "ln1", None) is None else self.ln1(O)

        return O, Q_pos, K_pos, row_norm

    def forward(self, Q, K):
        O, _, _, _ = self._compute_forward(Q, K)

        return O


class SABnoSoftmaxNonNeg(nn.Module):

    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SABnoSoftmaxNonNeg, self).__init__()

        self.mab = MABnoSoftmaxNonNeg(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)


# -----------------------------------------------------------------------------------------------------------------------
