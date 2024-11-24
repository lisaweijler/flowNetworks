from torch.nn import Module, Linear, GELU
import torch
from torch_geometric.nn.pool import ASAPooling
from torch_geometric.nn import (
    GATConv,
    global_mean_pool,
    global_max_pool,
    Sequential,
)


class ASAPoolingModel(Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        heads,
        edge_dim: int = None,
        add_self_loops: bool = True,
        ratio=0.5,
        dropout: float = 0.0,
    ):  # if we have edge_attr, and sparse tensor then add self loops not supported yet for gat
        super().__init__()

        gat_layers1 = [
            (
                GATConv(
                    in_channels,
                    hidden_channels,
                    heads,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    add_self_loops=add_self_loops,
                ),
                "x, edge_index, edge_attr -> x",
            ),
            GELU(),
            (
                GATConv(
                    hidden_channels * heads,
                    hidden_channels,
                    heads,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    add_self_loops=add_self_loops,
                ),
                "x, edge_index, edge_attr -> x",
            ),
        ]

        gat_layers2 = [
            (
                GATConv(
                    hidden_channels * heads,
                    hidden_channels,
                    heads,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    add_self_loops=add_self_loops,
                ),
                "x, edge_index, edge_attr -> x",
            ),
            GELU(),
            (
                GATConv(
                    hidden_channels * heads,
                    hidden_channels,
                    heads,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    add_self_loops=add_self_loops,
                ),
                "x, edge_index, edge_attr -> x",
            ),
        ]

        pred_head_layers = [
            (
                Linear(hidden_channels * heads * 3, hidden_channels),
                "x -> x",
            ),  # node embedings, mean pooling, maxpooling -> x3
            GELU(),
            (Linear(hidden_channels, out_channels), "x -> x"),
        ]

        self.act_f = GELU()
        self.gat1 = Sequential("x, edge_index, edge_attr", gat_layers1)
        self.gat2 = Sequential("x, edge_index, edge_attr", gat_layers2)

        self.pool_layer1 = ASAPooling(hidden_channels * heads, ratio=0.01)
        self.pool_layer2 = ASAPooling(hidden_channels * heads, ratio=0.005)
        self.pred_head = Sequential("x", pred_head_layers)

    def forward(self, x, edge_index, batch_size, edge_attr=None, y=None):
        # node embeddings:
        z1 = self.gat1(x, edge_index, edge_attr)
        z_pooled1, edge_index_pooled1, edge_weight_pooled1, _, _ = self.pool_layer1(
            z1, edge_index
        )
        z2 = self.gat2(self.act_f(z1), edge_index, edge_attr)
        z_pooled11 = self.gat2(z_pooled1, edge_index_pooled1, edge_weight_pooled1)
        z_pooled2, edge_index_pooled2, edge_weight_pooled2, _, _ = self.pool_layer2(
            z_pooled11, edge_index_pooled1
        )

        z_pooled_max = global_max_pool(z_pooled1, batch=None) + global_max_pool(
            z_pooled2, batch=None
        )
        z_pooled_mean = global_mean_pool(z_pooled1, batch=None) + global_mean_pool(
            z_pooled2, batch=None
        )
        # z_pooled = torch.cat((z_pooled_max, z_pooled_mean), dim = 1)
        out = torch.cat(
            (
                z2,
                z_pooled_max.repeat(z2.shape[0], 1),
                z_pooled_mean.repeat(z2.shape[0], 1),
            ),
            dim=1,
        )

        return self.pred_head(out)
