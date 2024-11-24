from torch.nn import Module, Linear, GELU
import torch
from einops import rearrange
from torch_geometric.nn import fps
from torch_geometric.nn import GATConv, Sequential, GINConv, GCNConv


from src.models.modules import FPSSAB
from src.models.mlp import MLP


class GIN(Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        hidden_layers,
        batch_norm: bool = False,
    ):  # if we have edge_attr, and sparse tensor then add self loops not supported yet for gat
        super().__init__()

        gin_layers = [
            (
                GINConv(
                    MLP(
                        in_channels,
                        hidden_channels,
                        1,
                        hidden_channels,
                        batch_norm=batch_norm,
                    )
                ),
                "x, edge_index -> x",
            ),
            GELU(),
        ]
        for _ in range(0, hidden_layers):
            gin_layers.append(
                (
                    GINConv(
                        MLP(
                            hidden_channels,
                            hidden_channels,
                            1,
                            hidden_channels,
                            batch_norm=batch_norm,
                        )
                    ),
                    "x, edge_index -> x",
                )
            )
            gin_layers.append(GELU())

        self.pred_head = Linear(hidden_channels, out_channels)
        self.m = Sequential("x, edge_index", gin_layers)

    def forward(self, x, edge_index, batch_size=None):
        out = self.m(x, edge_index)

        return self.pred_head(out)


class GAT(Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        heads,
        hidden_layers,
        edge_dim: int = None,
        add_self_loops: bool = True,
        dropout: float = 0.0,
    ):  # if we have edge_attr, and sparse tensor then add self loops not supported yet for gat
        super().__init__()

        gat_layers = [
            (
                GATConv(
                    in_channels,
                    hidden_channels,
                    heads,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    add_self_loops=add_self_loops,
                ),
                "x, edge_index -> x",
            ),
            GELU(),
        ]
        for _ in range(0, hidden_layers):
            gat_layers.append(
                (
                    GATConv(
                        hidden_channels * heads,
                        hidden_channels,
                        heads,
                        dropout=dropout,
                        edge_dim=edge_dim,
                        add_self_loops=add_self_loops,
                    ),
                    "x, edge_index -> x",
                )
            )
            gat_layers.append(GELU())

        self.pred_head = Linear(hidden_channels * heads, out_channels)
        self.m = Sequential("x, edge_index", gat_layers)

    def forward(self, x, edge_index, batch_size=None):
        out = self.m(x, edge_index)

        return self.pred_head(out)


class GCN(Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        hidden_layers,
        add_self_loops: bool = True,
    ):  # if we have edge_attr, and sparse tensor then add self loops not supported yet for gat
        super().__init__()

        gat_layers = [
            (
                GCNConv(
                    in_channels,
                    hidden_channels,
                    improved=True,
                    add_self_loops=add_self_loops,
                ),
                "x, edge_index -> x",
            ),
            GELU(),
        ]
        for _ in range(0, hidden_layers):
            gat_layers.append(
                (
                    GCNConv(
                        hidden_channels,
                        hidden_channels,
                        improved=True,
                        add_self_loops=add_self_loops,
                    ),
                    "x, edge_index -> x",
                )
            )
            gat_layers.append(GELU())

        self.pred_head = Linear(hidden_channels, out_channels)
        self.m = Sequential("x, edge_index", gat_layers)

    def forward(self, x, edge_index, batch_size=None):
        out = self.m(x, edge_index)

        return self.pred_head(out)


class GINFPSST(Module):
    def __init__(
        self, dim_input, dim_hidden, dim_output, num_heads, fps_ratio, layer_norm
    ):  # if we have edge_attr, and sparse tensor then add self loops not supported yet for gat
        super().__init__()
        self.fps_ratio = fps_ratio
        self.st_layer1 = FPSSAB(
            dim_hidden + dim_input, dim_hidden, num_heads, ln=layer_norm
        )
        self.st_layer2 = FPSSAB(dim_hidden, dim_hidden, num_heads, ln=layer_norm)
        self.st_layer3 = FPSSAB(dim_hidden, dim_hidden, num_heads, ln=layer_norm)

        gin_layers = [
            (
                GINConv(MLP(dim_input, dim_hidden, 1, dim_hidden, batch_norm=False)),
                "x, edge_index -> x",
            ),
            GELU(),
        ]

        self.pred_head = Linear(dim_hidden, dim_output)
        self.m = Sequential("x, edge_index", gin_layers)

    def forward(self, x, edge_index, batch_size):

        n_events = x.shape[0] // batch_size
        n_marker = x.shape[1]

        b = torch.tensor(range(batch_size), device=x.device)
        batch_lbl = torch.flatten(b.repeat((n_events, 1)).transpose(0, 1))
        fps_idx = fps(
            x.reshape(batch_size * n_events, n_marker),
            batch=batch_lbl,
            ratio=self.fps_ratio,
            batch_size=batch_size,
        )

        fps_n_points = int(fps_idx.shape[0] / batch_size)

        # correct index since return index are for full range -> need to correct to batches
        batch_lbl_sampled = (
            torch.flatten(b.repeat((fps_n_points, 1)).transpose(0, 1)) * n_events
        )
        fps_idx -= batch_lbl_sampled
        # apply model
        fps_idx = fps_idx.reshape(batch_size, fps_n_points).to(torch.int64)
        x1 = self.m(x, edge_index)
        x1 = torch.cat((x, x1), dim=-1)
        out = self.st_layer1(rearrange(x1, "(b n) c -> b n c", b=batch_size), fps_idx)
        out = self.st_layer2(out, fps_idx)
        out = self.st_layer3(out, fps_idx)

        return self.pred_head(rearrange(out, "b n c -> (b n) c", b=batch_size))


class GATFPSST(Module):
    def __init__(
        self,
        dim_input,
        dim_hidden,
        dim_output,
        num_heads,
        fps_ratio,
        layer_norm,
        edge_dim: int = None,
        add_self_loops: bool = True,
        dropout: float = 0.0,
    ):  # if we have edge_attr, and sparse tensor then add self loops not supported yet for gat
        super().__init__()
        self.fps_ratio = fps_ratio
        self.st_layer1 = FPSSAB(
            dim_hidden + dim_input, dim_hidden, num_heads, ln=layer_norm
        )
        self.st_layer2 = FPSSAB(dim_hidden, dim_hidden, num_heads, ln=layer_norm)
        self.st_layer3 = FPSSAB(dim_hidden, dim_hidden, num_heads, ln=layer_norm)

        gat_layers = [
            (
                GATConv(
                    dim_input,
                    dim_hidden // num_heads,
                    num_heads,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    add_self_loops=add_self_loops,
                ),
                "x, edge_index -> x",
            ),
            GELU(),
        ]

        self.pred_head = Linear(dim_hidden, dim_output)
        self.m = Sequential("x, edge_index", gat_layers)

    def forward(self, x, edge_index, batch_size):

        n_events = x.shape[0] // batch_size
        n_marker = x.shape[1]

        b = torch.tensor(range(batch_size), device=x.device)
        batch_lbl = torch.flatten(b.repeat((n_events, 1)).transpose(0, 1))
        fps_idx = fps(
            x.reshape(batch_size * n_events, n_marker),
            batch=batch_lbl,
            ratio=self.fps_ratio,
            batch_size=batch_size,
        )

        fps_n_points = int(fps_idx.shape[0] / batch_size)

        # correct index since return index are for full range -> need to correct to batches
        batch_lbl_sampled = (
            torch.flatten(b.repeat((fps_n_points, 1)).transpose(0, 1)) * n_events
        )
        fps_idx -= batch_lbl_sampled
        # apply model
        fps_idx = fps_idx.reshape(batch_size, fps_n_points).to(torch.int64)
        x1 = self.m(x, edge_index)
        x1 = torch.cat((x, x1), dim=-1)
        out = self.st_layer1(rearrange(x1, "(b n) c -> b n c", b=batch_size), fps_idx)
        out = self.st_layer2(out, fps_idx)
        out = self.st_layer3(out, fps_idx)

        return self.pred_head(rearrange(out, "b n c -> (b n) c", b=batch_size))
