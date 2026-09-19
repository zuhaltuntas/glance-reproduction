"""Dynamic graph refinement: multi-head edge attention + adaptive pruning.

Paper correspondence
--------------------
Eq. (4)  : alpha_ij = 1/H * sum_h sigmoid(w_h^T [x_i || x_j])
Eq. (5)  : Threshold = quantile(alpha, p), edges below it are removed
Sec. 3.4 : "p is a percentile that adjusts dynamically to preserve graph
           connectivity"

PyG convention: ``edge_index[SRC_ROW]`` holds sources, ``edge_index[TGT_ROW]``
holds targets, and a message flows source -> target.
"""

import torch
import torch.nn as nn
from torch_geometric.utils import scatter

from src import constants as C

SRC_ROW = 0
TGT_ROW = 1

class EdgeAttention(nn.Module):
    """Eq. (4). Scores every edge in [0, 1]; does not modify the graph."""

    def __init__(
        self,
        input_dim: int,
        num_heads: int = C.NUM_HEADS,
    ) -> None:
        super().__init__()

        assert input_dim > 0, f'input_dim must be positive, got {input_dim}'
        assert num_heads > 0, f'num_heads must be positive, got {num_heads}'

        self.num_heads = num_heads

        # One Linear with num_heads outputs == num_heads independent w_h of
        # Eq. (4), computed as a single matmul instead of a Python loop.
        self.heads = nn.Linear(2 * input_dim, num_heads)

        self.init_params_()

    def init_params_(self) -> None:
        """Fills the existing tensors in place."""

        nn.init.xavier_uniform_(self.heads.weight)
        nn.init.zeros_(self.heads.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        x          : (N, F)
        edge_index : (2, E)
        returns    : (E,) attention score per edge
        """

        src = edge_index[SRC_ROW]                         # (E,)
        tgt = edge_index[TGT_ROW]                         # (E,)

        pair = torch.cat([x[src], x[tgt]], dim=1)         # (E, 2F)
        scores = torch.sigmoid(self.heads(pair))          # (E, 2F) -> (E, H)

        return scores.mean(dim=1)                         # (E, H) -> (E,)

def prune_edges(
    edge_index: torch.Tensor,
    alpha: torch.Tensor,
    percentile: float,
    keep_best_per_node: bool = True,
) -> torch.Tensor:
    """Eq. (5). Returns a boolean keep-mask of shape (E,).

    Returning a mask rather than a pruned ``edge_index`` lets the caller keep
    ``alpha`` and the mask aligned, and lets the analysis code ask "which
    edges died?" without re-running the model.

    ``keep_best_per_node``: a global quantile can strip every edge
    from a low-scoring node, which in a 183-node graph deletes that node's
    only evidence. I force each node to retain its single highest-scoring
    incoming edge, so pruning never isolates a node.

    The result is directed, even though the input is not. Eq. (4) scores
    ``[x_i || x_j]``, which is not the same vector as ``[x_j || x_i]``, so
    alpha(i,j) != alpha(j,i).
    """

    assert 0.0 <= percentile < 1.0, (
        f'percentile must be in [0,1), got {percentile}'
    )
    assert edge_index.size(1) == alpha.size(0), (
        f'edge_index has {edge_index.size(1)} edges but alpha has {alpha.size(0)}'
    )

    if alpha.numel() == 0:
        return torch.zeros(0, dtype=torch.bool, device=alpha.device)

    if percentile == 0.0:
        return torch.ones_like(alpha, dtype=torch.bool)

    scores = alpha.detach()                                       # (E,)

    threshold = torch.quantile(scores, percentile)
    keep = scores >= threshold

    if keep_best_per_node:
        tgt = edge_index[TGT_ROW]
        num_nodes = int(tgt.max().item()) + 1

        # argmax of alpha per target node, as an edge id
        best_edge = scatter(
            scores, tgt, dim=0, dim_size=num_nodes, reduce='max'
        )                                                         # (N,)

        is_node_best = scores == best_edge[tgt]                   # (E,)
        keep = keep | is_node_best                                # (E,)

    return keep
