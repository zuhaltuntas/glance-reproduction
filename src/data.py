"""

Loading the WebKB datasets.

"""

import os
from dataclasses import dataclass

import torch
from torch_geometric.datasets import WebKB
from torch_geometric.utils import remove_self_loops, to_undirected

from src import constants as C

@dataclass(frozen=True)
class GraphTask:
    """One node-classification problem, fully materialised on one device.

    Attributes
    ----------
    x : (N, F)   node features, already degree-augmented
    edge_index : (2, E)   undirected, deduplicated, self-loop free
    y : (N,)   class labels
    train_mask / val_mask / test_mask : (N,)  boolean, one fixed split
    """

    name: str
    split_idx: int
    x: torch.Tensor
    edge_index: torch.Tensor
    y: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor

    add_degree: bool = True
    standardize_degree: bool = True

    @property
    def num_nodes(self) -> int:
        return self.x.size(0)

    @property
    def num_features(self) -> int:
        return self.x.size(1)

    @property
    def num_classes(self) -> int:
        return int(self.y.max().item()) + 1

    @property
    def num_edges(self) -> int:
        """Undirected edge count: (2, E) stores each edge in both directions."""

        return self.edge_index.size(1) // 2


def _node_degrees(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Out-degree of every node. On an undirected graph this is the degree.

    Returns (N, 1) so it can be concatenated directly."""

    degrees = torch.zeros(num_nodes, device=edge_index.device)          # (N,)
    ones = torch.ones(edge_index.size(1), device=edge_index.device)     # (E,)
    degrees.index_add_(0, edge_index[0], ones)                          # (N,)
    return degrees.unsqueeze(1)                                         # (N,) -> (N, 1)

def augment_with_degree(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    standardize: bool = True,
) -> torch.Tensor:
    """Eq. (2): concatenate node degree onto the feature vector.

    Concatenating the raw count lets one column dominate the first linear layer
    as degrees reach ~100. I am keeping the raw option (``--raw_degree``) so the literal
    reading of Eq. (2) is still reachable.
    """

    degrees = _node_degrees(edge_index, x.size(0))                      # (N, 1)

    if standardize:
        degrees = torch.log1p(degrees)                                  # (N, 1)
        # z-score standardization
        std = degrees.std()
        if std > 0:
            degrees = (degrees - degrees.mean()) / std                  # (N, 1)

    return torch.cat([x, degrees], dim=1)                               # (N, F) -> (N, F+1)

def to_glance_edge_index(
    edge_index: torch.Tensor,
    num_nodes: int
) -> torch.Tensor:
    """Turn a raw WebKB edge list into the E that Sec. 2 of the paper defines.

    Sec. 2 defines E as a set of *undirected* edges, while the Geom-GCN files
    are directed, carry self-loops and list some pairs in both directions. The
    result here is symmetric, self-loop free and duplicate free.

    ``to_undirected`` sorts and coalesces on its way.

    edge_index : (2, E_raw)
    returns    : (2, 2 * E_undirected)
    """

    edge_index, _ = remove_self_loops(edge_index)
    return to_undirected(edge_index, num_nodes=num_nodes)

def load_task(
    dataset: C.DatasetName,
    split_idx: int = 0,
    device:torch.device | str = 'cpu',
    add_degree: bool = True,
    root: str = C.DATA_DIR,
) -> GraphTask:
    """Load one WebKB graph.

    Edge preprocessing is ``to_paper_edge_index``; doing it here means the
    model never has to wonder which convention it was handed.
    """

    assert isinstance(dataset, C.DatasetName), (
        f'dataset must be a DatasetName enum, got {type(dataset)!r}'
    )
    assert 0 <= split_idx < C.NUM_PROVIDED_SPLITS, (
        f'split_idx must be in [0, {C.NUM_PROVIDED_SPLITS}), got {split_idx}'
    )

    data = WebKB(root=root, name=dataset.value)[0]

    edge_index = to_glance_edge_index(data.edge_index, data.num_nodes)

    x = data.x
    if add_degree:
        x = augment_with_degree(x, edge_index, standardize=standardize_degree)

    return GraphTask(
        name=dataset.value,
        split_idx=split_idx,
        x=x.to(device),
        edge_index=edge_index.to(device),
        y=data.y.to(device),
        train_mask=data.train_mask[:,split_idx].to(device),
        val_mask=data.val_mask[:,split_idx].to(device),
        test_mask=test_mask[:,split_idx].to(device),
        add_degree=add_degree,
        standardize_degree=standardize_degree
    )
