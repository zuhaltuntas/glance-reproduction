"""Adaptive clustering -> cluster-enhanced node features.

Paper correspondence
--------------------
Eq. (3)      : c_k = (1/|C_k|) * sum_{i in C_k} x_i
Sec. 3.3     : KMeans over node features
Algorithm 1  : line 6, "Apply KMeans on X'"

Line 6 clusters ``X'``, the degree-augmented *input* matrix, inside the epoch
loop. ``X'`` does not change between epochs, so the assignment it produces is
identical every time: running KMeans 300 times computes the same partition
300 times. Therefore I compute it once and cache it (``ClusterSource.
INPUT_FEATURES``), which is behaviourally identical to the paper and ~300x
cheaper. ``ClusterSource.HIDDEN`` is the variant where clustering genuinely
adapts, re-fitted on the previous epoch's learned embeddings; it is an
ablation, not the paper. This module only decides *when* to refit; which
tensor gets clustered is chosen by ``GLANCE.refresh_clusters``.

Centroids are recomputed from the assignment on every forward rather
than cached from sklearn: one source of truth, no second buffer to
keep in sync, and the mean stays a differentiable function of its
members should a live tensor ever be passed in.
"""

import torch
import torch.nn as nn
from sklearn.cluster import KMeans

from src import constants as C

class ClusterEnhancement(nn.Module):
    """Maps each node to the centroid of its cluster.

    Returns centroids rather than cluster ids so the caller can concatenate
    a vector.
    """

    def __init__(
        self,
        num_clusters: int = C.NUM_CLUSTERS,
        source: C.ClusterSource = C.ClusterSource.INPUT_FEATURES,
        recluster_every: int = C.RECLUSTER_EVERY,
        seed: int = 0,
    ) -> None:
        super().__init__()

        assert num_clusters >= 1, (
            f'num_clusters must be >= 1, got {num_clusters}'
        )
        assert isinstance(source, C.ClusterSource), (
            f'source must be a ClusterSource enum, got {type(source)!r}'
        )
        assert recluster_every >= 1, (
            f'recluster_every must be >= 1, got {recluster_every}'
        )

        self.num_clusters = num_clusters
        self.source = source
        self.recluster_every = recluster_every
        self.seed = seed

        # Buffer, not attribute: the assignment is part of the fitted model
        # and must travel with the checkpoint. Declared as None so the key
        # exists before the first fit.
        self.register_buffer('assignment', None)

    @property
    def is_fitted(self) -> bool:
        return self.assignment is not None

    def should_refit(self, epoch: int) -> bool:
        """Whether ``fit_`` needs to run at the start of ``epoch``."""

        if not self.is_fitted:
            return True
        if self.source is C.ClusterSource.INPUT_FEATURES:
            return False
        return epoch % self.recluster_every == 0

    @torch.no_grad()
    def fit_(self, features: torch.Tensor) -> None:
        """Run KMeans and store the assignment. In place.

        features : (N, D)
        """

        num_nodes = features.size(0)
        effective_clusters = min(self.num_clusters, num_nodes)

        if effective_clusters == 1:
            labels = torch.zeros(
                num_nodes, dtype=torch.long, device=features.device
            )                                                           # (N,)
        else:
            kmeans = KMeans(
                n_clusters=effective_clusters,
                random_state=self.seed,
                n_init=10,
            )
            numpy_labels = kmeans.fit_predict(
                features.detach().cpu().numpy()
            )                                                           # (N,)
            labels = torch.as_tensor(
                numpy_labels, dtype=torch.long, device=features.device
            )                                                           # (N,)

        self.assignment = labels

    def forward(self, features:torch.Tensor) -> torch.Tensor:
        """Eq. (3): each node receives its cluster's mean feature vector.

        features : (N, D)
        returns  : (N, D) centroid of the cluster each node belongs to
        """

        assert self.is_fitted, (
            'fit_() must be called before forward(); the model does this in '
            'its own forward, so seeing this means fit_ was bypassed.'
        )
        assert features.size(0) == self.assignment.size(0), (
            f'features has {features.size(0)} nodes but the cached assignment '
            f'has {self.assignment.size(0)}'
        )

        num_clusters = int(self.assignment.max().item()) + 1

        sums = torch.zeros(
            num_clusters, features.size(1),
            dtype=features.dtype, device=features.device
        )                                                      # (K, D)
        counts = torch.zeros(
            num_clusters, dtype=features.dtype, device=features.device
        )                                                      # (K,)

        sums.index_add_(0, self.assignment, features)          # (K, D)
        counts.index_add_(
            0,
            self.assignment,
            torch.ones(
                features.size(0), dtype=features.dtype, device=features.device
            )
        )                                                      # (K,)

        centroids = sums / counts.clamp_min(1.0).unsqueeze(1)  # (K, D)

        return centroids[self.assignment]                      # (K, D) -> (N, D)
