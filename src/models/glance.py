"""GLANCE: Graph Logic Attention Network with Cluster Enhancement.

Paper correspondence
--------------------
Fig. 1, Sec. 3.6, Algorithm 1 -- the full pipeline:

    x'_i = [x_i || deg(i)]                     Eq. (2), done in src/data.py
      -> alpha_ij = mean_h sigma(w_h[x_i||x_j])  Eq. (4)
      -> prune edges below quantile(alpha, p)    Eq. (5)
      -> c_k = mean of cluster k                 Eq. (3)
      -> h_i = attention-weighted propagation over the pruned graph
      -> l_i = L(h_i),  z_i = [h_i || l_i]       Eq. (6)
      -> logits = W z_i

Assumptions not stated in the paper
-----------------------------------
* Backbone. Sec. 3.5 says the logic layer acts on "reweighted node features
  h_i" but never defines the message-passing that produces them. I use
  attention-weighted mean aggregation over the pruned graph with the ego term
  kept in a separate weight matrix (h_i and its neighbours do not share W).
  Mixing them is what makes a GCN fail under heterophily, so separating them
  is the minimal defensible choice.
* Where clustering attaches. Sec. 3.6 puts cluster features (stage 3) before
  the logic layer (stage 4); I concatenate the centroid onto the input
  features, so the cluster context is visible to every propagation step.
* L_prune. Sec. 2 lists it but gives no form. I use mean(alpha), an L1
  pull towards sparse attention.
"""

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
from torch_geometric.utils import scatter

from src import constants as C
from src.models.clustering import ClusterEnhancement
from src.models.logic import LogicLayer
from src.models.refinement import SRC_ROW, TGT_ROW, EdgeAttention, prune_edges

@dataclass
class GlanceConfig:
    """Everything that defines the architecture of one run."""

    hidden_dim: int = C.HIDDEN_DIM
    num_layers: int = C.NUM_LAYERS
    num_heads: int = C.NUM_HEADS
    num_clusters: int = C.NUM_CLUSTERS
    prune_percentile: float = C.PRUNE_PERCENTILE
    dropout: float = C.DROPOUT
    logic_loss: C.LogicLoss = C.LogicLoss.SATURATION
    cluster_source: C.ClusterSource = C.ClusterSource.INPUT_FEATURES
    recluster_every: int = C.RECLUSTER_EVERY

    # Logic layer, per Petersen et al. [11, 12]; see src/models/logic.py.
    logic_layer_widths: tuple[int, ...] = C.LOGIC_LAYER_WIDTHS
    num_feature_predicates: int = C.NUM_FEATURE_PREDICATES
    logic_head: C.LogicHead = C.LogicHead.LINEAR
    residual_init: bool = True
    cluster_predicates: bool = True

    # Ablation switches.
    use_attention: bool = True   # False -> uniform edge weights, no pruning
    use_prune: bool = True       # False -> attention reweights but nothing is cut
    use_cluster: bool = True     # False -> no centroid concatenation
    use_logic: bool = True       # False -> classifier sees h_i alone

    def serializable(self) -> dict:
        """asdict() with enums flattened to their values, for JSON/checkpoints."""

        raw = asdict(self)
        return {
            key: (
                value.value if hasattr(value, 'value')
                else list(value) if isinstance(value, tuple)
                else value
            )
            for key, value in raw.items()
        }

@dataclass
class GlanceOutput:
    """Named forward-pass results. A dataclass, not a dict, so a misspelled
    field fails at the call site instead of returning None three layers up."""

    logits: torch.Tensor            # (N, C)
    alpha: torch.Tensor             # (E,)  attention on the original edges
    keep_mask: torch.Tensor         # (E,)  bool, which edges survived pruning
    hidden: torch.Tensor            # (N, hidden_dim)  the "reweighted" h_i
    logic: torch.Tensor             # (N, logic width)  l_i, empty if ablated
    logic_loss: torch.Tensor        # scalar
    prune_loss: torch.Tensor        # scalar
    cluster_assignment: torch.Tensor | None = None   # (N,)

class GLANCE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        config: GlanceConfig | None = None
    ) -> None:
        super().__init__()

        config = config if config is not None else GlanceConfig()

        assert input_dim > 0, f'input_dim must be positive, got {input_dim}'
        assert num_classes > 1, (
            f'num_classes must be at least 2, got {num_classes}'
        )
        assert 0.0 <= config.dropout < 1.0, (
            f'dropout must be in [0,1), got {config.dropout}'
        )
        assert config.num_layers >= 1, (
            f'num_layers must be >=1, got {config.num_layers}'
        )
        assert config.use_attention or not config.use_prune, (
            f'use prune=True requires use_attention=True: pruning is defined '
            'by the attention quantile'
        )

        self.config = config
        self.dropout = nn.Dropout(config.dropout)

        # ---- Eq. (4): edge attention over the input features -------------
        self.edge_attention = EdgeAttention(
            input_dim=input_dim,
            num_heads=config.num_heads
        )

        if config.use_cluster:
            self.cluster = ClusterEnhancement(
                num_clusters=config.num_clusters,
                source=config.cluster_source,
                recluster_every=config.recluster_every
            )
            encoder_input_dim = 2 * input_dim    # [X' || centroid]
        else:
            self.cluster = None
            encoder_input_dim = input_dim

        # Snapshot of the most recent hidden state, used only by
        # ClusterSource.HIDDEN.
        self._last_hidden: torch.Tensor | None = None

        # ---- propagation --------------------------------------------------
        self.encoder = nn.Linear(encoder_input_dim, config.hidden_dim)

        self.self_weights = nn.ModuleList([
            nn.Linear(config.hidden_dim, config.hidden_dim)
            for _ in range(config.num_layers)
        ])
        self.neighbor_weights = nn.ModuleList([
            nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
            for _ in range(config.num_layers)
        ])

        # ---- Eq. (6): logic ----------------------------------------------
        if config.use_logic:
            self.logic = LogicLayer(
                input_dim=config.hidden_dim,
                num_classes=num_classes,
                layer_widths=tuple(config.logic_layer_widths),
                num_feature_predicates=config.num_feature_predicates,
                num_cluster_predicates=(
                    config.num_clusters
                    if (config.use_cluster and config.cluster_predicates) else 0
                ),
                loss_kind=config.logic_loss,
                residual_init=config.residual_init
            )
            classifier_input_dim = config.hidden_dim + self.logic.output_dim
        else:
            self.register_module('logic', None)
            classifier_input_dim = config.hidden_dim

        # GROUP_SUM scores classes inside the logic layer ([11] Eq. 3), so
        # there is no linear classifier to build.
        if config.use_logic and config.logic_head is C.LogicHead.GROUP_SUM:
            self.register_module('classifier', None)
        else:
            self.classifier = nn.Linear(classifier_input_dim, num_classes)

        self.init_params_()

    def init_params_(self) -> None:

        for linear in (self.encoder, self.classifier):
            if linear is None:
                continue
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)

        for linear in list(self.self_weights) + list(self.neighbor_weights):
            nn.init.xavier_uniform_(linear.weight)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)

    def _propagate(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        weights: torch.Tensor,
        layer_idx: int
    ) -> torch.Tensor:
        """One attention-weighted propagation step on the pruned graph.

        h           : (N, H)
        edge_index  : (2, E_kept)
        weights     : (E_kept,)  alpha of the surviving edges
        returns     : (N, H)

            agg_i = sum_j alpha_ji h_j / sum_j alpha_ji
            h_i'  = ReLU(W_self h_i + W_neigh agg_i)
        """

        src = edge_index[SRC_ROW]                               # (E_kept,)
        tgt = edge_index[TGT_ROW]                               # (E_kept,)
        num_nodes = h.size(0)

        messages = weights.unsqueeze(1) * h[src]                # (E_kept, H)
        numerator = scatter(
            messages, tgt, dim=0, dim_size=num_nodes, reduce='sum'
        )                                                       # (N, H)
        denominator = scatter(
            weights, tgt, dim=0, dim_size=num_nodes, reduce='sum'
        ).clamp_min(C.EPS).unsqueeze(1)                         # (N, 1)

        aggregated = numerator / denominator

        return torch.relu(
            self.self_weights[layer_idx](h)
            + self.neighbor_weights[layer_idx](aggregated)
        )                                                       # (N, H)

    def refresh_clusters(self, x:torch.Tensor, epoch: int) -> None:
        """Algorithm 1 line 6.

        Under ClusterSource.INPUT_FEATURES the partition is fitted on ``x``,
        exactly as Algorithm 1 writes it, and only once because ``x`` is
        constant.

        Under ClusterSource.HIDDEN it is fitted on the hidden state of the
        *previous* epoch. Clustering sits upstream of the propagation that
        produces h, so the current h does not exist yet; a one-epoch lag is
        the usual way out of that circularity. Before the first forward there
        is no h at all, so ``x`` bootstraps it.

        Only the cluster *membership* comes from hidden space. ``forward``
        still averages input features within each cluster, which keeps the
        centroid in x's space and the encoder's input dimension fixed at 2F.
        """

        if self.cluster is None:
            return
        if not self.cluster.should_refit(epoch):
            return

        source = x

        if (
            self.config.cluster_source is C.ClusterSource.HIDDEN
            and self._last_hidden is not None
        ):
            source = self._last_hidden

        self.cluster.fit_(source)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        hard: bool | None = None
    ) -> GlanceOutput:
        """
        x          : (N, F)   degree-augmented node features
        edge_index : (2, E)   undirected, both directions present
        hard       : None lets the logic layer follow train/eval mode, which
                     is what difflogic does: relaxed gates while training,
                     discrete gates at evaluation. Pass True/False only to
                     measure the discretization gap.
        """

        num_edges = edge_index.size(1)

        # ---- Eq. (4) ------------------------------------------------------
        if self.config.use_attention:
            alpha = self.edge_attention(x, edge_index)         # (E,)
        else:
            alpha = torch.ones(
                num_edges, device=x.device, dtype=x.dtype
            )                                                  # (E,)

        # ---- Eq. (5) ------------------------------------------------------
        if self.config.use_prune:
            keep_mask = prune_edges(
                edge_index, alpha, self.config.prune_percentile
            )                                                  # (E,) bool
        else:
            keep_mask = torch.ones(
                num_edges, device=x.device, dtype=torch.bool
            )                                                  # (E,)

        kept_edge_index = edge_index[:, keep_mask]             # (2, E_kept)
        kept_alpha = alpha[keep_mask]                          # (E_kept,)

        # ---- Eq. (3) ------------------------------------------------------
        if self.cluster is not None:
            if not self.cluster.is_fitted:
                self.cluster.fit_(x)
            centroids = self.cluster(x)                        # (N, F)
            encoder_input = torch.cat([x, centroids], dim=1)   # (N, 2F)
            cluster_assignment = self.cluster.assignment       # (N,)
        else:
            encoder_input = x                                  # (N, F)
            cluster_assignment = None

        # ---- propagation --------------------------------------------------
        h = self.dropout(encoder_input)                        # (N, F or 2F)
        h = torch.relu(self.encoder(h))                        # (N, F|2F) -> (N, H)

        for layer_idx in range(self.config.num_layers):
            h = self.dropout(h)                                # (N, H)
            h = self._propagate(h, kept_edge_index, kept_alpha, layer_idx)

        # Hand this epoch's embeddings to the next epoch's clustering.
        if (
            self.cluster is not None
            and self.config.cluster_source is C.ClusterSource.HIDDEN
            and self.training
        ):
            self._last_hidden = h.detach()                     # (N, H)

        # ---- Eq. (6) ------------------------------------------------------
        if self.logic is not None:
            cluster_probs = None

            if self.logic.predicates.num_cluster_predicates > 0:
                cluster_probs = torch.nn.functional.one_hot(
                    cluster_assignment, num_classes=self.config.num_clusters
                ).to(h.dtype)

            logic, logic_loss = self.logic(h, cluster_probs, hard=hard)  # (N, G), scalar

            if self.config.logic_head is C.LogicHead.GROUP_SUM:
                logits = self.logic.class_logits(logic)        # (N, G) -> (N, C)
            else:
                z = torch.cat([h, logic], dim=1)               # (N, H+G)
                logits = self.classifier(self.dropout(z))      # (N, H+G) -> (N, C)

        else:
            logic = h.new_zeros((h.size(0), 0))                # (N, 0)
            logic_loss = h.new_zeros(())
            logits = self.classifier(self.dropout(h))          # (N, H) -> (N, C)

        return GlanceOutput(
            logits=logits,
            alpha=alpha,
            keep_mask=keep_mask,
            hidden=h,
            logic=logic,
            logic_loss=logic_loss,
            prune_loss=alpha.mean(),
            cluster_assignment=cluster_assignment
        )

def build_model(
    input_dim: int,
    num_classes: int,
    config: GlanceConfig | None = None
) -> GLANCE:
    return GLANCE(
        input_dim=input_dim,
        num_classes=num_classes,
        config=config if config is not None else GlanceConfig()
    )
