"""Differentiable logic gate network, as specified by GLANCE's references.

Paper correspondence
--------------------
GLANCE Eq. (6)  : l_i = L(h_i),  z_i = [h_i || l_i]
GLANCE Sec. 3.5 : "a logic layer inspired by [11, 12]", "predefined or learned
                  logical rules (e.g., AND, OR, XOR)", and the example rule
                  "a node belongs to a certain class if it satisfies a
                  combination of features or cluster assignments"

[11] Petersen et al., Deep Differentiable Logic Gate Networks, NeurIPS 2022
[12] Petersen et al., Convolutional Differentiable LGNs, NeurIPS 2024

GLANCE gives no architecture for L, so the architecture is taken from [11]
and [12] rather than invented. Every design decision below cites the section
of those papers it comes from.

Reading guide
-------------
``all_binary_ops``   the 16 relaxations of [11] Table 1, verbatim.
``OP_NAMES``         their names, in the paper's own ID order.
``OP_TEMPLATES``     the same 16 as fillable text, for rule extraction.
``PredicateLayer``   continuous h_i -> Boolean-ish inputs. [11] requires
                     Boolean inputs (it thresholds pixels); GLANCE feeds the
                     layer a continuous h_i, so this is the one adaptation.
``LogicGateLayer``   [11] Sec. 4: fixed random wiring, learned 16-way gate
                     choice, plus the residual initialization of [12] Sec 3.2.
``GroupSum``         [11] Eq. (3): the classification head of both papers.
``LogicLayer``       assembles them behind GLANCE's Eq. (6) interface.

"""

import math
import torch
import torch.nn as nn
from src import constants as C

# [11] Table 1, in the paper's own ID order. ID 3 ('A') is the pass-through
# gate that [12]'s residual initialization favours; PASS_THROUGH_OP records
# that index so the two never drift apart.

OP_NAMES: tuple[str, ...] = (
    '0', 'A and B', 'not(A implies B)', 'A',
    'not(B implies A)', 'B', 'A xor B', 'A or B',
    'not(A or B)', 'not(A xor B)', 'not(B)', 'B implies A',
    'not(A)', 'A implies B', 'not(A and B)', '1',
)

NUM_OPS = len(OP_NAMES)
PASS_THROUGH_OP = 3

# Readable templates for rule extraction. {a}/{b} placeholders rather than
# the letters A/B, because a literal "A" would also match inside "NAND".
OP_TEMPLATES: tuple[str, ...] = (
    'FALSE', '{a} AND {b}', '{a} AND NOT {b}', '{a}',
    'NOT {a} AND {b}', '{b}', '{a} XOR {b}', '{a} OR {b}',
    'NOT ({a} OR {b})', '{a} XNOR {b}', 'NOT {b}', '{a} OR NOT {b}',
    'NOT {a}', 'NOT {a} OR {b}', 'NOT ({a} AND {b})', 'TRUE',
)

def all_binary_ops(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """The 16 real-valued relaxations of [11] Table 1.

    a, b    : (..., G) truth values in [0, 1]
    returns : (..., G, 16)
    """

    ab = a * b

    return torch.stack(
        (
            torch.zeros_like(a),         # 0  False
            ab,                          # 1  A and B
            a - ab,                      # 2  not(A implies B)
            a,                           # 3  A
            b - ab,                      # 4  not(B implies A)
            b,                           # 5  B
            a + b - 2 * ab,              # 6  A xor B
            a + b - ab,                  # 7  A or B
            1 - (a + b - ab),            # 8  not(A or B)
            1 - (a + b - 2 * ab),        # 9  not(A xor B)
            1 - b,                       # 10 not B
            1 - b + ab,                  # 11 A implied by B
            1 - a,                       # 12 not A
            1 - a + ab,                  # 13 A implies B
            1 - ab,                      # 14 not(A and B)
            torch.ones_like(a),          # True
        ),
        dim=-1
    )

class PredicateLayer(nn.Module):
    """Continuous features -> Boolean-ish truth values.

    [11] feeds its network genuine Booleans (pixels thresholded at fixed
    levels, Sec. 6.4). GLANCE's logic layer receives a continuous h_i, so the
    thresholds are learned instead of fixed:

        phi_k(h) = sigmoid( (w_k . h - b_k) / tau_k )

    using tempered sigmoid.

    Cluster assignments arrive already in [0, 1] and are appended unchanged.
    """

    def __init__(
        self,
        input_dim: int,
        num_feature_predicates: int = C.NUM_FEATURE_PREDICATES,
        num_cluster_predicates: int = 0,
        tau: float = C.PREDICATE_TAU
    ) -> None:
        super().__init__()

        assert input_dim > 0, f'input_dim must be positive got {input_dim}'
        assert num_feature_predicates > 0, (
            f'need at least one predicate, got {num_feature_predicates}'
        )
        assert tau > 0, f'tau must be positive, got {tau}'

        self.num_feature_predicates = num_feature_predicates
        self.num_cluster_predicates = num_cluster_predicates
        self.num_predicates = num_feature_predicates + num_cluster_predicates

        self.weight = nn.Parameter(torch.empty(num_feature_predicates, input_dim)) # (P, D)
        self.bias = nn.Parameter(torch.empty(num_feature_predicates))              # (P,)
        self.log_tau = nn.Parameter(torch.empty(num_feature_predicates))           # (P,)

        self.init_params_(tau)

    def init_params_(self, tau: float) -> None:
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        nn.init.constant_(self.log_tau, math.log(tau))

    def forward(
        self,
        h: torch.Tensor,
        cluster_probs: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        h             : (N, D)
        cluster_probs : (N, K) in [0, 1], required iff num_cluster_predicates > 0
        returns       : (N, num_predicates), all in [0, 1]
        """

        tau = self.log_tau.exp().clamp_min(1e-3)               # (P,)
        logits = (
            torch.nn.functional.linear(h, self.weight) - self.bias
        ) / tau                                            # (N, D) -> (N, P)
        predicates = torch.sigmoid(logits)                     # (N, P)

        if self.num_cluster_predicates > 0:
            assert cluster_probs is not None, (
                'this PredicateLayer was build with cluster predicates but '
                'forwawrd recieved no cluster_probs'
            )
            assert cluster_probs.size(1) == self.num_cluster_predicates, (
                f'expected {self.num_cluster_predicates} cluster columns, '
                f'got {cluster_probs.size(1)}'
            )

            predicates = torch.cat(
                [predicates, cluster_probs.clamp(0.0, 1.0)], dim=1,
            )                                                  # (N, P+K)

        return predicates

    def sharpness_loss(self, predicates: torch.Tensor) -> torch.Tensor:
        """Penalise undecided truth values; 0 when every predicate is 0 or 1.

        Scaled to [0, 1].
        """

        # p(1-p)
        return (4.0 * predicates * (1.0 - predicates)).mean()


class LogicGateLayer(nn.Module):
    """One layer of [11]: fixed random wiring, learned choice among 16 gates.

    Each node takes two inputs chosen once, at construction, from a seeded
    generator, and never changed ([11] Sec. 3: "the (weightless) connections
    between neurons are (pseudo-)randomly initialized and remain fixed").
    They live in buffers so a checkpoint restores the same circuit.

    Training learns z in R^16 per node; the relaxed output is the expectation
    over gates, [11] Eq. (2):

        a' = sum_i softmax(z)_i * f_i(a1, a2)

    ``hard=True`` is the discretization of [11] Sec. 4.1: take the argmax
    gate. Both papers report accuracies *after* discretization, so this is the
    inference path, not a debugging aid.
    """

    def __init__(
        self,
        num_inputs: int,
        num_gates: int,
        seed: int = 0,
        residual_init: bool = True,
        residual_logit: float = C.RESIDUAL_INIT_LOGIT,
    ) -> None:
        super().__init__()

        assert num_inputs >= 2, (
            f'a binary gate needs at least 2 inputs to choose from, '
            f'got {num_inputs}'
        )
        assert num_gates > 0, f'num_gates must be positive, got {num_gates}'
        # The reference implementation's own precondition: with fewer than
        # in_dim/2 gates some inputs could never be read at all.
        assert 2 * num_gates >= num_inputs, (
            f'{num_gates} gates cannot cover {num_inputs} inputs; a layer '
            f'needs at least num_inputs/2 gates'
        )

        self.num_inputs = num_inputs
        self.num_gates = num_gates
        self.residual_init = residual_init
        self.residual_logit = residual_logit

        generator = torch.Generator().manual_seed(seed)

        wires = torch.randperm(2 * num_gates, generator=generator) % num_inputs
        wires = torch.randperm(num_inputs, generator=generator)[wires]
        wires = wires.reshape(2, num_gates)

        self.register_buffer('wire_a', wires[0].contiguous())       #(G,)
        self.register_buffer('wire_b', wires[1].contiguous())       #(G,)

        self.gate_logits = nn.Parameter(torch.empty(num_gates, NUM_OPS))
        self.init_params_()

    def init_params_(self) -> None:
        """[12] Sec. 3.2 residual initialization, or [11]'s Gaussian one.

        Residual init puts ~90% of the mass on the pass-through gate 'A'
        (z_3 = 5, the rest 0), which keeps activations from washing out to 0.5
        and gradients from decaying through depth.
        """

        if self.residual_init:
            nn.init.zeros_(self.gate_logits)
            with torch.no_grad():
                self.gate_logits[:, PASS_THROUGH_OP] = self.residual_logit
        else:
            nn.init.normal_(self.gate_logits, mean=0.0, std=1.0)

    def gate_probabilities(self) -> torch.Tensor:
        """(G, 16) -- softmax over operators, [11] Eq. (2)."""
        return torch.softmax(self.gate_logits, dim=-1)

    def forward(
        self, p: torch.Tensor, hard: bool | None = None
    ) -> torch.Tensor:
        """
        p       : (N, num_inputs) truth values in [0, 1]
        hard    : None follows the module's mode -- relaxed while training,
                  discrete in eval.
        returns : (N, num_gates) truth values in [0, 1]
        """

        if hard is None:
            hard = not self.training

        a = p[:, self.wire_a]                                  # (N, G)
        b = p[:, self.wire_b]                                  # (N, G)
        ops = all_binary_ops(a, b)                             # (N, G, 16)

        if hard:
            chosen = self.gate_logits.argmax(dim=-1)           # (G,)
            index = chosen.view(1, -1, 1).expand(p.size(0), -1, 1)
            return ops.gather(2, index).squeeze(2)             # (N, G)

        weights = self.gate_probabilities()                    # (G, 16)
        return (ops * weights.unsqueeze(0)).sum(dim=-1)        # (N, G)

class GroupSum(nn.Module):
    """[11] Eq. (3): partition the outputs of LogicLayer into one group per
    class and count.

        y_c = ( sum_{j in group c} a_j ) / tau

    This is the classification head of both reference papers.

    tau follows the rule of thumb of [12] Sec. A.2 (tau* proportional to the
    square root of the number of outputs per class).
    """

    def __init__(self, num_gates: int, num_classes: int, tau: float | None = None):
        super().__init__()

        assert num_gates % num_classes == 0, (
            f'GroupSum needs equal groups: {num_gates} gates do not divide '
            f'into {num_classes} classes'
        )

        self.num_classes = num_classes
        self.outputs_per_class = num_gates // num_classes
        self.tau = (
            tau if tau is not None
            else max(1.0, math.sqrt(self.outputs_per_class) / 2.0)
        )

    def forward(self, l: torch.Tensor) -> torch.Tensor:
        """l : (N, G) -> (N, C)"""

        grouped = l.view(l.size(0), self.num_classes, self.outputs_per_class)
        return grouped.sum(dim=-1) / self.tau                   # (N, C)

class LogicLayer(nn.Module):
    """L of GLANCE Eq. (6), built from [11]/[12].

    forward returns (l_i, logic_loss) so that GLANCE can form z_i = [h_i||l_i];
    ``class_logits`` additionally offers the GroupSum readout of [11], which
    lets the logic path be scored on its own.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        layer_widths: tuple[int, ...] = C.LOGIC_LAYER_WIDTHS,
        num_feature_predicates: int = C.NUM_FEATURE_PREDICATES,
        num_cluster_predicates: int = 0,
        loss_kind: C.LogicLoss = C.LogicLoss.SATURATION,
        residual_init: bool = True,
        seed: int = 0
    ) -> None:
        super().__init__()

        assert len(layer_widths) >= 1, 'need at least one logic gate layer'
        assert isinstance(loss_kind, C.LogicLoss), (
            f'loss_kind must be a LogicLoss enum, got {type(loss_kind)!r}')

        self.loss_kind = loss_kind
        self.num_classes = num_classes

        self.predicates = PredicateLayer(
            input_dim=input_dim,
            num_feature_predicates=num_feature_predicates,
            num_cluster_predicates=num_cluster_predicates
        )

        # The last layer is padded up to a multiple of num_classes so the
        # GroupSum head always has equal groups.
        widths = list(layer_widths)
        remainder = widths[-1] % num_classes
        if remainder:
            widths[-1] += num_classes - remainder

        sizes = [self.predicates.num_predicates] + widths
        self.gates = nn.ModuleList([
            LogicGateLayer(
                num_inputs=sizes[i],
                num_gates=sizes[i+1],
                seed=seed + i,
                residual_init=residual_init,
            )
            for i in range(len(widths))
        ])

        self.output_dim = widths[-1]
        self.group_sum = GroupSum(self.output_dim, num_classes)

    def forward(
        self,
        h: torch.Tensor,
        cluster_probs: torch.Tensor | None = None,
        hard: bool | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        h       : (N, D)
        hard    : None means "relaxed in train mode, discrete in eval", as in
                  difflogic. Pass True/False only to measure the gap.
        returns : l (N, output_dim) and the scalar L_logic of Eq. (7)
        """

        predicates = self.predicates(h, cluster_probs)         # (N, P)

        l = predicates
        for layer in self.gates:
            l = layer(l, hard=hard)                            # (N, width)

        return l, self._loss(predicates, l)

    def class_logits(self, l: torch.Tensor) -> torch.Tensor:
        """[11] Eq. (3). (N, G) -> (N, C)"""
        return self.group_sum(l)

    def _loss(
        self, predicates: torch.Tensor, l: torch.Tensor
    ) -> torch.Tensor:
        """L_logic. GLANCE Eq. (7) requires the term; [11]/[12] define none."""

        if self.loss_kind is C.LogicLoss.NONE:
            return l.new_zeros(())
        if self.loss_kind is C.LogicLoss.L2:
            return l.pow(2).mean()
        return self.predicates.sharpness_loss(predicates)
