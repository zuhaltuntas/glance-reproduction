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
            return ops.gather(2, index).squeeze(2)              # (N, G)

        weights = self.gate_probabilities()                    # (G, 16)
        return (ops * weights.unsqueeze(0)).sum(dim=-1)        # (N, G)
