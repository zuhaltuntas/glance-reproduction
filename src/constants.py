"""

Pure constants and enums for the GLANCE reproduction.

"""

import os
from enum import Enum

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SRC_DIR, os.pardir)

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, 'outputs')
CHECKPOINTS_DIR = os.path.join(OUTPUTS_DIR, 'checkpoints')
RUNS_DIR = os.path.join(OUTPUTS_DIR, 'runs')
DIAGNOSTICS_DIR = os.path.join(OUTPUTS_DIR, 'diagnostics')
FIGURES_DIR = os.path.join(OUTPUTS_DIR, 'figures')

# ---------------------------------------------------------
# Bounded choices -> enums
# ---------------------------------------------------------

class DatasetName(Enum):
    """The three WebKB graphs used in the GLANCE paper (Table 1)."""

    CORNELL = 'Cornell'
    TEXAS = 'Texas'
    WISCONSIN = 'Wisconsin'

class ClusterSource(Enum):
    """Which representation KMeans is run on (Sec. 3.3, Algorithm 1 line 6).

    INPUT_FEATURES
        Paper-faithful. Algorithm 1 clusters ``X'`` (the degree-augmented
        *input* matrix which never changes during training). The assignment
        is therefore computed once and cached.
    HIDDEN
        Recluster the learned hidden embeddings every ``recluster_every``
        epochs. Not what the paper describes; kept as an ablation because
        "adaptive clustering" reads as if it should adapt.
    """

    INPUT_FEATURES = 'input_features'
    HIDDEN = 'hidden'

class LogicHead(Enum):
    """How the logic layer's output reaches the class scores.

    LINEAR
        GLANCE implementation. GLANCE Sec. 3.6 step 5: z_i = [h_i || l_i] goes through 
        "a linear layer for classification".
    GROUP_SUM
        Eq. (3) of Petersen et al. [11]: the logic outputs are partitioned
        into one group per class and counted. Faithful to the reference the
        logic layer comes from, and it forces the decision through the logic
        path instead of letting a linear layer route around it.
    """

    LINEAR = 'linear'
    GROUP_SUM = 'group_sum'

class LogicLoss(Enum):
    """Form of ``L_logic`` (Eq. 7).

    The paper names the term ("enforces logical consistency among
    representations") but never defines it. Both options below are
    assumptions, stated explicitly so they can be argued with.

    SATURATION
        Push the predicate truth values towards {0, 1}.
    L2
        Plain L2 penalty on the logic embedding.
    NONE
        Disable the term (ablation for the logic loss alone).
    """

    SATURATION = 'saturation'
    L2 = 'l2'
    NONE = 'none'

# --------------------------------------------------------------------------
# Hyperparameters fixed by the paper (Sec. 4.3)
# --------------------------------------------------------------------------

EPOCHS = 300                 # Sec. 4.3: "trained for 300 epochs"
LEARNING_RATE = 0.005        # Sec. 4.3: "initial learning rate of 0.005"
NUM_RUNS = 5                 # Sec. 4.3: "repeated five times"

# --------------------------------------------------------------------------
# Hyperparameters the paper does NOT state. Every value below is my choice;
# the comment says where it came from so it is never mistaken for a
# reproduced number.
# --------------------------------------------------------------------------

# Picked by a validation-set sweep over num_layers x dropout x hidden_dim on
# Cornell and Wisconsin, which independently selected the same point. The test
# sets were not consulted.
HIDDEN_DIM = 128
NUM_LAYERS = 1               # 2 propagation steps were consistently ~5 pts worse
DROPOUT = 0.3

# Not swept, just assumptions.
NUM_HEADS = 4                # H in Eq. 4
NUM_CLUSTERS = 8             # k in Eq. 3
PRUNE_PERCENTILE = 0.30      # p in Eq. 5
WEIGHT_DECAY = 5e-4          # the paper names Adam, not its weight decay
LAMBDA_LOGIC = 0.1           # lambda_logic in Eq. 7
LAMBDA_PRUNE = 0.01          # lambda_struct in Sec. 2
RECLUSTER_EVERY = 10         # only used by ClusterSource.HIDDEN

# --------------------------------------------------------------------------
# Logic layer. The shapes are mine; the mechanism is fixed by [11] and [12].
# --------------------------------------------------------------------------

NUM_FEATURE_PREDICATES = 32  # assumption: how many thresholded propositions
PREDICATE_TAU = 0.5          # assumption: tau in [11]'s Eq. 3
LOGIC_LAYER_WIDTHS = (128, 64)   # assumption: [11] uses 4-8 equal-width layers
RESIDUAL_INIT_LOGIT = 5.0    # [12] Sec. 3.2 and Fig. 11: z_3 = 5

# WebKB ships 10 fixed 48/32/20 splits from Geom-GCN [10]. The paper reports
# 5 runs with "different random seeds" without naming the splits, so I pair
# run i with split i.
NUM_PROVIDED_SPLITS = 10

# Numerical floor for divisions by a sum of attention weights.
EPS = 1e-12
