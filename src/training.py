"""Training loop, early stopping, checkpointing and the shared CLI.

Paper correspondence
--------------------
Sec. 4.3 : 300 epochs, Adam, lr = 0.005, accuracy on the test set, mean over
           5 runs with different seeds.
Eq. (7)  : L = L_CE + lambda_logic * L_logic, extended in Sec. 2 with
           lambda_struct * L_prune.
"""

import argparse
import json
import os
import platform
import random
import subprocess
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from src import constants as C
from src.data import GraphTask, load_task, majority_class_accuracy
from src.models.glance import GLANCE, GlanceConfig, build_model

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def resolve_device(requested: str) -> torch.device:
    """'auto' -> cuda when available. Always returned, never assumed:
    every function below takes the device as an argument so the whole thing
    still runs (and can be debugged) on CPU."""
    if requested == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(requested)

# --------------------------------------------------------------------------
# Early stopping / best-state tracking
# --------------------------------------------------------------------------

class EarlyStopping:
    """Tracks the best epoch and optionally stops training.

    ``patience <= 0`` disables stopping, which is the paper's setting (Sec. 4.3
    trains a fixed 300 epochs); the best-state tracking still applies.
    """

    def __init__(self, patience: int = 0, min_delta: float = 0.0) -> None:

        assert min_delta >= 0, f'min_delta must be >= 0, got {min_delta}'

        self.patience = patience
        self.min_delta = min_delta

        self.best_score = -float('inf')   # higher is better (accuracy)
        self.best_loss = float('inf')     # tie-break, lower is better
        self.best_epoch = -1
        self.epochs_without_improvement = 0

    def step(self, epoch: int, score: float, loss: float) -> bool:
        """Returns True when this epoch is the new best."""

        improved = score > self.best_score + self.min_delta
        tie_broken = (
            abs(score - self.best_score) <= self.min_delta
            and loss < self.best_loss
        )

        if improved or tie_broken:
            self.best_score = max(score, self.best_score)
            self.best_loss = loss
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
            return True

        self.epochs_without_improvement += 1
        return False

    @property
    def should_stop(self) -> bool:
        if self.patience <= 0:
            return False
        return self.epochs_without_improvement >= self.patience

# --------------------------------------------------------------------------
# Hyperparameters that are not architecture
# --------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    epochs: int = C.EPOCHS
    learning_rate: float = C.LEARNING_RATE
    weight_decay: float = C.WEIGHT_DECAY
    lambda_logic: float = C.LAMBDA_LOGIC
    lambda_prune: float = C.LAMBDA_PRUNE
    patience: int = 0

@dataclass
class RunResult:
    """One (dataset, seed) run. Serialised into outputs/runs/*.json."""
    dataset: str
    seed: int
    split_idx: int
    best_epoch: int
    val_accuracy: float
    train_accuracy: float
    test_accuracy: float
    majority_baseline: float
    checkpoint_path: str | None = None
    history: list[dict] = field(default_factory=list)

# --------------------------------------------------------------------------
# Loss and metrics
# --------------------------------------------------------------------------

def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """logits: (N, C), labels: (N,)"""

    if labels.numel() == 0:
        return float('nan')
    return (logits.argmax(dim=1) == labels).float().mean().item()

def total_loss(
    output,
    labels: torch.Tensor,
    mask: torch.Tensor,
    criterion: nn.Module,
    training_config: TrainingConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eq. (7) plus the L_prune term of Sec. 2.

    Returns (total, cross_entropy) so the two can be logged apart.
    """

    cross_entropy = criterion(output.logits[mask], labels[mask])
    combined = (
        cross_entropy
        + training_config.lambda_logic * output.logic_loss
        + training_config.lambda_prune * output.prune_loss
    )

    return combined, cross_entropy

@torch.no_grad()
def evaluate(
    model: GLANCE,
    task: GraphTask,
    mask: torch.Tensor,
    criterion: nn.Module,
    training_config: TrainingConfig
) -> tuple[float, float]:
    """Returns (accuracy, total loss) on one mask.

    eval() disables dropout -- with p=0.3 on a 183-node graph, measuring with
    it on would report a noisy, pessimistic number. no_grad() keeps the
    evaluation out of the autograd graph.
    """

    model.eval()
    output = model(task.x, task.edge_index)
    loss, _ = total_loss(output, task.y, mask, criterion, training_config)
    return accuracy(output.logits[mask], task.y[mask]), loss.item()

# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------

def train_single_run(
    task: GraphTask,
    model_config: GlanceConfig,
    training_config: TrainingConfig,
    device: torch.device,
    seed: int,
    should_test: bool = False,
    verbose: bool = True,
    log_every: int = 50
) -> tuple[GLANCE, RunResult]:
    """Train one model on one split. Returns the best-validation model."""

    set_seed(seed)

    model = build_model(
        input_dim=task.num_features,
        num_classes=task.num_classes,
        config=model_config
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    stopper = EarlyStopping(patience=training_config.patience)
    best_state = None
    history: list[dict] = []

    for epoch in range(training_config.epochs):
        model.refresh_clusters(task.x, epoch)

        model.train()
        optimizer.zero_grad()
        output = model(task.x, task.edge_index)
        loss, cross_entropy = total_loss(
            output, task.y, task.train_mask, criterion, training_config
        )
        loss.backward()
        optimizer.step()

        train_accuracy = accuracy(
            output.logits[task.train_mask].detach(), task.y[task.train_mask]
        )
        val_accuracy, val_loss = evaluate(
            model, task, task.val_mask, criterion, training_config
        )

        history.append({
            'epoch': epoch,
            'train_loss': loss.item(),
            'cross_entropy': cross_entropy.item(),
            'logic_loss': output.logic_loss.item(),
            'prune_loss': output.prune_loss.item(),
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'val_loss': val_loss,
            'edges_kept': int(output.keep_mask.sum().item()),
        })

        if stopper.step(epoch, val_accuracy, val_loss):
            best_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }

        if verbose and (epoch % log_every == 0 or epoch == training_config.epochs - 1):
            print(
                f'  epoch {epoch:3d} | loss {loss.item():.4f} '
                f'(ce {cross_entropy.item():.4f}) | '
                f'train {train_accuracy:.3f} | val {val_accuracy:.3f}'
            )

        if stopper.should_stop:
            if verbose:
                print(f'  early stop at epoch {epoch}')
            break

    assert best_state is not None, (
        'no epoch was ever recorded as best and training produces no model'
    )

    model.load_state_dict(best_state)

    test_accuracy = None

    if should_test:
        test_accuracy, _ = evaluate(
            model, task, task.test_mask, criterion, training_config
        )

    final_train_accuracy, _ = evaluate(
        model, task, task.train_mask, criterion, training_config
    )

    result = RunResult(
        dataset=task.name,
        seed=seed,
        split_idx=task.split_idx,
        best_epoch=stopper.best_epoch,
        val_accuracy=stopper.best_score,
        train_accuracy=final_train_accuracy,
        test_accuracy=test_accuracy,
        majority_baseline=majority_class_accuracy(task),
        history=history,
    )

    return model, result

# --------------------------------------------------------------------------
# Shared CLI
# --------------------------------------------------------------------------

def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Only settings that change often live on the CLI. Architecture constants
    that define the experiment (hidden width, head count, gate count) stay in
    src/constants.py.
    """

    parser.add_argument(
        '--dataset',
        type=str,
        default=C.DatasetName.CORNELL.value,
        choices=[member.value for member in C.DatasetName],  # choices from the enum
        help='WebKB graph to use.',
    )
    parser.add_argument(
        '--device', type=str, default='auto',
        help="'auto', 'cpu', or e.g. 'cuda:0'.",
    )
    parser.add_argument(
        '--data_root', type=str, default=C.DATA_DIR,
        help='Where the WebKB files live.',
    )
    parser.add_argument(
        '--raw_degree', action='store_true',
        help=(
            'Concatenate the raw degree count (literal Eq. 2) instead of the '
            'standardised one. See src/data.augment_with_degree.'
        ),
    )
    parser.add_argument(
        '--no_degree', action='store_true',
        help='Ablation: skip structural feature augmentation entirely.',
    )

def add_architecture_arguments(parser: argparse.ArgumentParser) -> None:
    """The values Sec. 3 leaves unspecified.

    These are on the CLI even though they are architectural because the
    paper does not fix them.
    """
    parser.add_argument('--hidden_dim', type=int, default=C.HIDDEN_DIM)
    parser.add_argument('--num_layers', type=int, default=C.NUM_LAYERS)
    parser.add_argument('--num_heads', type=int, default=C.NUM_HEADS,
                        help='H in Eq. (4).')
    parser.add_argument('--num_clusters', type=int, default=C.NUM_CLUSTERS,
                        help='k in Eq. (3).')
    parser.add_argument(
        '--logic_widths', type=int, nargs='+', default=list(C.LOGIC_LAYER_WIDTHS),
        help='Widths of the logic gate layers ([11] uses 4-8 equal layers).',
    )
    parser.add_argument(
        '--num_predicates', type=int, default=C.NUM_FEATURE_PREDICATES,
        help='Thresholded propositions fed to the first logic layer.',
    )
    parser.add_argument(
        '--logic_head', type=str, default=C.LogicHead.LINEAR.value,
        choices=[member.value for member in C.LogicHead],
        help=(
            'linear = GLANCE Sec. 3.6; group_sum = Eq. (3) of Petersen et al., '
            'which forces the decision through the logic path.'
        ),
    )
    parser.add_argument(
        '--no_residual_init', action='store_true',
        help='Ablation: Gaussian gate init ([11]) instead of [12] Sec. 3.2.',
    )
    parser.add_argument(
        '--no_cluster_predicates', action='store_true',
        help='Ablation: do not expose cluster membership as logic predicates.',
    )
    parser.add_argument('--prune_percentile', type=float,
                        default=C.PRUNE_PERCENTILE, help='p in Eq. (5).')
    parser.add_argument('--dropout', type=float, default=C.DROPOUT)

def add_ablation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--no_attention', action='store_true',
                        help='Ablation: uniform edge weights, no pruning.')
    parser.add_argument('--no_prune', action='store_true',
                        help='Ablation: keep every edge, attention still reweights.')
    parser.add_argument('--no_cluster', action='store_true',
                        help='Ablation: no cluster centroid concatenation.')
    parser.add_argument('--no_logic', action='store_true',
                        help='Ablation: classifier sees h_i without l_i.')
    parser.add_argument(
        '--logic_loss', type=str, default=C.LogicLoss.SATURATION.value,
        choices=[member.value for member in C.LogicLoss],
        help='Form of L_logic (the paper does not define one).',
    )
    parser.add_argument(
        '--cluster_source', type=str,
        default=C.ClusterSource.INPUT_FEATURES.value,
        choices=[member.value for member in C.ClusterSource],
        help="INPUT_FEATURES reproduces Algorithm 1; HIDDEN is a variant.",
    )

def model_config_from_args(args: argparse.Namespace) -> GlanceConfig:
    """One place where CLI flags become an architecture."""
    return GlanceConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_clusters=args.num_clusters,
        prune_percentile=args.prune_percentile,
        logic_layer_widths=tuple(args.logic_widths),
        num_feature_predicates=args.num_predicates,
        logic_head=C.LogicHead(args.logic_head),
        residual_init=not args.no_residual_init,
        cluster_predicates=not args.no_cluster_predicates,
        dropout=args.dropout,
        logic_loss=C.LogicLoss(args.logic_loss),
        cluster_source=C.ClusterSource(args.cluster_source),
        use_attention=not args.no_attention,
        use_prune=not (args.no_prune or args.no_attention),
        use_cluster=not args.no_cluster,
        use_logic=not args.no_logic
    )

def task_from_args(
    args: argparse.Namespace, split_idx: int, device: torch.device
) -> GraphTask:

    return load_task(
        dataset=C.DatasetName(args.dataset),
        split_idx=split_idx,
        device=device,
        add_degree=not args.no_degree,
        standardize_degree=not args.raw_degree,
        root=args.data_root
    )

def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as handle:
        json.dump(payload, handle, indent=2)
