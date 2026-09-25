# GLANCE — reproduction

PyTorch reproduction of *GLANCE: Graph Logic Attention Network with Cluster
Enhancement for Heterophilous Graph Representation Learning* (Sun et al.,
IJCKG 2025, LNAI 16297, pp. 51–62) on the node-classification task the paper
evaluates: the WebKB graphs **Cornell**, **Texas** and **Wisconsin**.

> **Status.** The model and the training protocol are complete and run end to
> end. The mechanism diagnostics, the test suite and the ablation tables are
> not written yet — see [Not done yet](#not-done-yet).

## Setup

    pip install -r requirements.txt

Python 3.12. For a CUDA build of PyTorch, install it first from pytorch.org
with the index URL for your CUDA version, then install the rest. Everything
runs on CPU as well -- a full 5-run experiment takes about 10 seconds either
way. The WebKB files download themselves into `data/` on the first run.

## Running

    python3 train.py --dataset Cornell                  # validation only
    python3 train.py --dataset Cornell --should_test    # also score the test set
    python3 train.py --help                             # every flag

The test set is not touched unless `--should_test` is passed. Sweeping
hyperparameters, trying ablations and debugging all happen on validation.

Each mechanism can be removed with a single flag:

    python3 train.py --dataset Cornell --no_logic     --should_test --tag nologic
    python3 train.py --dataset Cornell --no_attention --should_test --tag noattn
    python3 train.py --dataset Cornell --no_cluster   --should_test --tag nocluster
    python3 train.py --dataset Cornell --no_prune     --should_test --tag noprune
    python3 train.py --dataset Cornell --logic_head group_sum --should_test --tag gs

Every run writes `outputs/runs/<dataset>_<tag>.json` with the full argument
namespace, the architecture, the summary and the per-epoch history, so a
result does not depend on shell history.

## Results

Test accuracy (%), five runs, one WebKB split per run, 300 epochs, Adam at
lr 0.005 -- the protocol of Sec. 4.3, with the default linear head.

| Dataset | unmodified | modified | Paper's Table 2 | Majority class |
|---|---|---|---|---|
| Cornell   | 71.4 ± 4.4 | 73.5 ± 3.2 | 85.2 ± 5.6 | 40.5 |
| Texas     | 84.3 ± 2.6 | 85.9 ± 2.6 | 83.7 ± 7.8 | 64.9 |
| Wisconsin | 82.7 ± 3.6 | 84.3 ± 3.0 | 86.7 ± 7.2 | 52.9 |

**unmodified** is the mean ± std over the five runs. **modified** applies the
convention Table 2 of the paper states it uses -- "Modified test accuracy ...
with the lowest test accuracy replaced by the highest" -- and is printed so
the two conventions can be compared directly; it raises the mean and narrows
the spread.

**Majority class** is the accuracy of always predicting the most common
training class. Reporting it alongside model accuracy is therefore important
for interpreting results on these small, imbalanced datasets.

Two caveats on these numbers:

* **They move by 1-3 points between invocations of the same command.**
  Despite fixed seeds, small numerical differences in KMeans can occasionally 
  change assignments across invocations; with only 37–51 test nodes, a single 
  changed prediction corresponds to roughly 2–3 accuracy points.
* **They come from one configuration**, not from a search.

## Pipeline

```
WebKB features + degree          Eq. 2
        ↓
KMeans cluster enhancement       Eq. 3
        ↓
edge attention + pruning         Eqs. 4-5
        ↓
attention-weighted propagation   not specified by the paper
        ↓
             h_i
        ↓
differentiable logic layer       Eq. 6
        ↓
        [h_i || l_i]
        ↓
linear classifier
```


## What the paper leaves open

The paper describes four mechanisms: degree augmentation (Eq. 2), adaptive
clustering (Eq. 3), multi-head edge attention with pruning (Eqs. 4-5) and a
differentiable logic layer (Eq. 6), but does not give the architecture of
the logic layer, the backbone that produces `h_i`, the form of `L_logic`, or
any hyperparameter except epochs and learning rate. Each gap is filled with a
choice documented at the point it is made:

| Quantity | Paper | Here |
|---|---|---|
| Backbone producing `h_i` | not specified | one attention-weighted propagation step with separate transformations for ego and neighbour features, avoiding the shared transformation used in standard GCN-style propagation |
| Logic layer architecture | not specified; cites Petersen et al. | inspired by the cited Petersen et al. formulations: 16 binary Boolean operators with fixed wiring and learned operator selection; adapted to continuous GLANCE embeddings through learnable predicates |
| `L_logic` (Eq. 7) | named, never defined | predicate sharpness -- push the truth values entering the circuit towards {0, 1}; `--logic_loss l2/none` for the alternatives |
| Classification head | "a linear layer" (Sec. 3.6) | linear by default; `--logic_head group_sum` uses Eq. (3) of Petersen et al. instead, which forces the decision through the logic path |
| Hyperparameters | epochs and lr only | defaults in `src/constants.py`, each commented with where it came from |

Two deviations worth knowing about:

* **Degree is standardised** before concatenation (`log1p`, then z-score).
  Eq. 2 concatenates the raw count, but WebKB features are bag-of-words
  indicators in [0, 1] while degrees reach ~100, so the raw column dominates
  the first linear layer. `--raw_degree` restores the literal equation.
* **Clustering runs once, not every epoch.** Algorithm 1 line 6 clusters `X'`
  inside the epoch loop, but `X'` is the input matrix and never changes, so
  those 300 KMeans calls all produce the same partition. Caching it is
  behaviourally identical.

## Layout

    src/
    ├── constants.py          every fixed number, with its source in a comment
    ├── data.py               WebKB loading, Eq. 2, the fixed Geom-GCN splits
    ├── training.py           training loop, early stopping, the shared CLI
    └── models/
        ├── refinement.py     Eq. 4 edge attention, Eq. 5 adaptive pruning
        ├── clustering.py     Eq. 3 cluster enhancement
        ├── logic.py          Eq. 6 logic layer, built to Petersen et al.
        └── glance.py         the pipeline of Fig. 1
    train.py                  run N seeds on one dataset and aggregate

`src/constants.py` separates the numbers the paper fixes (300 epochs, lr
0.005, 5 runs) from the ones it does not, and separates those again into the
ones chosen by a validation sweep and the ones that are a single guess.

## Not done yet

* `analyze.py` -- the mechanism diagnostics. Accuracy cannot tell you whether
  a component did anything, so each one needs measuring directly: attention
  entropy against uniform, homophily before and after pruning, which of the
  16 operators each gate settles on, how committed those gates are, cluster
  purity, and the gap between the relaxed and the discretized circuit.
* `src/viz.py` -- the plots those diagnostics feed.
* `tests/` -- shape and invariant tests, and a check of the 16 operator
  relaxations against the truth table the reference paper prints.
* Checkpointing, so a trained model can be reopened and inspected.
* The ablation tables, once there is something to measure them with.
