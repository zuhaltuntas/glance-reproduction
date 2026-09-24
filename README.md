# GLANCE — reproduction

PyTorch reproduction of *GLANCE: Graph Logic Attention Network with Cluster
Enhancement for Heterophilous Graph Representation Learning* (Sun et al.,
IJCKG 2025) on the WebKB node-classification task.

    python3 train.py --dataset Cornell --should_test

Work in progress: the model and training loop are complete; the mechanism
diagnostics and test suite are not yet written.

## Setup

    pip install -r requirements.txt

For a CUDA build of PyTorch, install it first from pytorch.org with the
index URL for your CUDA version, then install the rest. Everything runs
on CPU as well -- a full 5-run experiment takes about 10 seconds either way.