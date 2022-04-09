# SimpleGN
Simple Graph Networks (GN)

This repository contains a simple pytorch implementation of [Graph Nets](https://arxiv.org/abs/1806.01261) (Battaglia et al., 2018) with an efficient batching mechanism implemented to batch graphs.

## Dependencies

Requires installing:
1. `pytorch` (get it [here](https://pytorch.org/))
2. `pytorch_scatter` (get it [here](https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html))

## How-to-use
1. The code blocks for defining Graph Networks are in the `utils.py` file.
2. The `SimpleGN.py` file provides an example for how to define a simple GN class using the modules defined in `utils.py`.
3. The code should work off-the-shelf if this repository is cloned into the top-level `utils` directory of another python project.