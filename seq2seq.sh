#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=""

/usr/bin/env python3 -m translate "$@"

# Running example
#   Train a model: ./seq2seq.sh config/default.yaml --train -v
