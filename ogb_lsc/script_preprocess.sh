#!/usr/bin/bash

DATASETDIR=$@

python preprocess_mag.py \
    --rootdir $DATASETDIR \
    --author-output-path ./author.npy \
    --inst-output-path ./inst.npy \
    --graph-output-path ./graph.dgl \
    --graph-as-homogeneous \
    --full-output-path ./full.npy