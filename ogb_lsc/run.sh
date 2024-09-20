#!/usr/bin/bash

BATCHSIZE=4096
PARTITION=0

EPOCHS=5
RUN_TEST=$@
CUDA="cuda:0"
DATASETDIR="/path/to/dataset"
PROFILE=""
WORKSPACE='--workspace home-3090'

if [ $RUN_TEST = "help" ]
then
    echo "Usage: ./run.sh [test]"
    echo "Tests:"
    echo "  - mag240m"
    echo "  - pcaqm4m"
    echo "  - wikikg90m"
    exit 0

elif [ $RUN_TEST = "mag240m" ]
then
    echo "-----Running mag240m-----"
    python3 train.py \
        --dataset mag240m \
        --model rgat \
        --device $CUDA \
        --num-layers 2 \
        --num-heads 4 \
        --num-hidden 1024 \
        --epochs $EPOCHS \
        --data-dir None \
        $WORKSPACE

elif [ $RUN_TEST = "pcaqm4m" ]
then
    echo "-----Running pcaqm4m-----"
    python3 train.py \
        --dataset pcaqm4m \
        --model gin \
        --device $CUDA \
        --batch 4096 \
        --epochs $EPOCHS \
        --use-tt

elif [ $RUN_TEST = "wikikg90m" ]
then
    echo "-----Running wikikg90m-----"
    python3 train.py \
        --dataset wikikg90m \
        --model rgat \
        --device $CUDA \
        --num-layers 2 \
        --num-heads 4 \
        --num-hidden 1024 \
        --epochs $EPOCHS \
        --data-dir None \
        $WORKSPACE

else
    echo "No test specified. Exiting..."
fi
