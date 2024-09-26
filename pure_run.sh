#!/usr/bin/bash

BATCHSIZE=4096
PARTITION=0

EPOCHS=50
RUN_TEST=$@
CUDA="cuda:0"
DATASETDIR="/path/to/dataset"


# baseline: 
# 1. sage-arxiv, 2. sage-product, 3. sage-paper, 4. gcn-arxiv, 5. gat-arxiv
# falcon
# 1. sage-arxiv, 2. sage-product, 3. sage-paper, 4. gcn-arxiv, 5. gat-arxiv
# 6. gcn-product, 7. gat-product, 8. gcn-paper, 9. gcn-paper

if [ $RUN_TEST = "print" ]
then
    echo "-----Printing parameters-----"
    echo "CUDA: ${CUDA}, Batchsize: ${BATCHSIZE}, Epochs: ${EPOCHS}, Partition: ${PARTITION}"

elif [ $RUN_TEST = "b1" ]
then
    python3 sage_dgl_partition.py --use-sample \
        --use-tt \
        --fan-out '3,5,15' \
        --epochs $EPOCHS \
        --device $CUDA \
        --partition $PARTITION \
        --tt-rank "16,16" \
        --p-shapes "50,60,60" \
        --q-shapes "8,4,4" \
        --batch $BATCHSIZE \
        --emb-name "fbtt" \
        --num-layers 3 \
        --num-hidden 256 \
        --dataset ogbn-arxiv \
        --init "noinit" \
        --logging-name "Baseline1-SAGE" \
        --logging

elif [ $RUN_TEST = "b2" ]
then
    echo "-----Running Baseline-----"
    python3 sage_dgl_partition.py --use-sample \
        --use-tt \
        --fan-out '3,5,15' \
        --epochs $EPOCHS \
        --device $CUDA \
        --partition $PARTITION \
        --tt-rank "16,16" \
        --p-shapes "125,140,140" \
        --q-shapes "5,5,4" \
        --emb-name "fbtt" \
        --num-layers 3 \
        --num-hidden 256 \
        --use-tt \
        --dataset "ogbn-products" \
        --batch $BATCHSIZE \
        --init "noinit" \
        --batch-count 1000 \
        --logging-name "Baseline2-SAGE" \
        --logging

elif [ $RUN_TEST = "b3" ]
then
    echo "-----Running Baseline-----"
    python3 sage_dgl_partition.py --use-sample \
        --use-tt \
        --fan-out '3,5,15' \
        --epochs 2 \
        --device $CUDA \
        --batch 4096 \
        --partition $PARTITION \
        --tt-rank "16,16" \
        --p-shapes "400,500,600" \
        --q-shapes "8,4,4" \
        --emb-name "fbtt" \
        --num-layers 3 \
        --num-hidden 256 \
        --dataset "ogbn-papers100M" \
        --init "noinit" \
        --logging-name "Baseline3-SAGE" \
        --logging

elif [ $RUN_TEST = "b4" ]
then
    echo "-----Running Baseline-----"
    python3 gcn_gat_partition.py \
        --use-tt \
        --epochs $EPOCHS \
        --device $CUDA \
        --partition $PARTITION \
        --tt-rank "16,16" \
        --p-shapes "50,60,60" \
        --q-shapes "8,4,4" \
        --emb-name "fbtt" \
        --dataset ogbn-arxiv \
        --use-labels \
        --use-linear \
        --model gcn \
        --init "noinit" \
        --logging-name "Baseline4-GCN" \
        --logging

elif [ $RUN_TEST = "b5" ]
then
    echo "-----Running Baseline-----"
    python3 gcn_gat_partition.py \
        --use-tt \
        --epochs $EPOCHS \
        --device $CUDA \
        --partition $PARTITION \
        --tt-rank "16,16,16" \
        --p-shapes "50,60,60" \
        --q-shapes "8,4,4" \
        --emb-name "fbtt" \
        --dataset ogbn-arxiv \
        --model gat \
        --init "noinit" \
        --logging-name "Baseline5-GAT" \
        --logging

elif [ $RUN_TEST = "f1" ]
then
    echo "-----Running with SAGE (ogbn-arxiv) Final-----"
    python3 sage_dgl_partition.py --use-sample \
        --use-tt \
        --fan-out '3,5,15' \
        --epochs $EPOCHS \
        --device $CUDA \
        --partition $PARTITION \
        --tt-rank "16,16" \
        --p-shapes "50,60,60" \
        --q-shapes "4,4,8" \
        --batch $BATCHSIZE \
        --emb-name "fbtt" \
        --num-layers 3 \
        --num-hidden 256 \
        --dataset ogbn-arxiv \
        --sparse \
        --init "noinit" \
        --batch-count 11000 \
        --cache-size 10 \
        --use-cached \
        --logging-name "Final1-SAGE" \
        --logging


elif [ $RUN_TEST = "f2" ]
then
    echo "-----Running with SAGE (ogbn-products) Final-----"
    python3 sage_dgl_partition.py --use-sample \
        --use-tt \
        --fan-out '3,5,15' \
        --epochs $EPOCHS \
        --device $CUDA \
        --partition $PARTITION \
        --tt-rank "16,16" \
        --p-shapes "125,140,140" \
        --q-shapes "4,5,5" \
        --batch $BATCHSIZE \
        --emb-name "fbtt" \
        --num-layers 3 \
        --num-hidden 256 \
        --dataset ogbn-products \
        --use-cached \
        --sparse \
        --cache-size 100 \
        --init "noinit" \
        --batch-count 14000 \
        --logging-name "Final2-SAGE" \
        --logging

elif [ $RUN_TEST = "f3" ]
then
    echo "-----Running with SAGE (ogbn-papers100M) Final-----"
    python3 sage_dgl_partition.py --use-sample \
        --use-tt \
        --fan-out '3,5,15' \
        --epochs $EPOCHS \
        --device $CUDA \
        --batch $BATCHSIZE \
        --partition $PARTITION \
        --tt-rank "16,16" \
        --p-shapes "400,500,600" \
        --q-shapes "4,4,8" \
        --emb-name "fbtt" \
        --num-layers 3 \
        --num-hidden 256 \
        --dataset "ogbn-papers100M" \
        --init "noinit" \
        --sparse \
        --use-cached \
        --cache-size 5 \
        --batch-count 14000 \
        --logging-name "Final3-SAGE" \
        --logging

elif [ $RUN_TEST = "f4" ]
then
    echo "-----Running with GCN (ogbn-arxiv) Final-----"
    python3 gcn_gat_partition.py \
        --use-tt \
        --epochs $EPOCHS \
        --device $CUDA \
        --partition $PARTITION \
        --tt-rank "16,16" \
        --p-shapes "50,60,60" \
        --q-shapes "4,4,8" \
        --emb-name "fbtt" \
        --dataset ogbn-arxiv \
        --use-labels \
        --use-linear \
        --model gcn \
        --sparse \
        --init "noinit" \
        --use-cache \
        --cache-size 10 \
        --batch-count 14000 \
        --logging-name "Final4-GCN" \
        --logging

elif [ $RUN_TEST = "f5" ]
then
    echo "-----Running with GAT (ogbn-arxiv) Final-----"
    python3 gcn_gat_partition.py \
        --use-tt \
        --epochs $EPOCHS \
        --device $CUDA \
        --partition $PARTITION \
        --tt-rank "16,16" \
        --p-shapes "50,60,60" \
        --q-shapes "4,4,8" \
        --emb-name "fbtt" \
        --dataset ogbn-arxiv \
        --model gat \
        --sparse \
        --init "noinit" \
        --use-cache \
        --cache-size 10 \
        --batch-count 14000 \
        --logging-name "Final5-GAT" \
        --logging

elif [ $RUN_TEST = "f6" ]
then
    echo "-----Running with GCN (ogbn-products) Final-----"
    python3 gcn_gat_partition.py \
        --use-tt \
        --epochs $EPOCHS \
        --device $CUDA \
        --partition $PARTITION \
        --tt-rank "16,16" \
        --p-shapes "125,140,140" \
        --q-shapes "4,5,5" \
        --emb-name "fbtt" \
        --dataset ogbn-products \
        --use-labels \
        --use-linear \
        --model gcn \
        --sparse \
        --init "noinit" \
        --use-cache \
        --cache-size 10 \
        --batch-count 14000 \
        --logging-name "Final6-GCN" \
        --logging

elif [ $RUN_TEST = "f7" ]
then
    echo "-----Running with GAT (ogbn-products) Final-----"
    python3 gcn_gat_partition.py \
        --use-tt \
        --epochs $EPOCHS \
        --device $CUDA \
        --partition $PARTITION \
        --tt-rank "16,16" \
        --p-shapes "125,140,140" \
        --q-shapes "4,5,5" \
        --emb-name "fbtt" \
        --dataset ogbn-products \
        --model gat \
        --sparse \
        --init "noinit" \
        --use-cache \
        --cache-size 10 \
        --batch-count 14000 \
        --logging-name "Final7-GAT" \
        --logging

elif [ $RUN_TEST = "autotuning" ]
then
    echo "-----Running with FBTT (ogbn-products) AUTOTUNE-----"
    python tuning_SAGE.py --use-sample \
        --use-tt \
        --fan-out '3,5,15' \
        --epochs $EPOCHS \
        --device cuda \
        --partition $PARTITION \
        --tt-rank "16,16" \
        --p-shapes "125,140,140" \
        --q-shapes "4,5,5" \
        --batch 4096 \
        --emb-name "fbtt" \
        --num-layers 3 \
        --num-hidden 256 \
        --dataset ogbn-products \
        --init "auto" \
        --batch-count 1000

else
    echo "No test specified. Exiting..."
fi