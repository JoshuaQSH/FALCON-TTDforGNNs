#!/usr/bin/bash

BATCHSIZE=4096
PARTITION=0

EPOCHS=5
RUN_TEST=$@
CUDA="cuda:0"
DATASETDIR="/path/to/dataset"
PROFILE=""
WORKSPACE='--workspace home-3090'

## Profiling Parameters
dram=(
    "dram__bytes.sum.peak_sustained",
    "dram__bytes.sum.per_second",
    "dram__bytes_read.sum",
    "dram__bytes_read.sum.pct_of_peak_sustained_elapsed",
    "dram__cycles_active.avg.pct_of_peak_sustained_elapsed",
    "dram__cycles_elapsed.avg.per_second",
    "dram__sectors_read.sum",
    "dram__sectors_write.sum")

memory=(
    "sass__inst_executed_shared_loads",
    "sass__inst_executed_shared_stores",
    "smsp__inst_executed_op_ldsm.sum",
    "smsp__inst_executed_op_shared_atom.sum",
    "smsp__inst_executed_op_global_red.sum",                        
    "smsp__inst_executed_op_ldsm.sum.pct_of_peak_sustained_elapsed",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_atom.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "sass__inst_executed_global_loads",
    "sass__inst_executed_global_stores",
    "sm__sass_inst_executed_op_ldgsts_cache_bypass.sum",
    "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum",
    "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum",
    "smsp__sass_inst_executed_op_memory_128b.sum",
    "smsp__sass_inst_executed_op_memory_16b.sum",
    "smsp__sass_inst_executed_op_memory_32b.sum",
    "smsp__sass_inst_executed_op_memory_64b.sum")

inst=(
    "sm__inst_executed_pipe_cbu_pred_on_any.avg.pct_of_peak_sustained_elapsed",
    "sm__inst_executed_pipe_fma_type_fp16.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_ipa.avg.pct_of_peak_sustained_elapsed",
    "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed",
    "sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active")

quest1=(
    "dram__bytes.sum.peak_sustained",
    "dram__bytes.sum.per_second",
    "dram__bytes_read.sum",
    "dram__sectors_read.sum",
    "dram__sectors_write.sum",
    "sass__inst_executed_shared_loads",
    "sass__inst_executed_shared_stores",
    "smsp__inst_executed_op_ldsm.sum",
    "smsp__inst_executed_op_shared_atom.sum",
    "smsp__inst_executed_op_global_red.sum")

for item in "${quest1[@]}";do
	PROFILE+="$item"
done

if [ $RUN_TEST = "print" ]
then
    echo "-----Printing parameters-----"
    echo "CUDA: ${CUDA}, Batchsize: ${BATCHSIZE}, Epochs: ${EPOCHS}, Partition: ${PARTITION}"
    echo "-----Profiling Parameters-----"
    echo "$PROFILE"


elif [ $RUN_TEST = "baseline" ]
then
    echo "-----Running Baseline-----"
    # python3 sage_dgl_partition.py --use-sample \
    #     --use-tt \
    #     --fan-out '3,5,15' \
    #     --epochs $EPOCHS \
    #     --device $CUDA \
    #     --partition $PARTITION \
    #     --tt-rank "16,16,16" \
    #     --p-shapes "125,140,140,140" \
    #     --q-shapes "5,5,2,2" \
    #     --emb-name "fbtt" \
    #     --num-layers 3 \
    #     --num-hidden 256 \
    #     --use-tt \
    #     --dataset "ogbn-products" \
    #     --batch $BATCHSIZE \
    #     --init "noinit" \
    #     --batch-count 1000 \
    #     $WORKSPACE


    python3 sage_dgl_partition.py --use-sample \
        --use-tt \
        --fan-out '3,5,15' \
        --epochs $EPOCHS \
        --device $CUDA \
        --partition $PARTITION \
        --tt-rank "16,16,16" \
        --p-shapes "50,60,60,60" \
        --q-shapes "2,4,4,4" \
        --batch $BATCHSIZE \
        --emb-name "fbtt" \
        --num-layers 3 \
        --num-hidden 256 \
        --dataset ogbn-arxiv \
        --init "noinit" \
        --logging \
        $WORKSPACE

    # python3 sage_dgl_partition.py --use-sample \
    #     --use-tt \
    #     --fan-out '3,5,15' \
    #     --epochs 5 \
    #     --device $CUDA \
    #     --partition $PARTITION \
    #     --tt-rank "16,16,16" \
    #     --p-shapes "125,140,140,140" \
    #     --q-shapes "5,5,2,2" \
    #     --batch $BATCHSIZE \
    #     --emb-name "fbtt" \
    #     --num-layers 3 \
    #     --num-hidden 256 \
    #     --dataset ogbn-products \
    #     --init "noinit" \
    #     --logging \
    #     $WORKSPACE
    
    # python3 sage_dgl_partition.py --use-sample \
    #     --use-tt \
    #     --fan-out '3,5,15' \
    #     --epochs $EPOCHS \
    #     --device $CUDA \
    #     --partition $PARTITION \
    #     --tt-rank "16,16" \
    #     --p-shapes "50,60,60" \
    #     --q-shapes "8,4,4" \
    #     --batch $BATCHSIZE \
    #     --emb-name "fbtt" \
    #     --num-layers 3 \
    #     --num-hidden 256 \
    #     --dataset ogbn-arxiv \
    #     --init "noinit" \
    #     --logging \
    #    $WORKSPACE

    # python3 sage_dgl_partition.py --use-sample \
    #     --use-tt \
    #     --fan-out '3,5,15' \
    #     --epochs 5 \
    #     --device "cuda:0" \
    #     --partition $PARTITION \
    #     --tt-rank "16,16" \
    #     --p-shapes "4000,5000,6000" \
    #     --q-shapes "4,4,8" \
    #     --batch 12 \
    #     --emb-name "fbtt" \
    #     --num-layers 3 \
    #     --num-hidden 256 \
    #     --dataset "ogbn-papers100M" \
    #     --init "noinit"

    python3 gcn_gat_partition.py \
        --use-tt \
        --epochs $EPOCHS \
        --device $CUDA \
        --partition $PARTITION \
        --tt-rank "16,16,16" \
        --p-shapes "50,60,60,60" \
        --q-shapes "2,4,4,4" \
        --emb-name "fbtt" \
        --dataset ogbn-arxiv \
        --use-labels \
        --use-linear \
        --model gcn \
        --init "noinit" \
        --logging \
        $WORKSPACE
    
    python3 gcn_gat_partition.py \
        --use-tt \
        --epochs $EPOCHS \
        --device $CUDA \
        --partition $PARTITION \
        --tt-rank "16,16,16" \
        --p-shapes "50,60,60,60" \
        --q-shapes "2,4,4,4" \
        --emb-name "fbtt" \
        --dataset ogbn-arxiv \
        --model gat \
        --init "noinit" \
        --logging \
        $WORKSPACE

elif [ $RUN_TEST = "p3" ]
then
    echo "-----Profiler only Script (GraphSAGE) p3-----"
    PROFILESIZE=1000
    CUDA_VISIBLE_DEVICE=1 python3 sage_profiler.py --use-sample \
            --fan-out '3,5,15' \
            --epochs 1 \
            --device "cuda:0" \
            --partition $PARTITION \
            --tt-rank "16,16" \
            --p-shapes "125,140,140" \
            --q-shapes "4,5,5" \
            --batch $PROFILESIZE \
            --emb-name "fbtt" \
            --num-layers 3 \
            --num-hidden 256 \
            --use-tt \
            --dataset "ogbn-products"

elif [ $RUN_TEST = "p4" ]
then
    echo "-----Profiler only Script (GraphSAGE) p4-----"
    PROFILESIZE=1000
    ncu --metrics dram__bytes_read,gpu__time_duration --clock-control none -o ncu-fwd-sage1-tt-bc1-4w-32g-${PROFILESIZE} -f --target-processes all \
        python3 sage_profiler.py --use-sample \
                --fan-out '3,5,15' \
                --epochs 1 \
                --device "cuda:0" \
                --partition $PARTITION \
                --tt-rank "16,16" \
                --p-shapes "440,450,425" \
                --q-shapes "4,5,5" \
                --batch $PROFILESIZE \
                --emb-name "fbtt" \
                --num-layers 3 \
                --num-hidden 256 \
                --use-tt \
                --dataset "ogbn-products"

elif [ $RUN_TEST = "cpu" ]
then
    echo "-----Running with CPU Embedding-----"
    ### products
    # python3 sage_dgl_partition.py --use-sample --epochs $EPOCHS --device "cpu" --partition $PARTITION --batch $BATCHSIZE --dataset "ogbn-products"

    ### arxiv
    # python3 sage_dgl_partition.py --epochs $EPOCHS --device "cpu" --partition $PARTITION --batch 1 --dataset "ogbn-arxiv" --access-count --plot

    ### papers
    python3 sage_dgl_partition.py --epochs $EPOCHS --device "cpu" --partition $PARTITION --batch 5 --dataset "ogbn-papers100M"
    
    ### proteins
    # python3 sage_dgl_partition.py --use-sample --epochs $EPOCHS --device "cpu" --partition $PARTITION --batch $BATCHSIZE --dataset "ogbn-proteins"

elif [ $RUN_TEST = "fbtt-products" ]
then
    echo "-----Running with FBTT (ogbn-products)-----"
    # compute-sanitizer --tool memcheck python3 sage_dgl_partition.py --use-sample --use-tt --epochs $EPOCHS --device $CUDA --partition $PARTITION --tt-rank "16,16" --p-shapes "50,80,100" --q-shapes "4,4,8" --batch $BATCHSIZE --emb-name "fbtt" --dataset "ogbn-arxiv"
    python sage_dgl_partition.py --use-sample \
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
        --init "noinit" \
        --batch-count 1000

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

elif [ $RUN_TEST = "final-p" ]
then
echo "-----Running with FBTT (ogbn-arxiv) Final-----"

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
        $WORKSPACE

elif [ $RUN_TEST = "final-p2" ]
then
echo "-----Running with FBTT (ogbn-products) Final-----"
    python3 sage_dgl_partition.py --use-sample \
        --use-tt \
        --fan-out '3,5,15' \
        --epochs $EPOCHS \
        --device $CUDA \
        --partition $PARTITION \
        --tt-rank "16,16,16" \
        --p-shapes "125,140,140,140" \
        --q-shapes "2,2,5,5" \
        --batch $BATCHSIZE \
        --emb-name "fbtt" \
        --num-layers 3 \
        --num-hidden 256 \
        --dataset ogbn-products \
        --use-cached \
        --sparse \
        --cache-size 10 \
        --init "noinit" \
        --batch-count 14000 \
        --logging \
        $WORKSPACE
    
    # python3 sage_dgl_partition.py --use-sample \
    #     --use-tt \
    #     --fan-out '3,5,15' \
    #     --epochs $EPOCHS \
    #     --device $CUDA \
    #     --partition $PARTITION \
    #     --tt-rank "16,16" \
    #     --p-shapes "50,60,60" \
    #     --q-shapes "4,4,8" \
    #     --batch $BATCHSIZE \
    #     --emb-name "fbtt" \
    #     --num-layers 3 \
    #     --num-hidden 256 \
    #     --dataset ogbn-arxiv \
    #     --sparse \
    #     --init "noinit" \
    #     --batch-count 11000 \
    #     --cache-size 10 \
    #     --use-cached \
    #     --logging \
    #     $WORKSPACE

    # python3 sage_dgl_partition.py --use-sample \
    #     --use-tt \
    #     --fan-out '3,5,15' \
    #     --epochs $EPOCHS \
    #     --device $CUDA \
    #     --partition $PARTITION\
    #     --tt-rank "16,16,16" \
    #     --p-shapes "125,140,100,100" \
    #     --q-shapes "2,2,5,5" \
    #     --batch $BATCHSIZE \
    #     --emb-name "fbtt" \
    #     --num-layers 3 \
    #     --num-hidden 256 \
    #     --dataset ogbn-products \
    #     --use-cached \
    #     --sparse \
    #     --cache-size 10 \
    #     --init "noinit" \
    #     --batch-count 14000 \
    #     $WORKSPACE

elif [ $RUN_TEST = "final-g" ]
then
echo "-----Running with GCN/GAT Final-----"
    python3 gcn_gat_partition.py \
        --use-tt \
        --epochs $EPOCHS \
        --device $CUDA \
        --partition $PARTITION \
        --tt-rank "16,16,16" \
        --p-shapes "50,60,60,60" \
        --q-shapes "4,2,4,4" \
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
        --logging \
        $WORKSPACE
    
    python3 gcn_gat_partition.py \
        --use-tt \
        --epochs $EPOCHS \
        --device $CUDA \
        --partition $PARTITION \
        --tt-rank "16,16,16" \
        --p-shapes "50,60,60,60" \
        --q-shapes "4,2,4,4" \
        --emb-name "fbtt" \
        --dataset ogbn-arxiv \
        --model gat \
        --sparse \
        --init "noinit" \
        --use-cache \
        --cache-size 10 \
        --batch-count 14000 \
        --logging \
        $WORKSPACE

elif [ $RUN_TEST = "gcn" ]
then
    echo "-----Running with GCN (${DATASET} in Full Batch) ----- "
    python3 gcn_gat_partition.py --use-tt --epochs $EPOCHS --device "cuda:0" --partition $PARTITION --tt-rank "16,16" --p-shapes "125,140,140" --q-shapes "4,4,8" --emb-name "fbtt" --dataset ogbn-arxiv --use-labels --use-linear --model gcn $WORKSPACE
    # python3 gcn_gat_partition.py --use-tt --epochs $EPOCHS --device "cuda:0" --partition $PARTITION --tt-rank "16,16" --p-shapes "125,140,140" --q-shapes "4,4,8" --emb-name "eff" --dataset ogbn-arxiv --use-labels --use-linear --model gcn $WORKSPACE

elif [ $RUN_TEST = "gat" ]
then
    echo "-----Running with GAT (${DATASET} in Full Batch) ----- "
    python3 gcn_gat_partition.py --use-tt --epochs $EPOCHS --device $CUDA --partition $PARTITION --tt-rank "16,16" --p-shapes "125,140,140" --q-shapes "4,4,8" --emb-name "fbtt" --dataset ogbn-arxiv --use-linear --num-heads 3 --model gat $WORKSPACE
    # python3 gcn_gat_partition.py --use-tt --epochs $EPOCHS --device $CUDA --partition $PARTITION --tt-rank "16,16" --p-shapes "125,140,140" --q-shapes "4,4,8" --emb-name "eff" --dataset ogbn-arxiv --use-linear --num-heads 3 --model gat $WORKSPACE

else
    echo "No test specified. Exiting..."
fi
