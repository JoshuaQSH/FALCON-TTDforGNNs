#!/bin/bash

BATCHSIZE=2048
PARTITION=0

EPOCHS=20
RUN_TEST=$@
CUDA="cuda:0"
PROFILE=""

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

for item in "${inst[@]}";do
	PROFILE+="$item"
done

if [ $RUN_TEST = "print" ]
then
    echo "-----Printing parameters-----"
    echo "CUDA: ${CUDA}, Batchsize: ${BATCHSIZE}, Epochs: ${EPOCHS}, Partition: ${PARTITION}"
    echo "-----Profiling Parameters-----"
    echo "$PROFILE"

elif [ $RUN_TEST = "rank" ]
then
    echo "-----Running with varying ranks-----"
    for Franks in {2..128..2}
    do
        for Sranks in {2..128..2}
        do
            echo "-----Running with [$Franks,$Sranks] ranks-----"
            python3 sage_dgl_partition.py --use-sample --use-tt --epochs $EPOCHS --device $CUDA --partition 125 --tt-rank "${Franks},${Sranks}" --p-shapes "125,140,140" --q-shapes "4,5,5" --batch $BATCHSIZE --logging
        done
    done

elif [ $RUN_TEST = "profile" ]
then
    echo "-----Running with profiling-----"
    ncu --set roofline -f -o sage_fbtt_roofline python3 sage_dgl_partition_.py --use-sample --use-tt --epochs 1 --device "cuda:1" --partition 0 --tt-rank "16,16" --p-shapes "125,140,140" --q-shapes "4,5,5" --batch 2048 --emb-name "fbtt"
    # ncu --metrics $PROFILE -f -o sage_fbtt \
    #     python3 sage_dgl_partition_.py \
    #     --use-sample \
    #     --use-tt \
    #     --epochs 1 \
    #     --device "cuda:1" \
    #     --partition 0 \
    #     --tt-rank "16,16" \
    #     --p-shapes "125,140,140" \
    #     --q-shapes "4,5,5" \
    #     --batch 2048 \
    #     --emb-name "fbtt"
    
elif [ $RUN_TEST = "partition" ]
then
    echo "-----Running with varying partitions-----"
    # for i in {0..20}
    # do
    #     echo "-----Running with partition $i-----"
    #     python3 sage_dgl_partition.py --use-sample --use-tt --epochs $EPOCHS --device $CUDA --partition $i --tt-rank "16,16" --p-shapes "125,140,140" --q-shapes "4,5,5" --batch $BATCHSIZE --logging
    # done
    # echo "-----Running with baseline -----"
    # python3 sage_dgl_partition.py --use-sample --use-tt --epochs $EPOCHS --device $CUDA --partition 0 --tt-rank "16,16" --p-shapes "125,140,140" --q-shapes "4,5,5" --emb-name "fbtt" --batch $BATCHSIZE --logging
    # echo "-----Running with rcmk -----"
    # python3 sage_dgl_partition.py --use-sample --use-tt --epochs $EPOCHS --device $CUDA --partition -2 --tt-rank "16,16" --p-shapes "125,140,140" --q-shapes "4,5,5" --emb-name "fbtt" --batch $BATCHSIZE --logging
    # echo "-----Running with METIS 3 -----"
    # python3 sage_dgl_partition.py --use-sample --use-tt --epochs $EPOCHS --device $CUDA --partition 3 --tt-rank "16,16" --p-shapes "125,140,140" --q-shapes "4,5,5" --emb-name "fbtt" --batch $BATCHSIZE --logging
    echo "-----Running with three level METIS -----"
    python3 sage_dgl_partition.py --use-sample --use-tt --epochs $EPOCHS --device $CUDA --partition -1 --tt-rank "16,16" --p-shapes "125,140,140" --q-shapes "4,5,5" --emb-name "fbtt" --batch $BATCHSIZE --logging

elif [ $RUN_TEST = "cpu" ]
then
    echo "-----Running with CPU Embedding-----"
    ### products
    # python3 sage_dgl_partition.py --use-sample --epochs $EPOCHS --device "cpu" --partition $PARTITION --batch $BATCHSIZE --dataset "ogbn-products"
    # python3 sage_dgl_partition.py --use-sample --epochs 5 --device "cpu" --partition $PARTITION --batch $BATCHSIZE --dataset "ogbn-products" --logging
    python3 sage_dgl_partition.py --use-sample --epochs 2 --device "cpu" --partition $PARTITION --batch 1 --dataset "ogbn-products"

    ### arxiv
    # python3 sage_dgl_partition.py --epochs $EPOCHS --device "cpu" --partition $PARTITION --batch 1 --dataset "ogbn-arxiv" --logging

    ### papers
    # python3 sage_dgl_partition.py --epochs $EPOCHS --device "cpu" --partition $PARTITION --batch 5 --dataset "ogbn-papers100M" --logging
    
    ### proteins
    # python3 sage_dgl_partition.py --use-sample --epochs $EPOCHS --device "cpu" --partition $PARTITION --batch $BATCHSIZE --dataset "ogbn-proteins"

elif [ $RUN_TEST = "fbtt" ]
then
    echo "-----Running with FBTT-----"
    # python3 sage_dgl_partition.py --use-sample --use-tt --epochs $EPOCHS --device $CUDA --partition $PARTITION --tt-rank "16,16" --p-shapes "125,140,140" --q-shapes "4,5,5" --batch $BATCHSIZE --emb-name "fbtt" --dataset "ogbn-products"
    # compute-sanitizer --tool memcheck python3 sage_dgl_partition.py --use-sample --use-tt --epochs $EPOCHS --device $CUDA --partition $PARTITION --tt-rank "16,16" --p-shapes "50,80,100" --q-shapes "4,4,8" --batch $BATCHSIZE --emb-name "fbtt" --dataset "ogbn-arxiv"
    # python3 sage_dgl_partition.py --use-sample --epochs $EPOCHS --device $CUDA --partition $PARTITION --tt-rank "16,16" --p-shapes "50,80,100" --q-shapes "4,4,8" --batch $BATCHSIZE --emb-name "fbtt" --dataset "ogbn-arxiv"

    python3 sage_dgl_partition.py --use-sample --fan-out '5,5,5' --epochs $EPOCHS --device $CUDA --partition $PARTITION --tt-rank "16,16" --p-shapes "200,220,300" --q-shapes "4,4,8" --batch $BATCHSIZE --emb-name "fbtt" --dataset "ogbn-papers100M"
    # python3 sage_dgl_partition.py --use-sample --use-tt --epochs $EPOCHS --device $CUDA --partition $PARTITION --tt-rank "16,16" --p-shapes "125,140,140" --q-shapes "4,5,5" --batch $BATCHSIZE --emb-name "fbtt" --dataset "ogbn-proteins"

elif [ $RUN_TEST = "fbtt4090" ]
then
    echo "-----Running with FBTT with 4090-----"
    # python3 sage_dgl_partition.py --use-sample --use-tt --epochs $EPOCHS --device $CUDA --partition $PARTITION --tt-rank "16,16" --p-shapes "125,140,140" --q-shapes "4,5,5" --batch 12 --emb-name "eff" --workspace home-3090
    # python3 sage_dgl_partition.py --use-sample --use-tt --epochs $EPOCHS --device $CUDA --partition $PARTITION --tt-rank "16,16" --p-shapes "125,140,140" --q-shapes "4,4,8" --batch $BATCHSIZE --emb-name "fbtt" --dataset "ogbn-arxiv" --workspace home-3090
    compute-sanitizer --tool memcheck --save cuda_mem.log python3 sage_dgl_partition.py --use-sample --use-tt --epochs $EPOCHS --device "cuda:1" --partition $PARTITION --tt-rank "16,16" --p-shapes "125,140,140" --q-shapes "4,5,5" --batch $BATCHSIZE --emb-name "eff" --dataset "ogbn-products" --workspace home-3090
    # python3 sage_dgl_partition.py --use-sample --epochs $EPOCHS --device $CUDA --partition $PARTITION --tt-rank "16,16" --p-shapes "125,140,140" --q-shapes "4,5,5" --batch $BATCHSIZE --emb-name "eff" --workspace home-3090

elif [ $RUN_TEST = "fbtt-full" ]
then
    echo "-----Running with FBTT full neighbors-----"
    python3 sage_dgl_partition.py --use-tt --epochs $EPOCHS --device $CUDA --partition $PARTITION --tt-rank "16,16" --p-shapes "125,140,140" --q-shapes "4,5,5" --batch $BATCHSIZE --emb-name "fbtt"

elif [ $RUN_TEST = "gcn" ]
then
    echo "-----Running with GCN -----"
    python3 gcn_gat_partition.py --use-tt --epochs $EPOCHS --device $CUDA --partition $PARTITION --tt-rank "16,16" --p-shapes "125,140,140" --q-shapes "4,4,8"  --batch $BATCHSIZE --emb-name "fbtt" --dataset "ogbn-arxiv" --use-labels --use-linear --model gcn

elif [ $RUN_TEST = "gat" ]
then
    echo "-----Running with GAT -----"
    python3 gcn_gat_partition.py --use-tt --epochs $EPOCHS --device $CUDA --partition $PARTITION --tt-rank "16,16" --p-shapes "125,140,140" --q-shapes "4,4,8"  --batch $BATCHSIZE --emb-name "fbtt" --dataset "ogbn-arxiv" --use-linear --num-heads 3 --model gat

elif [ $RUN_TEST = "eff" ]
then
    echo "-----Running with Efficient TT-----"
    compute-sanitizer --tool memcheck --save supp.xml --xml yes python3 sage_dgl_partition.py --use-sample --use-tt --epochs $EPOCHS --device $CUDA --partition $PARTITION --tt-rank "16,16" --p-shapes "125,140,140" --q-shapes "4,5,5" --batch $BATCHSIZE --emb-name "eff"

else
    echo "No test specified. Exiting..."
fi