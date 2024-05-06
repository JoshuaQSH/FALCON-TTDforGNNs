## Observation & Contribution

- Training larger graph at a time will give more a accurate result (FullNeighbor if possible)
  - But a tradeoff appeared, should consider the gradient loss as well
- A larger embedding table before GNN -> More tt-cores
- A one-time preprocessing step is required. Key idea is to align graph topological information with the TT data structure (customized partitioning)
- Reorder the graph nodes based on the partition results (nodes in the same partition will have continuous indices)
- Efficient TT Table for index computation reused

```python
# Step - 1: embedding layer init
self.embed_layer = TTEmbeddingBag(num_embeddings=num_nodes, embedding_dim=in_feats)
# init tt_cores
tt_cores = get_ortho()
self.embed_layer.tt_cores[i].data = th.tensor(tt_cores[i]).to(device)
# Step - 2: forward 
h = self.embed_layer(input_nodes, offsets)
for l, (layer, block) in enumerate(zip(self.layers, blocks)):
    # copy the representation of nodes on the RHS from the appropriate nodes on the LHS.
    h_dst = h[:block.num_dst_nodes()]
    h = layer(block, (h, h_dst)) # h = layer(block, h_dst)
```

## How to run

We recommand run the model with the `run_script.sh`, but to test each module, try run it seperately.

- With the scirpt:
```shell
# Run with GraphSAGE, dataset: ogbn-products, FBTT
./run_script fbtt

# Run with GCN, dataset: ogbn-arxiv, FBTT
./run_script gcn

# Run with GAT, dataset: ogbn-arxiv, FBTT
./run_script gat

# To profile the model
./run_script profile

# Run with three level partitioning
./run_script partition

# Run with eff
./run_script eff
```

- Without the script:
```
# With the logs
$ python3 sage_dgl_partition.py --use-sample --use-tt --epochs 2 --device "cuda:0" --logging

# Tune the parameters
$ python3 sage_dgl_partition.py --use-sample --use-tt --epochs 2 --device "cuda:0" --partition 20 --tt-rank "16,16" --p-shapes "125,140,140" --q-shapes "4,5,5" --batch 2048

# With eff TT opt
$ python3 sage_dgl_partition.py --use-sample --use-tt --epochs 2 --device "cuda:0" --partition 125 --tt-rank "16,16" --p-shapes "125,140,140" --q-shapes "4,5,5" --batch 2 --emb-name "eff"

# Train Full Graph Cora
$ python3 train_full_small_graph.py --dataset cora --device "cuda:0" 
```

## Profiling

We use nsight-compute for the profiling:

spline_and_spread

```shell
$ ncu --set roofline -f -o sage_fbtt python3 sage_dgl_partition_.py --use-sample --use-tt --epochs 1 --device "cuda:1" --partition 125 --tt-rank "16,16" --p-shapes "125,140,140" --q-shapes "4,5,5" --batch 2048 --emb-name "fbtt"

$ ncu --metrics sass__inst_executed_shared_loads,sass__inst_executed_global_loads -f -o sage_fbtt python3 sage_dgl_partition.py --use-sample --use-tt --epochs 1 --device "cuda:1" --partition -1 --tt-rank "16,16" --p-shapes "125,140,140" --q-shapes "4,5,5" --batch 2048 --emb-name "fbtt"

```

```text
##  Global and shared memory read/write
dram__bytes.sum.peak_sustained,dram__bytes.sum.per_second,dram__bytes_read.sum,dram__bytes_read.sum,dram__sectors_read.sum,dram__sectors_write.sum,sass__inst_executed_shared_loads,sass__inst_executed_shared_stores,smsp__inst_executed_op_ldsm.sum,smsp__inst_executed_op_shared_atom.sum,smsp__inst_executed_op_global_red.sum

##

```

## Dataset

| Dataset | #Node  |  #Edge | #Label  |  FeatLen |
|---|---|---|---|---|
| ogbn-arxiv | 169,343 | 1,166,243 | 40 | 128 |
| ogbn-products | 2,449,029 |  61,859,140	| 47 | 100 |
| ogbn-papers100M | 111,059,956 | 1,615,685,872 | 172 | 128 |

## GNNSAGE Settings

| Dataset | Notes | MemoryGPU (MB) | #Epochs | #BatchSize | TestAcc (%) | SamplingSize | Runtime (s) |
|---|---|---|---|---|---|---|---|
| ogbn-products | Baseline | 5923.5 | 2 | 1024 | 70.46% | [5, 10, 15] | 26.88 |
| ogbn-products | FullNeighbor | 16659.5 | / | 1024 | / | / | / |
| ogbn-products | NoTT-Sample-1 | 8555.3 | 2 | 1024 | 74.49% | [30, 50, 100] | 481.74 |
| ogbn-products | NoTT-Sample-2 | 5900.1 | 2 | 1024 | 29.52% | [1, 1, 1] | 20.21 |
| ogbn-products | NoTT-FullNeighbor-1 | 15152.3 | 2 | 128 | 72.09% | / | 13118.90 |
| ogbn-products | NoTT-Sample-3 | 5902.9 | 2 | 128 | 70.99% | [5, 10, 15] | 66.33 |
| ogbn-products | NoTT-Sample-3(metis-128) | 5925.1 | 2 | 1024 | 72.11% | [5, 10, 15] | 33.10 |
| ogbn-products | NoTT-Sample-4 | 5914.6 | 2 | 128 | 68.7% | [5, 5, 10] | 26.39 |
| ogbn-products | TTD-Embeddings-Full | 8506.3 | 2 | 128 | 58.92% | / | 13252.72 |
| ogbn-products | TTD-Embeddings-1 | 4691.4 | 2 | 1024 | 64.17% | [30, 50, 100] | 479.63 |
| ogbn-products | TTD-Embeddings-2 | 128.1 | 2 | 128 | 51.71% | [5, 10, 15] | 104.29 |
| ogbn-products | TTD-Embeddings-3 | 129.0 | 2 | 128 | 60.66% | [5, 10, 15] | 127.73 |
| ogbn-products | TTD-Embeddings-4 | 710.1 | 2 | 1024 | 49.31% | [5, 10, 15] | 53.51 |
| ogbn-products | TTD-Embeddings-5(metis-random) | 709.9 | 2 | 1024 | 57.94% | [5, 10, 15] | 53.29 |
| ogbn-products | TTD-Embeddings-5(metis-100) | 710.6 | 2 | 1024 | 56.75% | [5, 10, 15] | 58.50 |
| ogbn-products | TTD-Embeddings-5(metis-128) | 711.0 | 2 | 1024 | 69.34% | [5, 10, 15] | 56.89 | **
| ogbn-products | TTD-Embeddings-5(rcmk) | 710.1 | 2 | 1024 | 71.47% | [5, 10, 15] | 58.14 | **
| ogbn-products | TTD-Embeddings-4(R24) | 714.8 | 2 | 1024 | 67.62% | [5, 10, 15] | 88.25 |
| ogbn-products | TTD-Embeddings-4(R16) | 710.7 | 2 | 1024 | 66.62% | [5, 10, 15] | 58.15 |
| ogbn-products | TTD-Embeddings-4(R16)(metis-128) | 1208.1 | 2 | 2048 | 67.16% | [5, 10, 15] | 42.48 |
| ogbn-products | TTD-Embeddings-4-Cached(R16) | 860 | 2 | 1024 | 64.43% | [5, 10, 15] | 83.08 |
| ogbn-products | TTD-Embeddings-4(R4) | 709.3 | 2 | 1024 | 45.73% | [5, 10, 15] | 52.35 |
| ogbn-products | TTD-Embeddings-4(R2) | 708.7 | 2 | 1024 | 36.34% | [5, 10, 15] | 52.05 |

- NoTT-Sample-4 and TTD-Embeddings-4(R16) have similar TestAcc with the same batchsize and epoch settings. TTD saved 8x memory space but 50% Runtime drop.
- NoTT-Sample-3 gives even higher test acc but with more sampling neighbors in the second layer, slower runtime. TTD will save 8x memory and also gain 1.23x runtime speedup
- TTD-EMbeddings-5 partition with METIS-128 and TTD-Embedding-5 with rcmk offer quite well test acc (compared with only sampling), but with 2.11x speeddown (53% runtime drop)
- A demo test result: 0.3430ms (Embeddding), 12.5878ms (FBTT), 3.8593ms(Effi) 


### Hyperparameters (so far)

- tt_rank
- partition
- init
- fan_out

For the `TTEmbeddingBag()` setting, we follow:
```python
## ...
assert len(self.tt_p_shapes) >= 2
assert len(self.tt_p_shapes) <= 4
assert len(tt_ranks) + 1 == len(self.tt_p_shapes)
assert len(self.tt_p_shapes) == len(self.tt_q_shapes)
```

### Memcheck for debugging

```
compute-sanitizer --tool memcheck ... --save cuda_mem_log
```