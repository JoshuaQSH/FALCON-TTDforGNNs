## Observation & Contribution

- Training larger graph at a time will give more a accurate result (FullNeighbor if possible)
- A larger embedding table before GNN -> More tt-cores
- A one-time preprocessing step is required. Key idea is to align graph topological information with the TT data structure (customized partitioning)
- Reorder the graph nodes based on the partition results (nodes in the same partition will have continuous indices)
- Efficient TT Table 

## How to run
```
# With the logs
python3 sage_dgl_partition.py --use-sample --use-tt --epochs 2 --device "cuda:0" --logging
```

## Dataset

| Dataset | #Node  |  #Edge | #Label  |  FeatLen |
|---|---|---|---|---|
| ogbn-arxiv | 169,343 | 1,166,243 | 40 | 128 |
| ogbn-products | 2,449,029 |  61,859,140	| 47 | 100 |
| ogbn-papers100M | 111,059,956 | 1,615,685,872 | 172 | 128 |

## GNNSAGE Settings

| Dataset | Notes | MemoryGPU (MB) | #Epochs | #BatchSize | TestAcc (%) | SamplingSize | Runtime (s) |
|---|---|---|---|---|---|---|
| ogbn-products | Baseline | 5923.5 | 2 | 1024 | 70.46% | [5, 10, 15] | 26.88 |
| ogbn-products | FullNeighbor | 16659.5 | / | 1024 | / | / | / |
| ogbn-products | NoTT-Sample-1 | 8555.3 | 2 | 1024 | 74.49% | [30, 50, 100] | 481.74 |
| ogbn-products | NoTT-Sample-2 | 5900.1 | 2 | 1024 | 29.52% | [1, 1, 1] | 20.21 |
| ogbn-products | NoTT-FullNeighbor-1 | 15152.3 | 2 | 128 | 72.09% | / | 13118.90 |
| ogbn-products | NoTT-Sample-3 | 5902.9 | 2 | 128 | 70.99% | [5, 10, 15] | 66.33 |
| ogbn-products | TTD-Embeddings-Full | 8506.3 | 2 | 128 | 58.92% | / | 13252.72 |
| ogbn-products | TTD-Embeddings-1 | 4691.4 | 2 | 1024 | 64.17% | [30, 50, 100] | 479.63 |
| ogbn-products | TTD-Embeddings-2 | 128.1 | 2 | 128 | 51.71% | [5, 10, 15] | 104.29 |
| ogbn-products | TTD-Embeddings-3 | 710.1 | 2 | 1024 | 49.31% | [5, 10, 15] | 53.51 |

### Hyperparameters (so far)
- tt_rank
- partition
- init
- fan_out