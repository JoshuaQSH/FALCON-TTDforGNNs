import numpy as np
import torch as th
import time
import scipy

import sys
sys.path.insert(0, '/home/shenghao/FBTT-Embedding')
sys.path.insert(0, '/home/shenghao/home-3090/FBTT-Embedding')
from tt_embeddings_ops import TTEmbeddingBag
from tt_embeddings_ops import tt_matrix_to_full
import torch.nn.init as init

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch (and torchTT) GNN Training')
    # General
    parser.add_argument('--device', type=str, default="cpu", help='CUDA device (default: "cpu")')
    parser.add_argument('--model', type=str, default="sage", help='Model (default: "sage")')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate (default: 0.01)')
    parser.add_argument('--test-sparse', action="store_true", default=False, help='A unit test for the sparse format (default: False)')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight for L2 loss")
    parser.add_argument("--aggregator-type", type=str, default="gcn", help="Aggregator type: mean/gcn/pool/lstm")

    # Data loader
    parser.add_argument('--sample', type=int, default=30, help='Number of samples (default: 30)')
    parser.add_argument('--batch', type=int, default=1024, help='Batch size (default: 1024)')
    parser.add_argument('--neighbors', type=int, default=2, help='Number of neighbors for each sampling (default: 2)')
    # ["ogbn-arxiv", "ogbn-products", "ogbn-proteins", "ogbn-papers100M"]
    parser.add_argument('--dataset', type=str, default="ogbn-products", help='Dataset (default: "ogbn-products")')
    parser.add_argument('--use-sample', action="store_true", default=False, help='Whether to sample the dataset (default: False)')
    parser.add_argument('--num-workers', type=int, default=4, help="Number of sampling processes. Use 0 for no extra process.")

    # GNN layers related
    parser.add_argument('--num-hidden', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--num-heads', type=int, default=3, help="Number of attention heads (for GAT)")
    parser.add_argument("--use-linear", action="store_true", help="Use linear layer (for GCN).")
    parser.add_argument("--use-labels", action="store_true", help="Use labels in the training set as input features (also for GCN).")

    # TT
    parser.add_argument('--fan-out', type=str, default='5,10,15')
    parser.add_argument("--use-tt", action="store_true", default=False, help="Use tt-emb layer. Whether to use TT format (default: False). ")
    # parser.add_argument("--tt-rank", type=int, default=8)
    parser.add_argument("--partition", type=int, default=0, help="-1 for customized permute, >0 for METIS, ==0 for -2 for rcmk")
    parser.add_argument('--init', type=str, default="ortho")
    parser.add_argument('--emb-name', type=str, default="fbtt")
    parser.add_argument('--dim', type=int, default=100, help='Embedding dimension.')
    parser.add_argument('--tt-rank', type=str, default="16,16", help='The ranks of TT cores')
    parser.add_argument('--p-shapes', type=str, default="125,140,140", help='The product of all elements is not smaller than num_embeddings.')
    parser.add_argument('--q-shapes', type=str, default="5,5,4", help='The product of all elements is equal to embedding_dim.')
    
    # Extra parts
    parser.add_argument('--val-batch-size', type=int, default=10000)
    parser.add_argument('--log-every', type=int, default=20)
    parser.add_argument('--eval-every', type=int, default=1)
    parser.add_argument('--save-pred', type=str, default='')
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--workspace', type=str, default='')
    parser.add_argument('--access-counts', action="store_true", help="Whether to count the access times of embeddings.")
    
    parser.add_argument("--n-runs", type=int, default=1)
    parser.add_argument("--store-emb", action="store_true", help="Store training embedding")
    parser.add_argument('--save-model', action="store_true")
    parser.add_argument('--plot', action="store_true", help="Plot the graph degree distributitions")
    parser.add_argument('--logging', action="store_true", help="Whether to log the training process")

    global args
    args = parser.parse_args()
    print(args)

    return args

def compression_rate(N, R, cores):
    
    output = 'TT'
    output += ' with sizes and ranks:\n'
    output += 'N = ' + str(N) + '\n'
    output += 'R = ' + str(R) + '\n\n'
    output += 'Device: ' + \
    str(cores[0].device)+', dtype: ' + \
    str(cores[0].dtype)+'\n'
    entries = sum([th.numel(c) for c in cores])
    output += '#entries ' + str(entries) + ' compression ' + str(
    entries/np.prod(np.array(N, dtype=np.float64))) + '\n'
    
    return output

def get_eigen(g, k, name, mode='adj'):
    file_name = name + '_' + mode + '_' + str(k)
    if mode == 'adj':
        adj = g.adj(scipy_fmt='csr')
    elif mode == 'laplacian':
        adj = g.adj(scipy_fmt='csr') * -1
        diag = g.in_degrees().detach().numpy()
        diag = scipy.sparse.diags(diag)
        adj = diag - adj
    start = time.time()
    eigen_vals, eigen_vecs = scipy.sparse.linalg.eigs(adj.astype(np.float32), k=k, tol=1e-5, ncv=k*3)
    print('Compute eigen: {:.3f} seconds'.format(time.time() - start))

    return eigen_vals, eigen_vecs

def get_ortho(tt_ranks, tt_p_shapes, tt_q_shapes):
    tt_cores = []
    v1 = np.zeros(shape=(tt_ranks[0], tt_p_shapes[0], tt_q_shapes[0], tt_ranks[1]), dtype=np.float32)
    v2 = np.zeros(shape=(tt_ranks[1], tt_p_shapes[1], tt_q_shapes[1], tt_ranks[2]), dtype=np.float32)
    v3 = np.zeros(shape=(tt_ranks[2], tt_p_shapes[2], tt_q_shapes[2], tt_ranks[3]), dtype=np.float32)
    tt_rank = tt_ranks[1]

    #generate for G_0
    m = np.random.normal(loc=0, scale=1, size=(v1.shape[1]*tt_rank, v1.shape[1]*tt_rank)).astype(np.float32)
    q, r = np.linalg.qr(m)
    k = 0
    for i in np.arange(v1.shape[2]):
        v1[0, :, i, :] = np.reshape(q[k, :]/np.linalg.norm(q[k,:]), (v1.shape[1], v1.shape[3]))
        k = k + 1
    v1 = np.transpose(v1, (1, 0, 2, 3)).reshape((1, v1.shape[1], -1))
    tt_cores += [v1]

    #generate for G_1
    m = np.random.normal(loc=0, scale=1, size=(v2.shape[1]*tt_rank, v2.shape[1]*tt_rank)).astype(np.float32)
    q, r = np.linalg.qr(m)
    k = 0
    for i in np.arange(v2.shape[0]):
        for j in np.arange(v2.shape[2]):
            v2[i, :, j, :] = np.reshape(q[k, :]/np.linalg.norm(q[k,:]), (v2.shape[1], v2.shape[3]))
            k = k + 1
    v2 = np.transpose(v2, (1, 0, 2, 3)).reshape((1, v2.shape[1], -1))
    tt_cores += [v2]

    #generate for G_2
    m = np.random.normal(loc=0, scale=1, size=(v3.shape[1], v3.shape[1])).astype(np.float32)
    q, r = np.linalg.qr(m)
    k = 0
    for i in np.arange(v3.shape[0]):
        for j in np.arange(v3.shape[2]):
            v3[i, :, j, 0] = q[k,:]/np.linalg.norm(q[k,:])
            k = k+1
    v3 = np.transpose(v3, (1, 0, 2, 3)).reshape((1, v3.shape[1], -1))
    tt_cores += [v3]
    return tt_cores

def tt_matrix_decomp(matrix, tt_ranks, tt_p_shapes, tt_q_shapes):
    # breakpoint()
    dims = [tt_p_shapes[i] * tt_q_shapes[i] for i in range(3)]
    tensor = np.reshape(matrix, tt_p_shapes + tt_q_shapes)
    tensor = np.transpose(tensor, (0, 3, 1, 4, 2, 5))
    tensor = np.reshape(tensor, dims)

    norm = np.linalg.norm(tensor)
    temp = tensor
    #d = len(np.shape(tensor))
    d = 3
    cores = []
    dims = list(tensor.shape)

    ranks = [1] * (d+1)
    for i in range(d-1):
        curr_mode = [tt_p_shapes[i], tt_q_shapes[i]]
        rows = ranks[i] * dims[i]
        temp = np.reshape(temp, (rows, -1))  
        cols = temp.shape[-1]
        
        if tt_ranks[i+1] == 1:
            ranks[i+1] = 1
        else:
            ranks[i+1] = min(tt_ranks[i+1], cols, rows)
            
        [u, s, vh] = np.linalg.svd(temp, full_matrices=False)     
        u = u[:, :ranks[i+1]]
        s = s[:ranks[i+1]]
        vh = vh[:ranks[i+1], :]
                 
        new_core = np.reshape(u, [ranks[i]] +  curr_mode + [ranks[i+1]])
        new_core = np.transpose(new_core, (1, 0, 2, 3))
        new_core = np.reshape(new_core, [1, tt_p_shapes[i], -1])
        cores.append(th.tensor(new_core, dtype=th.float))
        #print('core', i, new_core.shape)

        temp = np.diag(s).dot(vh)
                
    last_mode = [tt_p_shapes[-1], tt_q_shapes[-1]]
    new_core = np.reshape(temp, [ranks[d-1]] +  last_mode + [1])
    new_core = np.transpose(new_core, (1, 0, 2, 3))
    new_core = np.reshape(new_core, [1, tt_p_shapes[-1], -1])
    cores.append(th.tensor(new_core, dtype=th.float))

    return cores, ranks
