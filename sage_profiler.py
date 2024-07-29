import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl.nn.pytorch as dglnn
import time
import argparse
import tqdm
from ogb.nodeproppred import DglNodePropPredDataset

import unittest
from typing import List, Tuple

import hypothesis.strategies as st

import os
from tt_utils import *
from utils import Logger, gpu_timing, memory_usage, calculate_access_percentages, plot_access_percentages

# from gnn_model import SAGE
from graphloader import dgl_graph_loader

from FBTT.tt_embeddings_ops import (
    OptimType,
    TableBatchedTTEmbeddingBag,
    tt_matrix_to_full,
    TTEmbeddingBag,
)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'

def generate_random_edges(num_src_nodes, num_dst_nodes, num_edges):
    # Ensure that the number of edges does not exceed the maximum possible number (full connectivity)
    max_edges = num_src_nodes * num_dst_nodes
    if num_edges > max_edges:
        raise ValueError("Number of edges exceeds the maximum possible for complete bipartite graph connectivity.")

    # Randomly generate unique edge pairs
    src_ids = np.random.randint(0, num_src_nodes, size=num_edges)
    dst_ids = np.random.randint(0, num_dst_nodes, size=num_edges)
    return torch.tensor(src_ids), torch.tensor(dst_ids)

def create_block(num_src_nodes, num_dst_nodes, num_edges):
    # Generate edges
    edges = generate_random_edges(num_src_nodes, num_dst_nodes, num_edges)

    # Create a graph with specified edges
    g = dgl.heterograph({
        ('_N', '_E', '_N'): edges
    }, num_nodes_dict={'_N': num_src_nodes + num_dst_nodes})

    # Split nodes into src and dst
    src_nodes = torch.arange(0, num_src_nodes)
    dst_nodes = torch.arange(num_src_nodes, num_src_nodes + num_dst_nodes)

    # Convert to block
    block = dgl.to_block(g, {'_N': src_nodes, '_E': edges[0], '_N': dst_nodes})

    return block

# Function to create a DGL graph
def create_graph(num_src_nodes, num_dst_nodes, num_edges):
    src_nodes = torch.randint(0, num_src_nodes, (num_edges,))
    dst_nodes = torch.randint(0, num_dst_nodes, (num_edges,))
    graph = dgl.graph((src_nodes, dst_nodes))
    
    return graph

def generate_sparse_feature(
    batch_size,
    num_embeddings: int,
    pooling_factor: float,
    pooling_factor_std: float,
    generate_scores: bool = False,
    unary: bool = False,
    unique: bool = False,
) -> Tuple[List, List, List, List]:
    if not unary:
        lengths = np.round(
            np.random.normal(pooling_factor, pooling_factor_std, batch_size)
        ).astype(np.int64)
        lengths = list(np.where(lengths < 0, 0, lengths))
        total_length = np.sum(lengths)
    else:
        lengths = list(np.ones(batch_size).astype(np.int64))
        total_length = batch_size
    indices = list(
        np.random.choice(
            range(num_embeddings), size=total_length, replace=not unique
        ).astype(np.int64)
    )
    print("indices Size: ", len(indices))
    if generate_scores:
        scores = list(np.round(np.random.random(total_length) * 20))
    else:
        scores = []
    offsets = [0] + list(np.cumsum(lengths))
    return (lengths, indices, offsets, scores)

# Hook to track input indices of the embedding layer
embedding_indices = []

# Hook to track indices
def hook(module, input, output):
    global embedding_indices
    # print("Layer: ", module)
    print("Input nodes: ", input[0].cpu().numpy() if isinstance(input[0], torch.Tensor) else [i.cpu().numpy() for i in input])
    print("Output size: ", output.size())


class SAGE_ONLY(nn.Module):
    def __init__(self,
                 num_nodes,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 graph=None,
                 device='cpu'):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()

        if n_layers == 1:
            self.layers.append(dglnn.SAGEConv(in_feats, n_classes, 'mean'))

        else:
            self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
            for i in range(1, n_layers - 1):
                self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
            self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))

        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.num_nodes= num_nodes
        self.device = device

    def forward(self, blocks, h):
        if self.n_layers == 1:
            h = self.layers[0](blocks[0], h)
            # h_dst = h[:blocks[0].num_dst_nodes()]
            # h = self.layers[0](blocks[0], (h, h_dst))
            h = self.activation(h)
            h = self.dropout(h)
        
        else:
            for l, (layer, block) in enumerate(zip(self.layers, blocks)):
                h_dst = h[:block.num_dst_nodes()]
                h = layer(block, (h, h_dst))
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

        return h


class SAGE(nn.Module):
    def __init__(self,
                 num_nodes,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 use_tt=False,
                 tt_rank=[16,16],
                 p_shapes=None,
                 q_shapes=None,
                 dist=None,
                 graph=None,
                 device='cpu',
                 embed_name ='fbtt',
                 access_counts=False,
                 use_cached=False,
                 cache_size=0,
                 batch_count=1000):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        
        if n_layers > 1:
            self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
            for i in range(1, n_layers - 1):
                self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
            self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        elif n_layers == 1:
            self.layers.append(dglnn.SAGEConv(in_feats, n_classes, 'mean'))

        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.num_nodes= num_nodes
        self.use_tt = use_tt
        self.device = device
        self.use_cached = use_cached

        # default: 10% of the embeddings (num_nodes, also known as num_embeddings)
        self.cache_size = int(0.1 * cache_size * num_nodes)
        # default: num_embeddings
        self.hashtbl_size = num_nodes
        if use_tt:
            self.embed_layer = TTEmbeddingBag(
                    num_embeddings=num_nodes,
                    embedding_dim=in_feats,
                    tt_ranks=tt_rank,
                    tt_p_shapes=p_shapes, # The factorization of num_embeddings
                    tt_q_shapes=q_shapes, # Same as the in_feats
                    sparse=False,
                    use_cache=False,
                    cache_size=self.cache_size,
                    hashtbl_size=self.hashtbl_size,
                    weight_dist="normal",
                    batch_count=batch_count,)
        else:
            self.embed_layer = torch.nn.Embedding(num_nodes, in_feats)                      

    def forward(self, blocks, input_nodes):
        if self.use_tt:
            offsets = th.arange(input_nodes.shape[0] + 1).to(self.device)
            input_nodes = input_nodes.to(self.device)   
            h = self.embed_layer(input_nodes, offsets)

        else:
            h = self.embed_layer(input_nodes.to(self.device))
        if self.n_layers == 1:
            h = self.layers[0](blocks, h)
            h = self.activation(h)
            h = self.dropout(h)
        else:
            for l, (layer, block) in enumerate(zip(self.layers, blocks)):
                h_dst = h[:block.num_dst_nodes()]
                h = layer(block, (h, h_dst))
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)
        return h

class TestTTEmbeddingBag(unittest.TestCase):
    def __init__(self, batch_size, tt_ndims, tt_ranks, tt_p_shapes, tt_q_shapes, device, use_tt):
        super(TestTTEmbeddingBag, self).__init__()
        # tt_p_shapes=[120, 90, 110]
        # tt_q_shapes=[4, 4, 4]
        # tt_ranks=[12, 14]
        self.tt_ndims = tt_ndims
        self.tt_ranks = tt_ranks
        self.tt_p_shapes = tt_p_shapes
        self.tt_q_shapes = tt_q_shapes
        self.device = device
        self.use_tt = use_tt
        self.batch_size = batch_size
        torch.cuda.set_device(device)

    
    def test_forward(self, indices, offsets):
        tt_p_shapes = self.tt_p_shapes[:self.tt_ndims]
        tt_q_shapes = self.tt_q_shapes[:self.tt_ndims]
        tt_ranks = self.tt_ranks[: (self.tt_ndims - 1)]
        num_embeddings = np.prod(np.array(tt_p_shapes))
        embedding_dim = np.prod(np.array(tt_q_shapes))
        
        # create TT-Embedding op
        offsets = torch.tensor(offsets, dtype=torch.int64, device=self.device)
        indices = torch.tensor(indices, dtype=torch.int64, device=self.device)
        batch_count = 1000
        
        if self.use_tt:
            tt_emb = TTEmbeddingBag(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                tt_p_shapes=tt_p_shapes,
                tt_q_shapes=tt_q_shapes,
                tt_ranks=tt_ranks,
                sparse=False,
                use_cache=False,
                weight_dist="uniform",
                batch_count=batch_count,
            )
            tt_emb.to(self.device)
            output = tt_emb(indices, offsets)

        else:
            fake_weight = torch.rand((num_embeddings, embedding_dim), device=device, requires_grad=True)
            emb = torch.nn.EmbeddingBag(
                num_embeddings,
                embedding_dim,
                sparse=True,
                mode="sum",
                # _weight=tt_emb.full_weight(),
                _weight=fake_weight,
                include_last_offset=True,
            )
            emb.to(self.device)
            output_ref = emb(indices.long(), offsets.long())

        # output = tt_emb(indices, offsets)
        # output_ref = emb(indices.long(), offsets.long())
        # torch.testing.assert_allclose(output, output_ref)

    def test_backward_dense(self, indices, offsets):
        # tt_p_shapes = [7, 9, 11, 5]
        # tt_q_shapes = [3, 4, 5, 7]
        # tt_ranks = [13, 12, 7]
        tt_p_shapes = self.tt_p_shapes[:self.tt_ndims]
        tt_q_shapes = self.tt_q_shapes[:self.tt_ndims]
        tt_ranks = self.tt_ranks[: (self.tt_ndims - 1)]

        # batch count
        batch_count = 5000

        # create TT-Embedding op
        offsets = offsets.to(self.device)
        indices = torch.tensor(indices, dtype=torch.int64, device=self.device)
        

        num_embeddings = np.prod(np.array(tt_p_shapes))
        embedding_dim = np.prod(np.array(tt_q_shapes))

        if self.use_tt:
            tt_emb = TTEmbeddingBag(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                tt_p_shapes=tt_p_shapes,
                tt_q_shapes=tt_q_shapes,
                tt_ranks=tt_ranks,
                sparse=False,
                use_cache=False,
                weight_dist="uniform",
                batch_count=batch_count
            )
            tt_emb.to(self.device)
            # d_output = torch.rand(self.batch_size, embedding_dim, device=self.device) * 0.1
            tt_cores = [tt.clone().detach().requires_grad_(True) for tt in tt_emb.tt_cores]
            full_weight = tt_matrix_to_full(tt_p_shapes, tt_q_shapes, tt_ranks, tt_cores, [1, 0, 2, 3])
            output = tt_emb(indices, offsets)
            d_output = torch.rand(output.shape[0], embedding_dim, device=self.device) * 0.1
            output.backward(d_output)
            
        else:
            fake_weight = torch.rand((num_embeddings, embedding_dim), device=device, requires_grad=True)
            emb = torch.nn.EmbeddingBag(
                num_embeddings,
                embedding_dim,
                sparse=True,
                mode="sum",
                # _weight=tt_emb.full_weight(),
                _weight=fake_weight,
                include_last_offset=True,
            )
            emb.to(self.device)
            d_output = torch.rand(self.batch_size, embedding_dim, device=self.device) * 0.1
            output_ref = emb(indices.long(), offsets.long())
            output_ref.backward(d_output)
            
            # d_weight_ref = emb.weight.grad.to_dense()
            # full_weight.backward(d_weight_ref)
        
        # asserting for difference
        # for i in range(tt_ndims):
        #     torch.testing.assert_allclose(tt_emb.tt_cores[i].grad, tt_cores[i].grad)


    def test_backward_sgd(self, indices, offsets):
        tt_p_shapes = self.tt_p_shapes[:self.tt_ndims]
        tt_q_shapes = self.tt_q_shapes[:self.tt_ndims]
        tt_ranks = self.tt_ranks[: (self.tt_ndims - 1)]

        # create TT-Embedding op
        offsets = torch.tensor(offsets, dtype=torch.int64, device=self.device)
        indices = torch.tensor(indices, dtype=torch.int64, device=self.device)
        num_embeddings = np.prod(np.array(tt_p_shapes))
        embedding_dim = np.prod(np.array(tt_q_shapes))
        learning_rate = 0.1

        if self.use_tt:
            tt_emb = TTEmbeddingBag(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                tt_p_shapes=tt_p_shapes,
                tt_q_shapes=tt_q_shapes,
                tt_ranks=tt_ranks,
                sparse=False,
                use_cache=False,
                optimizer=OptimType.SGD,
                learning_rate=learning_rate,
                weight_dist="uniform",
            )
            tt_emb.to(device)
            d_output = torch.rand(self.batch_size, embedding_dim, device=device) * 0.1
            tt_cores = [tt.clone().detach().requires_grad_(True) for tt in tt_emb.tt_cores]
            full_weight = tt_matrix_to_full(
                tt_p_shapes, tt_q_shapes, tt_ranks, tt_cores, [1, 0, 2, 3]
            )
            
            # tt_emb
            output = tt_emb(indices, offsets)
            output.backward(d_output)
            # new_tt_cores = []
            # new_tt_cores = [(t - t.grad * learning_rate) for t in tt_cores]

        else:
            emb = torch.nn.EmbeddingBag(
                num_embeddings,
                embedding_dim,
                sparse=True,
                mode="sum",
                _weight=tt_emb.full_weight(),
                include_last_offset=True,
            )
            emb.to(device)
            # reference
            output_ref = emb(indices.long(), offsets.long())
            output_ref.backward(d_output)
            # d_weight_ref = emb.weight.grad.to_dense()
            # full_weight.backward(d_weight_ref)

        
        # for i in range(tt_ndims):
        #     torch.testing.assert_allclose(tt_emb.tt_cores[i], new_tt_cores[i])

    def test_backward_adagrad(self, indices, offsets):
        tt_p_shapes = self.tt_p_shapes[:self.tt_ndims]
        tt_q_shapes = self.tt_q_shapes[:self.tt_ndims]
        tt_ranks = self.tt_ranks[: (self.tt_ndims - 1)]


        num_embeddings = np.prod(np.array(tt_p_shapes))
        embedding_dim = np.prod(np.array(tt_q_shapes))
        learning_rate = 0.1
        eps = 0.0001

        # create TT-Embedding op
        offsets = torch.tensor(offsets, dtype=torch.int64, device=device)
        indices = torch.tensor(indices, dtype=torch.int64, device=device)

        if self.use_tt:
            tt_emb = TTEmbeddingBag(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                tt_p_shapes=tt_p_shapes,
                tt_q_shapes=tt_q_shapes,
                tt_ranks=tt_ranks,
                sparse=False,
                use_cache=False,
                optimizer=OptimType.EXACT_ADAGRAD,
                learning_rate=learning_rate,
                eps=eps,
                weight_dist="uniform",
            )
            tt_emb.to(device)
            d_output = torch.rand(self.batch_size, embedding_dim, device=device) * 0.1
            tt_cores = [tt.clone().detach().requires_grad_(True) for tt in tt_emb.tt_cores]
            full_weight = tt_matrix_to_full(
                tt_p_shapes, tt_q_shapes, tt_ranks, tt_cores, [1, 0, 2, 3]
            )
            # tt_emb
            output = tt_emb(indices, offsets)
            output.backward(d_output)
            # new_optimizer_state = []
            # new_optimizer_state = [torch.mul(t.grad, t.grad) for t in tt_cores]
            # new_tt_cores = []
            # new_tt_cores = [
            #     (
            #         t
            #         - torch.div(
            #             t.grad * learning_rate, torch.sqrt(new_optimizer_state[i]) + eps
            #         )
            #     )
            #     for i, t in enumerate(tt_cores)
            # ]

        else:
            emb = torch.nn.EmbeddingBag(
                num_embeddings,
                embedding_dim,
                sparse=True,
                mode="sum",
                _weight=tt_emb.full_weight(),
                include_last_offset=True,
            )
            emb.to(device)
            # reference
            output_ref = emb(indices.long(), offsets.long())
            output_ref.backward(d_output)
            # d_weight_ref = emb.weight.grad.to_dense()
            # full_weight.backward(d_weight_ref)

        
        # for i in range(tt_ndims):
        #     torch.testing.assert_allclose(
        #         tt_emb.optimizer_state[i], new_optimizer_state[i]
        #     )
        #     torch.testing.assert_allclose(tt_emb.tt_cores[i], new_tt_cores[i])

def test_fwd(indices, offsets, device, ctx, args):
    print("Forward Test Started!")
    tt_ndims, tt_ranks, tt_p_shapes, tt_q_shapes, num_embeddings, embedding_dim = ctx
    test_bb = TestTTEmbeddingBag(args.batch, tt_ndims, tt_ranks, tt_p_shapes, tt_q_shapes, device, args.use_tt)
    start = time.time()
    test_bb.test_forward(indices, offsets)
    end = time.time()
    print("Use TT: ", args.use_tt)
    print("Done with time: %.2fs" % (end - start))

def test_bwd(indices, offsets, device, ctx, args):
    print("Backward Test Started!")
    tt_ndims, tt_ranks, tt_p_shapes, tt_q_shapes, num_embeddings, embedding_dim = ctx
    test_bb = TestTTEmbeddingBag(args.batch, tt_ndims, tt_ranks, tt_p_shapes, tt_q_shapes, device, args.use_tt)
    start = time.time()
    test_bb.test_backward_dense(indices, offsets)
    # test_bb.test_backward_sgd(indices, offsets)
    # test_bb.test_backward_adagrad(indices, offsets)
    end = time.time()
    print("Use TT: ", args.use_tt)
    print("Done with time: %.2fs" % (end - start))

def test_sage_tt_bwd(train_loader, data, ctx, args):
    print("SAGE Forward Test Started!")
    # Unpack data
    train_nid, val_nid, test_nid, in_feats, labels, n_classes, nfeat, g = data
    tt_ndims, tt_ranks, tt_p_shapes, tt_q_shapes = ctx


def test_sage_tt_fwd(block, input_nodes, offsets, ctx, args):
    print("SAGE Forward Test Started!")
    
    # Unpack data
    tt_ndims, tt_ranks, tt_p_shapes, tt_q_shapes, num_embeddings, embedding_dim = ctx
    n_classes = 50

    # Define model and optimizer
    device = torch.device(args.device)
    model = SAGE_ONLY(
        num_nodes = num_embeddings, 
        in_feats = embedding_dim, 
        n_hidden = args.num_hidden, 
        n_classes = n_classes,
        n_layers = 1,
        # n_layers = args.num_layers, 
        activation = F.relu, 
        dropout = args.dropout, 
        graph = None, 
        device = args.device)
    model = model.to(device)
    
    emb = TTEmbeddingBag(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                tt_p_shapes=tt_p_shapes,
                tt_q_shapes=tt_q_shapes,
                tt_ranks=tt_ranks,
                sparse=False,
                use_cache=False,
                weight_dist="uniform")
    emb.to(device)
    
    # emb = torch.nn.Embedding(num_embeddings, embedding_dim)
    # emb.to(device)

    blocks = [block.to(device), block.to(device), block.to(device)]    
    offsets = torch.arange(input_nodes.shape[0] + 1).to(device)
    input_nodes = input_nodes.to(device)
    
    start = time.time()
    h = emb(input_nodes, offsets)
    # h = emb(input_nodes)
    end = time.time()
    print("TT fwd time: %.2fs" % (end - start))

    start = time.time()
    h = model.forward(blocks, h)
    end = time.time()
    print("SAGE fwd time: %.2fs" % (end - start))

def graph_to_block(graph):
    src, dst = graph.edges()
    
    # Create a DGL block
    block = dgl.heterograph(
        {('_N', '_E', '_N'): (src, dst)},
        num_nodes_dict={'_N': graph.number_of_nodes()}
    )
    
    # Copy node and edge data from the original graph to the block
    for key, value in graph.ndata.items():
        block.srcdata[key] = value
    for key, value in graph.edata.items():
        block.edata[key] = value    
    return block


def reorder_block_with_metis(block):
    src, dst = block.edges()
    
    # Create a subgraph from the block
    subgraph = dgl.graph((src, dst), num_nodes=block.number_of_src_nodes())
    subgraph.ndata[dgl.NID] = block.srcdata[dgl.NID]
    subgraph.edata[dgl.EID] = block.edata[dgl.EID]
    
    # Apply METIS partitioning to reorder the subgraph
    partitioned_subgraph = dgl.reorder_graph(subgraph, 'metis', permute_config={'k':3})
    # partitioned_subgraph = transform.metis_partition(subgraph, k=1)[0]
    
    # Get the reordered node indices
    reordered_indices = partitioned_subgraph.ndata[dgl.NID]
    
    # Create a mapping from the reordered indices to the original node IDs
    original_indices = block.srcdata[dgl.NID][reordered_indices]
    
    # Reorder block node and edge data
    reordered_srcdata = {key: block.srcdata[key][reordered_indices] for key in block.srcdata.keys()}
    reordered_dstdata = {key: block.dstdata[key][reordered_indices] for key in block.dstdata.keys()}
    reordered_edata = {key: block.edata[key] for key in block.edata.keys()}

    # Create a reordered block
    reordered_block = dgl.heterograph(
        {('_N', '_E', '_N'): (partitioned_subgraph.edges()[0], partitioned_subgraph.edges()[1])},
        num_nodes_dict={'_N': block.number_of_nodes()}
    )

    # Assign reordered node and edge data
    for key, value in reordered_srcdata.items():
        reordered_block.srcdata[key] = value
    for key, value in reordered_dstdata.items():
        reordered_block.dstdata[key] = value
    for key, value in reordered_edata.items():
        reordered_block.edata[key] = value

    return reordered_block, original_indices

def run_one_sage(train_loader, data, args):
    # Unpack data
    train_nid, val_nid, test_nid, in_feats, labels, n_classes, nfeat, g = data

    # Define model and optimizer
    device = torch.device(args.device)
    
    model = SAGE(
        num_nodes = g.number_of_nodes(), 
        in_feats = in_feats, 
        n_hidden = args.num_hidden, 
        n_classes = n_classes, 
        n_layers = args.num_layers,
        activation = F.relu, 
        dropout = args.dropout, 
        use_tt = args.use_tt, 
        tt_rank = [int(i) for i in args.tt_rank.split(',')],
        p_shapes = [int(i) for i in args.p_shapes.split(',')],
        q_shapes = [int(i) for i in args.q_shapes.split(',')],
        dist = args.init, 
        graph = g, 
        device = args.device,
        embed_name = args.emb_name,
        access_counts=args.access_counts,
        use_cached=args.use_cached,
        cache_size = args.cache_size,
        batch_count = args.batch_count)
    
    model = model.to(device)

    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.8,
                                                                  patience=800, verbose=True)
    
    iter_dataloader = iter(train_loader)

    input_nodes, seeds, blocks = next(iter_dataloader)
    blocks = [blk.int().to(device) for blk in blocks]


    ### Partitioning the graph
    blocks = [blk.int() for blk in blocks]
    first_block = blocks[0]
    block = blocks[0]
    subgraph = dgl.block_to_graph(first_block)
    subgraph = dgl.graph((first_block.edges()[0], first_block.edges()[1]), idtype=th.int64)
    partitioned_subgraph = dgl.reorder_graph(subgraph, 'metis', permute_config={'k':100})
    reordered_indices = partitioned_subgraph.ndata[dgl.NID]
    block = dgl.to_block(partitioned_subgraph, dst_nodes=partitioned_subgraph.ndata[dgl.NID], src_nodes=first_block.srcdata[dgl.NID])
    block = block.to(device)


    ### Running all the blocks, too slow...
    batch_inputs = nfeat[input_nodes]
    batch_labels = labels[seeds]
    batch_labels = batch_labels.to(device)

    # batch_inputs = nfeat[reordered_indices]
    batch_inputs = batch_inputs.to(device)

    start_time = time.time()
    print("input nodes Shape: ", input_nodes.shape)
    batch_pred = model(blocks, batch_inputs)
    print("batch output Shape: ", batch_pred.shape)
    print("FWD Runtime: ", time.time() - start_time)
    
    start_time_bwd = time.time()
    loss = loss_fcn(batch_pred, batch_labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    lr_scheduler.step(loss)
    print("BWD Runtime: ", time.time() - start_time_bwd)
    print("Total Runtime: ", time.time() - start_time)
    print("--- Run Once Done! ---")

def run_one(train_loader, data, args, total_time=[]):
    # Unpack data
    train_nid, val_nid, test_nid, in_feats, labels, n_classes, nfeat, g = data

    # Define model and optimizer
    device = torch.device(args.device)
    
    model = SAGE(
        num_nodes = g.number_of_nodes(), 
        in_feats = in_feats, 
        n_hidden = args.num_hidden, 
        n_classes = n_classes, 
        n_layers = args.num_layers,
        activation = F.relu, 
        dropout = args.dropout, 
        use_tt = args.use_tt, 
        tt_rank = [int(i) for i in args.tt_rank.split(',')],
        p_shapes = [int(i) for i in args.p_shapes.split(',')],
        q_shapes = [int(i) for i in args.q_shapes.split(',')],
        dist = args.init, 
        graph = g, 
        device = args.device,
        embed_name = args.emb_name,
        access_counts=args.access_counts,
        use_cached=args.use_cached,
        cache_size = args.cache_size,
        batch_count = args.batch_count)
    
    model = model.to(device)

    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.8,
                                                                  patience=800, verbose=True)
    
    # model.embed_layer.register_forward_hook(hook)
    # epoch_start_time = time.time()
    for step, (input_nodes, seeds, blocks) in enumerate(train_loader):
        
        sorted_input_nodes, sorted_indices = torch.sort(input_nodes)

        blocks = [blk.int().to(device) for blk in blocks]
        batch_inputs = nfeat[input_nodes]
        batch_labels = labels[seeds]
        batch_labels = batch_labels.to(device)
        batch_inputs = batch_inputs.to(device)
        
        start_time = time.time()
        # print("input nodes Shape: ", input_nodes.shape)
        batch_pred = model(blocks, batch_inputs)

        ### Analyze the embedding indices for continuity
        embedding_indices_sorted = np.sort(embedding_indices)
        
        # print("batch output Shape: ", batch_pred.shape)
        # print("FWD Runtime: ", time.time() - start_time)
        
        start_time_bwd = time.time()
        loss = loss_fcn(batch_pred, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step(loss)

        total_time.append(time.time() - start_time)
        break
        # print("BWD Runtime: ", time.time() - start_time_bwd)
        # print("Total Runtime: ", time.time() - start_time)
        
    
    # print("Epoch Stop Total Runtime: ", time.time() - epoch_start_time)
    # print("--- Run Once Done! ---")
    return total_time

if __name__ == '__main__':
    args = parse_args()
    
    if args.device != 'cpu' and torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device('cpu')

    target_dataset = args.dataset
    if args.workspace is not None:
        root = os.path.join(os.environ['HOME'], args.workspace, 'gnn_related', 'dataset')
    else:
        root = args.data_dir
    
    ### parametrs for profiling
    tt_ndims = 3
    tt_ranks = [int(i) for i in args.tt_rank.split(',')]
    tt_p_shapes = [int(i) for i in args.p_shapes.split(',')]
    tt_q_shapes = [int(i) for i in args.q_shapes.split(',')]
    num_embeddings = np.prod(np.array(tt_p_shapes))
    embedding_dim = np.prod(np.array(tt_q_shapes))
    pooling_factor = 10
    pooling_factor_std = 20

    print("---- Num Embeddings: ", num_embeddings)
    print("---- Embeddings Dim: ", embedding_dim)
    print("Memory Usage: ", num_embeddings * embedding_dim * 4 / 1024 / 1024, "MB")
    print("Memory Usage: ", num_embeddings * embedding_dim * 4 / 1024 / 1024 / 1024, "GB")

    if args.run_one:
        print("Running with one forward and backward pass with true data")
        
        ### True graph
        train_loader, full_neighbor_loader, data = dgl_graph_loader(target_dataset, root, device, args)

        ### case - 1: run with one fwd and bwd pass
        total_time = run_one(train_loader, data, args)
        print("Mean Time: ", total_time)
        # total_time = []
        # for i in range(20):
        #     total_ = run_one(train_loader, data, args)
        #     total_time.append(total_[0])
        # # 5.5250 - sorted, 5.3221 - unsorted
        # print("Mean Time: ", np.mean(total_time))

    else: 
        ### Packing the parameters
        ctx = tt_ndims, tt_ranks, tt_p_shapes, tt_q_shapes, num_embeddings, embedding_dim

        ### Random sparse feature generation
        _, indices, offsets, _ = generate_sparse_feature(
                args.batch,
                num_embeddings=num_embeddings,
                pooling_factor=float(pooling_factor),
                pooling_factor_std=float(pooling_factor_std),
                generate_scores=False,
                unary=False,
                unique=False,
            )

        print(len(indices), len(offsets))

        num_src_nodes = len(indices)
        num_dst_nodes = int(len(indices) * 0.4)
        num_edges = int(len(indices) * 0.4)

        ### Graph creation
        graph = create_graph(num_src_nodes, num_dst_nodes, num_edges)

        ### Random block creation
        block = create_block(num_src_nodes, num_dst_nodes, num_edges)
        
        input_nodes = torch.arange(0, block.number_of_src_nodes())
        offsets = torch.arange(input_nodes.shape[0] + 1)
        print("Num. of input nodes: ", input_nodes.shape[0])
        print("Block: ", block)

        par_g = dgl.reorder_graph(graph, 'metis', permute_config={'k':args.partition})

        ### case - 1.1: run with three layers with GraphSAGE
        # run_one_sage(train_loader, data, args)

        ### case - 2: run with one fwd pass (fbtt)
        test_fwd(indices, offsets, device, ctx, args)

        ### case - 3: run with one bwd pass (fbtt)
        test_bwd(indices, offsets, device, ctx, args)

        ### case - 4: run with one fwd pass (sage)
        test_sage_tt_fwd(block, input_nodes, offsets, ctx, args)

