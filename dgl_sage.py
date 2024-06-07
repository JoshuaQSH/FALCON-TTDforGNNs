import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn.pytorch as dglnn
import tqdm

import torch
import sys
from tt_utils import *
from torch.utils.cpp_extension import load
from FBTT.tt_embeddings_ops import TTEmbeddingBag
from Efficient_TT.efficient_tt import Eff_TTEmbedding

# Eff_TT_embedding_cuda = load(name="efficient_tt_table", sources=[
#     "/home/shenghao/tensor-train-for-gcn/TT4GNN/Efficient_TT/efficient_kernel_wrap.cpp", 
#     "/home/shenghao/tensor-train-for-gcn/TT4GNN/Efficient_TT/efficient_tt_cuda.cu", 
#     ], verbose=True)


class LoggingEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(LoggingEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.access_counts = th.zeros(num_embeddings, dtype=th.long)

    def forward(self, input):
        self.log_accesses(input)
        return self.embedding(input)
    
    def log_accesses(self, indices):
        unique_indices, counts = indices.unique(return_counts=True)
        self.access_counts[unique_indices] += counts

    def get_access_counts(self):
        return self.access_counts

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
                 sparse=False,
                 batch_count=1000):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        self.num_nodes= num_nodes
        self.use_tt = use_tt
        # Hacked for now
        self.device = th.device(device)
        self.embed_name = embed_name

        self.access_counts = access_counts
        self.use_cached = use_cached
        self.batch_count = batch_count

        # cache map
        if self.use_cached:
            # default: 10% of the embeddings (num_nodes, also known as num_embeddings)
            self.cache_size = int(0.1 * cache_size * num_nodes)
            # default: num_embeddings
            self.hashtbl_size = num_nodes
        else:
            self.cache_size = 0
            self.hashtbl_size = 0
        
        self.cache = TensorCache(cache_size, in_feats, self.device)
        
        if use_tt and device != 'cpu':            
            if device != 'cpu':
                if self.embed_name == "fbtt":
                    print("Using FBTT")
                    print("---- num_embeddings: ", num_nodes)
                    print("---- embedding_dim: ", in_feats)
                    self.embed_layer = TTEmbeddingBag(
                            num_embeddings=num_nodes,
                            embedding_dim=in_feats,
                            tt_ranks=tt_rank,
                            tt_p_shapes=p_shapes, # The factorization of num_embeddings
                            tt_q_shapes=q_shapes, # Same as the in_feats
                            sparse=sparse,
                            use_cache=self.use_cached,
                            cache_size=self.cache_size,
                            hashtbl_size=self.hashtbl_size,
                            weight_dist="normal",
                            batch_count=self.batch_count,
                            )
                # TODO: hardcoded for the batch_size - default 1024
                elif self.embed_name == "eff":
                    print("Using Efficient TT")
                    self.embed_layer = Eff_TTEmbedding(
                        num_embeddings = num_nodes,
                        embedding_dim = in_feats,
                        tt_p_shapes=p_shapes,
                        tt_q_shapes=q_shapes,
                        tt_ranks = tt_rank,
                        weight_dist = "uniform",
                        batch_size = 1024
                    ).to(self.device)
                else:
                    print("Unknown embedding type")                                
            
            if dist == 'eigen':
                eigen_vals, eigen_vecs = get_eigen(graph, in_feats, name='ogbn-products')
                eigen_vecs = th.tensor(eigen_vecs * np.sqrt(eigen_vals).reshape((1, len(eigen_vals))), dtype=th.float32)
                emb_pad = np.zeros(shape=(125 * 140 * 140, 100)).astype(np.float32)
                emb_pad[:eigen_vecs.shape[0], :] = eigen_vecs

                tt_cores, _ = tt_matrix_decomp(
                        emb_pad,
                        [1, tt_rank[0], tt_rank[1], 1],
                        p_shapes,
                        [4, 5, 5]
                    )
                for i in range(3):
                    self.embed_layer.tt_cores[i].data = tt_cores[i].to(self.device)

            # TODO: init Here
            elif dist == 'ortho':
                print("initialized from orthogonal cores")
                tt_cores = get_ortho(
                    [1, tt_rank[0], tt_rank[1], 1],
                    p_shapes,
                    [4, 5, 5]
                )
                # TODO: initialized the tt_cores weights
                for i in range(3):
                    self.embed_layer.tt_cores[i].data = th.tensor(tt_cores[i]).to(device)
                    
            elif dist == 'dortho':
                print('initialized from decomposing orthogonal matrix')
                rand_A = np.random.random(size=(125 * 140 * 140, 100)).astype(np.float32)
                emb_w, _ = np.linalg.qr(rand_A)
                print('ortho', emb_w.shape)

                tt_cores, _ = tt_matrix_decomp(
                        emb_w,
                        [1, tt_rank[0], tt_rank[1], 1],
                        p_shapes,
                        [4, 5, 5]
                    )
                for i in range(3):
                    self.embed_layer.tt_cores[i].data = tt_cores[i].to(self.device)
            else:
                print("No initialization for weights")
        
        # A cpu version will call nn.Embedding, with or withhout the access counts
        else:
            if self.access_counts:
                print("Logging Embedding and counting access")
                self.embed_layer = LoggingEmbedding(num_nodes, in_feats)
                    
            else:
                print("Using CPU with torch Embedding")
                self.embed_layer = th.nn.Embedding(num_nodes, in_feats)   
    
    # A torch level cache for embedding lookup
    def embedding_lookup(self, node_ids, offsets):
        embeddings = []
        for idx, node_id in enumerate(node_ids):
            cached_embedding = self.cache.get(node_id.item())
            if cached_embedding is None:
                computed_embedding = self.embed_layer(node_ids[idx:idx+1], torch.tensor([0, 1], device=self.device))
                self.cache.put(node_id.item(), computed_embedding)
                embeddings.append(computed_embedding)
            else:
                embeddings.append(cached_embedding)
        return th.stack(embeddings, dim=0).squeeze(1)
    
    def embedding_lookup_cpu(self, node_ids):
        embeddings = [] 
        for node_id in node_ids:
            embedding = self.cache.get(node_id.item())
            if embedding is None:
                embedding = self.embed_layer(node_id)
                self.cache.put(node_id.item(), embedding)
            embeddings.append(embedding)
        return th.stack(embeddings)

    def forward(self, blocks, input_nodes):
        # h = x
        if self.use_tt and self.device.type != 'cpu':
            offsets = th.arange(input_nodes.shape[0] + 1).to(self.device)
            input_nodes = input_nodes.to(self.device)
            
            if self.use_cached:
                h = self.embed_layer(input_nodes, offsets)
                # h = self.embedding_lookup(input_nodes, offsets)
            else:
                h = self.embed_layer(input_nodes, offsets)

        elif self.device.type == 'cpu':
            if self.use_cached:
                # h = self.embed_layer(input_nodes.to(self.device))
                h = self.embedding_lookup_cpu(input_nodes.to(self.device))
            else:
                h = self.embed_layer(input_nodes.to(self.device))
        
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            # We need to first copy the representation of nodes on the RHS from the
            # appropriate nodes on the LHS.
            # Note that the shape of h is (num_nodes_LHS, D) and the shape of h_dst
            # would be (num_nodes_RHS, D)
            h_dst = h[:block.num_dst_nodes()]
            # Then we compute the updated representation on the RHS.
            # The shape of h now becomes (num_nodes_RHS, D)
            h = layer(block, (h, h_dst))
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def inference(self, g, device, dataloader):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.
        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        if self.use_tt and device.type != 'cpu':
            ids = th.arange(g.num_nodes()).to(device)
            offsets = th.arange(g.num_nodes() + 1).to(device)
            x = self.embed_layer(ids.to(device), offsets)
        else:
            ids = th.arange(g.num_nodes()).to(device)
            x = self.embed_layer(ids.to(device))

        for l, layer in enumerate(self.layers):
            y = th.zeros(g.num_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes).to(device)

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0].int().to(device)

                h = x[input_nodes]

                h_dst = h[:block.num_dst_nodes()]
                h = layer(block, (h, h_dst))
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes] = h

            x = y
        return y
