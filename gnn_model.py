import dgl
import dgl.nn.pytorch as dglnn
from dgl import function as fn
from dgl._ffi.base import DGLError
from dgl.nn.pytorch.utils import Identity
from dgl.nn.functional import edge_softmax
from dgl.utils import expand_as_pair

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import tqdm
import numpy as np
from tt_utils import *
from FBTT.tt_embeddings_ops import TTEmbeddingBag

from ogb.graphproppred.mol_encoder import AtomEncoder
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims 

# from conv import GNN_node, GNN_node_Virtualnode
from dgl.nn.pytorch import (
    AvgPooling,
    GlobalAttentionPooling,
    MaxPooling,
    Set2Set,
    SumPooling,
)

### Counting the embedding accesses
class LoggingEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(LoggingEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.access_counts = torch.zeros(num_embeddings, dtype=torch.long)

    def forward(self, input):
        self.log_accesses(input)
        return self.embedding(input)
    
    def log_accesses(self, indices):
        unique_indices, counts = indices.unique(return_counts=True)
        self.access_counts[unique_indices] += counts

    def get_access_counts(self):
        return self.access_counts

### GraphSAGE Model with TTD
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
                 init=None,
                 graph=None,
                 device='cpu',
                 embed_name ='fbtt',
                 access_counts=False,
                 use_cached=False,
                 cache_size=0,
                 sparse=False,
                 batch_count=1000,
                 weights_init=None):
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
        self.device = torch.device(device)
        self.embed_name = embed_name

        self.access_counts = access_counts
        self.use_cached = use_cached
        self.batch_count = batch_count

        # cache map
        if self.use_cached:
            # default: 10% of the embeddings (num_nodes, also known as num_embeddings)
            self.cache_size = int(0.01 * cache_size * num_nodes)
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
                else:
                    print("Unknown embedding type")                                
            
            if init == 'eigen':
                eigen_vals, eigen_vecs = get_eigen(graph, in_feats, name='ogbn-products')
                eigen_vecs = torch.tensor(eigen_vecs * np.sqrt(eigen_vals).reshape((1, len(eigen_vals))), dtype=torch.float32)
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
            
            # autotune
            elif init == 'auto':
                if weights_init is not None:
                    print("Using the auto-tuned weights")
                    for i in range(3):
                        self.embed_layer.tt_cores[i].data = torch.tensor(weights_init[i]).to(torch.float32).to(device)
                else:
                    print("No initialization for weights")

            # TODO: init Here
            elif init == 'ortho':
                print("initialized from orthogonal cores")
                tt_cores = get_ortho(
                    [1, tt_rank[0], tt_rank[1], 1],
                    p_shapes,
                    [4, 5, 5]
                )
                # TODO: initialized the tt_cores weights
                for i in range(3):
                    self.embed_layer.tt_cores[i].data = torch.tensor(tt_cores[i]).to(device)
    
            elif init == 'dortho':
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
                self.embed_layer = torch.nn.Embedding(num_nodes, in_feats)   


    def forward(self, blocks, input_nodes):
        # h = input_nodes
        if self.device.type == 'cpu':  
            h = self.embed_layer(input_nodes.to(self.device))
        
        elif self.use_tt and self.device.type != 'cpu':
            offsets = torch.arange(input_nodes.shape[0] + 1).to(self.device)
            input_nodes = input_nodes.to(self.device)
            if self.use_cached:
                h = self.embed_layer(input_nodes, offsets)
            else:
                h = self.embed_layer(input_nodes, offsets)
        
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
            ids = torch.arange(g.num_nodes()).to(device)
            offsets = torch.arange(g.num_nodes() + 1).to(device)
            x = self.embed_layer(ids.to(device), offsets)
        else:
            ids = torch.arange(g.num_nodes()).to(device)
            x = self.embed_layer(ids.to(device))

        for l, layer in enumerate(self.layers):
            y = torch.zeros(g.num_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes).to(device)

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

class Bias(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.bias = nn.Parameter(torch.Tensor(size))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return x + self.bias

### GCN Model
class GCN(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout, use_linear):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.use_linear = use_linear

        self.convs = nn.ModuleList()
        if use_linear:
            self.linear = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(n_layers):
            in_hidden = n_hidden if i > 0 else in_feats
            out_hidden = n_hidden if i < n_layers - 1 else n_classes
            bias = i == n_layers - 1

            self.convs.append(dglnn.GraphConv(in_hidden, out_hidden, "both", bias=bias, allow_zero_in_degree=True))
            if use_linear:
                self.linear.append(nn.Linear(in_hidden, out_hidden, bias=False))
            if i < n_layers - 1:
                self.bns.append(nn.BatchNorm1d(out_hidden))

        self.dropout0 = nn.Dropout(min(0.1, dropout))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, graph, feat):
        h = feat
        h = self.dropout0(h)

        for i in range(self.n_layers):
            conv = self.convs[i](graph, h)

            if self.use_linear:
                linear = self.linear[i](h)
                h = conv + linear
            else:
                h = conv

            if i < self.n_layers - 1:
                h = self.bns[i](h)
                h = self.activation(h)
                h = self.dropout(h)

        return h

### GAT Layer define
class GATConv(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        num_heads=1,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=False,
        activation=None,
        allow_zero_in_degree=False,
        norm="none",
    ):
        super(GATConv, self).__init__()
        if norm not in ("none", "both"):
            raise DGLError('Invalid norm value. Must be either "none", "both".' ' But got "{}".'.format(norm))
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._norm = norm
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer("res_fc", None)
        self.reset_parameters()
        self._activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        if hasattr(self, "fc"):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat):
        with graph.local_scope():
            # if not self._allow_zero_in_degree:
            #     if (graph.in_degrees() == 0).any():
            #         assert False

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, "fc_src"):
                    self.fc_src, self.fc_dst = self.fc, self.fc
                feat_src, feat_dst = h_src, h_dst
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src, feat_dst = h_src, h_dst
                feat_src = feat_dst = self.fc(h_src).view(-1, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[: graph.number_of_dst_nodes()]

            if self._norm == "both":
                degs = graph.out_degrees().float().clamp(min=1)
                norm = torch.pow(degs, -0.5)
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = torch.reshape(norm, shp)
                feat_src = feat_src * norm

            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({"ft": feat_src, "el": el})
            graph.dstdata.update({"er": er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v("el", "er", "e"))
            e = self.leaky_relu(graph.edata.pop("e"))
            # compute softmax
            graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))
            # message passing
            graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            rst = graph.dstdata["ft"]

            if self._norm == "both":
                degs = graph.in_degrees().float().clamp(min=1)
                norm = torch.pow(degs, 0.5)
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = torch.reshape(norm, shp)
                rst = rst * norm

            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval
            # activation
            if self._activation is not None:
                rst = self._activation(rst)
            return rst

### GAT Model
class GAT(nn.Module):
    def __init__(
        self, in_feats, n_classes, n_hidden, n_layers, n_heads, activation, dropout=0.0, attn_drop=0.0, norm="none"
    ):
        super().__init__()
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.num_heads = n_heads

        self.convs = nn.ModuleList()
        self.linear = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.biases = nn.ModuleList()

        for i in range(n_layers):
            in_hidden = n_heads * n_hidden if i > 0 else in_feats
            out_hidden = n_hidden if i < n_layers - 1 else n_classes
            # in_channels = n_heads if i > 0 else 1
            out_channels = n_heads

            self.convs.append(GATConv(in_hidden, out_hidden, num_heads=n_heads, attn_drop=attn_drop, norm=norm))

            self.linear.append(nn.Linear(in_hidden, out_channels * out_hidden, bias=False))
            if i < n_layers - 1:
                self.bns.append(nn.BatchNorm1d(out_channels * out_hidden))

        self.bias_last = Bias(n_classes)

        self.dropout0 = nn.Dropout(min(0.1, dropout))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, graph, feat):
        h = feat
        h = self.dropout0(h)

        for i in range(self.n_layers):
            conv = self.convs[i](graph, h)
            linear = self.linear[i](h).view(conv.shape)

            h = conv + linear

            if i < self.n_layers - 1:
                h = h.flatten(1)
                h = self.bns[i](h)
                h = self.activation(h)
                h = self.dropout(h)

        h = h.mean(1)
        h = self.bias_last(h)

        return h

class ExternalNodeCollator(dgl.dataloading.NodeCollator):
    def __init__(self, g, idx, sampler, offset, feats, label):
        super().__init__(g, idx, sampler)
        self.offset = offset
        self.feats = feats
        self.label = label

    def collate(self, items):
        input_nodes, output_nodes, mfgs = super().collate(items)
        # Copy input features
        mfgs[0].srcdata["x"] = torch.FloatTensor(self.feats[input_nodes])
        mfgs[-1].dstdata["y"] = torch.LongTensor(
            self.label[output_nodes - self.offset]
        )
        return input_nodes, output_nodes, mfgs

### RGAT Model
class RGAT(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        num_etypes,
        num_layers,
        num_heads,
        dropout,
        pred_ntype,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.skips = nn.ModuleList()

        self.convs.append(
            nn.ModuleList(
                [
                    dglnn.GATConv(
                        in_channels,
                        hidden_channels // num_heads,
                        num_heads,
                        allow_zero_in_degree=True,
                    )
                    for _ in range(num_etypes)
                ]
            )
        )
        self.norms.append(nn.BatchNorm1d(hidden_channels))
        self.skips.append(nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(
                nn.ModuleList(
                    [
                        dglnn.GATConv(
                            hidden_channels,
                            hidden_channels // num_heads,
                            num_heads,
                            allow_zero_in_degree=True,
                        )
                        for _ in range(num_etypes)
                    ]
                )
            )
            self.norms.append(nn.BatchNorm1d(hidden_channels))
            self.skips.append(nn.Linear(hidden_channels, hidden_channels))

        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )
        self.dropout = nn.Dropout(dropout)

        self.hidden_channels = hidden_channels
        self.pred_ntype = pred_ntype
        self.num_etypes = num_etypes

    def forward(self, mfgs, x):
        for i in range(len(mfgs)):
            mfg = mfgs[i]
            x_dst = x[: mfg.num_dst_nodes()]
            n_src = mfg.num_src_nodes()
            n_dst = mfg.num_dst_nodes()
            mfg = dgl.block_to_graph(mfg)
            x_skip = self.skips[i](x_dst)
            for j in range(self.num_etypes):
                subg = mfg.edge_subgraph(
                    mfg.edata["etype"] == j, relabel_nodes=False
                )
                x_skip += self.convs[i][j](subg, (x, x_dst)).view(
                    -1, self.hidden_channels
                )
            x = self.norms[i](x_skip)
            x = F.elu(x)
            x = self.dropout(x)
        return self.mlp(x)



class BondEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__()

        full_bond_feature_dims = get_bond_feature_dims()

        self.bond_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_bond_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:,i])

        return bond_embedding

### GIN convolution along the graph structure
class GINConv(nn.Module):
    def __init__(self, emb_dim):
        """
        emb_dim (int): node embedding dimensionality
        """

        super(GINConv, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.eps = nn.Parameter(torch.Tensor([0]))

        self.bond_encoder = BondEncoder(emb_dim=emb_dim)

    def forward(self, g, x, edge_attr):
        with g.local_scope():
            edge_embedding = self.bond_encoder(edge_attr)
            g.ndata["x"] = x
            g.apply_edges(fn.copy_u("x", "m"))
            g.edata["m"] = F.relu(g.edata["m"] + edge_embedding)
            g.update_all(fn.copy_e("m", "m"), fn.sum("m", "new_x"))
            out = self.mlp((1 + self.eps) * x + g.ndata["new_x"])

            return out


### GCN convolution along the graph structure
class GCNConv(nn.Module):
    def __init__(self, emb_dim):
        """
        emb_dim (int): node embedding dimensionality
        """

        super(GCNConv, self).__init__()

        self.linear = nn.Linear(emb_dim, emb_dim)
        self.root_emb = nn.Embedding(1, emb_dim)
        self.bond_encoder = BondEncoder(emb_dim=emb_dim)

    def forward(self, g, x, edge_attr):
        with g.local_scope():
            x = self.linear(x)
            edge_embedding = self.bond_encoder(edge_attr)

            # Molecular graphs are undirected
            # g.out_degrees() is the same as g.in_degrees()
            degs = (g.out_degrees().float() + 1).to(x.device)
            norm = torch.pow(degs, -0.5).unsqueeze(-1)  # (N, 1)
            g.ndata["norm"] = norm
            g.apply_edges(fn.u_mul_v("norm", "norm", "norm"))

            g.ndata["x"] = x
            g.apply_edges(fn.copy_u("x", "m"))
            g.edata["m"] = g.edata["norm"] * F.relu(
                g.edata["m"] + edge_embedding
            )
            g.update_all(fn.copy_e("m", "m"), fn.sum("m", "new_x"))
            out = g.ndata["new_x"] + F.relu(
                x + self.root_emb.weight
            ) * 1.0 / degs.view(-1, 1)

            return out


### GNN to generate node embedding
class GNN_node(nn.Module):
    """
    Output:
        node representations
    """

    def __init__(
        self,
        num_layers,
        emb_dim,
        drop_ratio=0.5,
        JK="last",
        residual=False,
        gnn_type="gin",
    ):
        """
        num_layers (int): number of GNN message passing layers
        emb_dim (int): node embedding dimensionality
        """

        super(GNN_node, self).__init__()
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)

        ###List of GNNs
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for layer in range(num_layers):
            if gnn_type == "gin":
                self.convs.append(GINConv(emb_dim))
            elif gnn_type == "gcn":
                self.convs.append(GCNConv(emb_dim))
            else:
                ValueError("Undefined GNN type called {}".format(gnn_type))

            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

    def forward(self, g, x, edge_attr):
        ### computing input node embedding
        h_list = [self.atom_encoder(x)]
        for layer in range(self.num_layers):
            h = self.convs[layer](g, h_list[layer], edge_attr)
            h = self.batch_norms[layer](h)

            if layer == self.num_layers - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(
                    F.relu(h), self.drop_ratio, training=self.training
                )

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layers):
                node_representation += h_list[layer]

        return node_representation


### Virtual GNN to generate node embedding
class GNN_node_Virtualnode(nn.Module):
    """
    Output:
        node representations
    """

    def __init__(
        self,
        num_layers,
        emb_dim,
        drop_ratio=0.5,
        JK="last",
        residual=False,
        gnn_type="gin",
    ):
        """
        num_layers (int): number of GNN message passing layers
        emb_dim (int): node embedding dimensionality
        """

        super(GNN_node_Virtualnode, self).__init__()
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)

        ### set the initial virtual node embedding to 0.
        self.virtualnode_embedding = nn.Embedding(1, emb_dim)
        nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        ### List of GNNs
        self.convs = nn.ModuleList()
        ### batch norms applied to node embeddings
        self.batch_norms = nn.ModuleList()

        ### List of MLPs to transform virtual node at every layer
        self.mlp_virtualnode_list = nn.ModuleList()

        for layer in range(num_layers):
            if gnn_type == "gin":
                self.convs.append(GINConv(emb_dim))
            elif gnn_type == "gcn":
                self.convs.append(GCNConv(emb_dim))
            else:
                ValueError("Undefined GNN type called {}".format(gnn_type))

            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

        for layer in range(num_layers - 1):
            self.mlp_virtualnode_list.append(
                nn.Sequential(
                    nn.Linear(emb_dim, emb_dim),
                    nn.BatchNorm1d(emb_dim),
                    nn.ReLU(),
                    nn.Linear(emb_dim, emb_dim),
                    nn.BatchNorm1d(emb_dim),
                    nn.ReLU(),
                )
            )
        self.pool = SumPooling()

    def forward(self, g, x, edge_attr):
        ### virtual node embeddings for graphs
        virtualnode_embedding = self.virtualnode_embedding(
            torch.zeros(g.batch_size).to(x.dtype).to(x.device)
        )

        h_list = [self.atom_encoder(x)]
        batch_id = dgl.broadcast_nodes(
            g, torch.arange(g.batch_size).to(x.device)
        )
        for layer in range(self.num_layers):
            ### add message from virtual nodes to graph nodes
            h_list[layer] = h_list[layer] + virtualnode_embedding[batch_id]

            ### Message passing among graph nodes
            h = self.convs[layer](g, h_list[layer], edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layers - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(
                    F.relu(h), self.drop_ratio, training=self.training
                )

            if self.residual:
                h = h + h_list[layer]

            h_list.append(h)

            ### update the virtual nodes
            if layer < self.num_layers - 1:
                ### add message from graph nodes to virtual nodes
                virtualnode_embedding_temp = (
                    self.pool(g, h_list[layer]) + virtualnode_embedding
                )
                ### transform virtual nodes using MLP
                virtualnode_embedding_temp = self.mlp_virtualnode_list[layer](
                    virtualnode_embedding_temp
                )

                if self.residual:
                    virtualnode_embedding = virtualnode_embedding + F.dropout(
                        virtualnode_embedding_temp,
                        self.drop_ratio,
                        training=self.training,
                    )
                else:
                    virtualnode_embedding = F.dropout(
                        virtualnode_embedding_temp,
                        self.drop_ratio,
                        training=self.training,
                    )

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layers):
                node_representation += h_list[layer]

        return node_representation

class GNN(nn.Module):
    def __init__(
        self,
        num_tasks=1,
        num_layers=5,
        emb_dim=300,
        gnn_type="gin",
        virtual_node=True,
        residual=False,
        drop_ratio=0,
        JK="last",
        graph_pooling="sum",
    ):
        """
        num_tasks (int): number of labels to be predicted
        virtual_node (bool): whether to add virtual node or not
        """
        super(GNN, self).__init__()

        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(
                num_layers,
                emb_dim,
                JK=JK,
                drop_ratio=drop_ratio,
                residual=residual,
                gnn_type=gnn_type,
            )
        else:
            self.gnn_node = GNN_node(
                num_layers,
                emb_dim,
                JK=JK,
                drop_ratio=drop_ratio,
                residual=residual,
                gnn_type=gnn_type,
            )

        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = SumPooling()
        elif self.graph_pooling == "mean":
            self.pool = AvgPooling()
        elif self.graph_pooling == "max":
            self.pool = MaxPooling
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttentionPooling(
                gate_nn=nn.Sequential(
                    nn.Linear(emb_dim, 2 * emb_dim),
                    nn.BatchNorm1d(2 * emb_dim),
                    nn.ReLU(),
                    nn.Linear(2 * emb_dim, 1),
                )
            )

        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, n_iters=2, n_layers=2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = nn.Linear(2 * self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = nn.Linear(self.emb_dim, self.num_tasks)

    def forward(self, g, x, edge_attr):
        h_node = self.gnn_node(g, x, edge_attr)

        h_graph = self.pool(g, h_node)
        output = self.graph_pred_linear(h_graph)

        if self.training:
            return output
        else:
            return torch.clamp(output, min=0, max=50)