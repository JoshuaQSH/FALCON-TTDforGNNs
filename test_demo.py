import scipy.sparse.linalg
import numpy as np
import torch
from tt_utils import get_eigen, get_ortho, tt_matrix_decomp, compression_rate
import dgl

# Mock graph object with necessary methods
class MockGraph:
    def __init__(self, adj_matrix):
        self.adj_matrix = adj_matrix
    
    def adj(self, scipy_fmt='csr'):
        if scipy_fmt == 'csr':
            return self.adj_matrix
        else:
            raise ValueError("Unsupported format")
    
    def in_degrees(self):
        # Return the sum of the adjacency matrix's columns as in-degrees
        return np.array(self.adj_matrix.sum(axis=0)).flatten()

adj_csr = np.array([[0., 1., 0., 0., 1.],
       [1., 0., 1., 0., 0.],
       [0., 1., 0., 1., 0.],
       [0., 0., 1., 0., 1.],
       [1., 0., 0., 1., 0.]], dtype=np.float32)
matrix = torch.ones(2, 12)
"""
TT_ranks: the number of dimensions of the tensor plus one

"""
tt_ranks =  [1, 2, 1] 
tt_p_shapes = [1, 2, 3]
tt_q_shapes = [4, 1]

# Initialize the mock graph with the adjacency matrix
g = MockGraph(adj_csr)

# Parameters for `get_eigen`
k = 3  # Number of eigenvalues/vectors to compute
name = "test_graph"
mode = "adj"  # or "laplacian"

# Execute the function
eigen_vals, eigen_vecs = get_eigen(g, k, name, mode)
print("Eigenvalues:", eigen_vals)

tt_rank = 8
p_shapes = [125, 140, 140]
tt_cores = get_ortho([1, tt_rank, tt_rank, 1], p_shapes, [4, 5, 5])
print(tt_cores[0].shape, tt_cores[1].shape, tt_cores[2].shape)

# Graph partitioning
g = dgl.graph((torch.tensor([0, 1, 2, 3, 4]), torch.tensor([2, 2, 3, 2, 3])))
g.ndata['h'] = torch.arange(g.num_nodes() * 2).view(g.num_nodes(), 2)
g.edata['w'] = torch.arange(g.num_edges() * 1).view(g.num_edges(), 1)
print("Before partitioning: (ndata)", g.ndata)
print("Before partitioning: (edata)", g.edata)

rg = dgl.reorder_graph(g, node_permute_algo='rcmk')
print("After partitioning: (rcmk)", rg.ndata)
print("After partitioning: (rcmk)", rg.edata)

rg = dgl.reorder_graph(g, node_permute_algo='metis', permute_config={'k':2})
print("After partitioning: (metis)", rg.ndata)
print("After partitioning: (metis)", rg.edata)

# cores, ranks = tt_matrix_decomp(matrix, tt_ranks, tt_p_shapes, tt_q_shapes)
# print("Decomposed TT cores shapes:", [core.shape for core in cores])
# print("Ranks after decomposition:", ranks)

# compression_rate_output = compression_rate(matrix.shape, ranks, cores)
# print(compression_rate_output)
