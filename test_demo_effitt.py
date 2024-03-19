import torch.nn as nn
import torch

import sys
sys.path.insert(0, '/home/shenghao/FBTT-Embedding')
from tt_embeddings_ops import TTEmbeddingBag


from Efficient_TT.efficient_tt import Eff_TTEmbedding

device = torch.device("cuda:0")

embedding_sum = nn.EmbeddingBag(16, 4, mode='sum').to(device)
input = torch.tensor([1, 2, 4, 5], dtype=torch.long).to(device)
offsets = torch.tensor([0, 2], dtype=torch.long).to(device)
print(embedding_sum(input, offsets))

embedding_tt = TTEmbeddingBag(16, 4, tt_ranks=[2, 2, 2]).to(device)
print(embedding_tt(input, offsets))

effi_tt = Eff_TTEmbedding(1600000, 400, tt_ranks=[2, 2]).to(device)
print(effi_tt(input, offsets))
