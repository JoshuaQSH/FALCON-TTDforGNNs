import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt

from functools import namedtuple
from sklearn.metrics import accuracy_score, f1_score
import community as community_louvain
from networkx.readwrite import json_graph

from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

import dgl
import dgl.function as fn
from dgl.dataloading import NeighborSampler
from dgl.distributed import DistGraph, DistDataLoader, node_split
from dgl.data import PPIDataset

import os
import json

class TrainDataset(Dataset):
    def __init__(self, triples, nentity, nrelation, negative_sample_size, mode, count, true_head, true_tail):
        self.len = len(triples['head'])
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = count
        self.true_head = true_head
        self.true_tail = true_tail
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        head, relation, tail = self.triples['head'][idx], self.triples['relation'][idx], self.triples['tail'][idx]
        positive_sample = [head, relation, tail]

        subsampling_weight = self.count[(head, relation)] + self.count[(tail, - relation - 1)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        negative_sample = torch.randint(0, self.nentity, (self.negative_sample_size,))
        positive_sample = torch.LongTensor(positive_sample)
            
        return positive_sample, negative_sample, subsampling_weight, self.mode
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, subsample_weight, mode
    
class TestDataset(Dataset):
    def __init__(self, triples, args, mode, random_sampling):
        self.len = len(triples['head'])
        self.triples = triples
        self.nentity = args.nentity
        self.nrelation = args.nrelation
        self.mode = mode
        self.random_sampling = random_sampling
        if random_sampling:
            self.neg_size = args.neg_size_eval_train

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        head, relation, tail = self.triples['head'][idx], self.triples['relation'][idx], self.triples['tail'][idx]
        positive_sample = torch.LongTensor((head, relation, tail))

        if self.mode == 'head-batch':
            if not self.random_sampling:
                negative_sample = torch.cat([torch.LongTensor([head]), torch.from_numpy(self.triples['head_neg'][idx])])
            else:
                negative_sample = torch.cat([torch.LongTensor([head]), torch.randint(0, self.nentity, size=(self.neg_size,))])
        elif self.mode == 'tail-batch':
            if not self.random_sampling:
                negative_sample = torch.cat([torch.LongTensor([tail]), torch.from_numpy(self.triples['tail_neg'][idx])])
            else:
                negative_sample = torch.cat([torch.LongTensor([tail]), torch.randint(0, self.nentity, size=(self.neg_size,))])

        return positive_sample, negative_sample, self.mode
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        mode = data[0][2]

        return positive_sample, negative_sample, mode
    
class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0
        
    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data
    
    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data

# def get_evaluator(name):
#     if name in ["cora"]:
#         evaluator = ACCEvaluator()
#     elif name in ["yelp", "ppi", "ppi_large", "reddit", "flickr"]:
#         evaluator = F1Evaluator(average="micro")
#     else:
#         evaluator = get_ogb_evaluator(name)
#     return evaluator

# def load_dataset(device, args):
#     """
#     Load dataset and move graph and features to device
#     """
#     if args.dataset in ["reddit", "cora", "ppi", "ppi_large", "yelp", "flickr"]:
#         # raise RuntimeError("Dataset {} is not supported".format(name))
#         if args.dataset == "reddit":
#             from dgl.data import RedditDataset
#             data = RedditDataset(self_loop=True)
#             g = data[0]
#             g = dgl.add_self_loop(g)
#             n_classes = data.num_classes
#         elif args.dataset == "cora":
#             from dgl.data import CitationGraphDataset
#             data = CitationGraphDataset('cora', raw_dir=os.path.join(args.data_dir, 'cora'))
#             g = data[0]
#             g = dgl.remove_self_loop(g)
#             g = dgl.add_self_loop(g)
#             n_classes = data.num_classes
#         elif args.dataset == "ppi":
#             data = load_ppi_data(args.data_dir)
#             g = data.g
#             n_classes = data.num_classes
#         elif args.dataset == "ppi_large":
#             data = load_ppi_large_data()
#             g = data.g
#             n_classes = data.num_classes
#         elif args.dataset == "yelp":
#             from torch_geometric.datasets import Yelp
#             pyg_data = Yelp(os.path.join(args.data_dir, 'yelp'))[0]
#             feat = pyg_data.x
#             labels = pyg_data.y
#             u, v = pyg_data.edge_index
#             g = dgl.graph((u, v))
#             g.ndata['feat'] = feat
#             g.ndata['label'] = labels
#             g.ndata['train_mask'] = pyg_data.train_mask
#             g.ndata['val_mask'] = pyg_data.val_mask
#             g.ndata['test_mask'] = pyg_data.test_mask
#             n_classes = labels.size(1)
#         elif args.dataset == "flickr":
#             from torch_geometric.datasets import Flickr
#             pyg_data = Flickr(os.path.join(args.data_dir, "flickr"))[0]
#             feat = pyg_data.x
#             labels = pyg_data.y
#             # labels = torch.argmax(labels, dim=1)
#             u, v = pyg_data.edge_index
#             g = dgl.graph((u, v))
#             g.ndata['feat'] = feat
#             g.ndata['label'] = labels
#             g.ndata['train_mask'] = pyg_data.train_mask
#             g.ndata['val_mask'] = pyg_data.val_mask
#             g.ndata['test_mask'] = pyg_data.test_mask
#             n_classes = labels.max().item() + 1
        
#         train_mask = g.ndata['train_mask']
#         val_mask = g.ndata['val_mask']
#         test_mask = g.ndata['test_mask']
#         train_nid = train_mask.nonzero().squeeze().long()
#         val_nid = val_mask.nonzero().squeeze().long()
#         test_nid = test_mask.nonzero().squeeze().long()
#         g = g.to(device)
#         labels = g.ndata['label']

#     else:
#         dataset = DglNodePropPredDataset(name=args.dataset, root=args.data_dir)
#         splitted_idx = dataset.get_idx_split()
#         train_nid = splitted_idx["train"]
#         val_nid = splitted_idx["valid"]
#         test_nid = splitted_idx["test"]
#         g, labels = dataset[0]
#         n_classes = dataset.num_classes
#         g = g.to(device)

#         if args.dataset == "ogbn-arxiv":
#             g = dgl.add_reverse_edges(g, copy_ndata=True)
#             g = dgl.add_self_loop(g)
#             g.ndata['feat'] = g.ndata['feat'].float()

#         elif args.dataset == "ogbn-papers100M":
#             g = dgl.add_reverse_edges(g, copy_ndata=True)
#             g.ndata['feat'] = g.ndata['feat'].float()
#             labels = labels.long()

#         elif args.dataset == "ogbn-mag":
#             # MAG is a heterogeneous graph. The task is to make prediction for
#             # paper nodes
#             path = os.path.join(args.emb_path, f"{args.pretrain_model}_mag")
#             labels = labels["paper"]
#             train_nid = train_nid["paper"]
#             val_nid = val_nid["paper"]
#             test_nid = test_nid["paper"]
#             features = g.nodes['paper'].data['feat']
#             author_emb = torch.load(os.path.join(path, "author.pt"), map_location=torch.device("cpu")).float()
#             topic_emb = torch.load(os.path.join(path, "field_of_study.pt"), map_location=torch.device("cpu")).float()
#             institution_emb = torch.load(os.path.join(path, "institution.pt"), map_location=torch.device("cpu")).float()

#             g.nodes["author"].data["feat"] = author_emb.to(device)
#             g.nodes["institution"].data["feat"] = institution_emb.to(device)
#             g.nodes["field_of_study"].data["feat"] = topic_emb.to(device)
#             g.nodes["paper"].data["feat"] = features.to(device)
#             paper_dim = g.nodes["paper"].data["feat"].shape[1]
#             author_dim = g.nodes["author"].data["feat"].shape[1]
#             if paper_dim != author_dim:
#                 paper_feat = g.nodes["paper"].data.pop("feat")
#                 rand_weight = torch.Tensor(paper_dim, author_dim).uniform_(-0.5, 0.5)
#                 g.nodes["paper"].data["feat"] = torch.matmul(paper_feat, rand_weight.to(device))
#                 print(f"Randomly project paper feature from dimension {paper_dim} to {author_dim}")

#             labels = labels.to(device).squeeze()
#             n_classes = int(labels.max() - labels.min()) + 1
        
#         else:
#             g.ndata['feat'] = g.ndata['feat'].float()

#         labels = labels.squeeze()

#     evaluator = get_evaluator(args.dataset)

#     print(f"# Nodes: {g.number_of_nodes()}\n"
#           f"# Edges: {g.number_of_edges()}\n"
#           f"# Train: {len(train_nid)}\n"
#           f"# Val: {len(val_nid)}\n"
#           f"# Test: {len(test_nid)}\n"
#           f"# Classes: {n_classes}")

#     return g, labels, n_classes, train_nid, val_nid, test_nid, evaluator

def dgl_graph_loader(target_dataset, root, device, args):
    # load data
    data = DglNodePropPredDataset(name=target_dataset, root=root)
    # Split the dataset into train, validation and test set
    splitted_idx = data.get_idx_split()
    train_idx, val_idx, test_idx = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
    graph, labels = data[0]
    n_classes = data.num_classes
    
    
    if target_dataset != 'ogbn-papers100M':
        print('add reversed edges')
        srcs, dsts = graph.all_edges()
        graph.add_edges(dsts, srcs)
        labels = labels[:, 0]
    
    else:
        print('Dealing with papers100M')
        graph = dgl.add_reverse_edges(graph, copy_ndata=True)
        graph.ndata['feat'] = graph.ndata['feat'].float()
        labels = labels.long()
    nfeat = graph.ndata.pop('feat')
    
    print("train idx shape (DGL): ", train_idx.shape)
    print("nfeat shape (DGL): ", nfeat.shape)
    print("labels shape (DGL): ", labels.shape)
    # print("degree (DGL): ", graph.in_degrees())

    if args.plot:
        thread_sum = 0
        for threshold in torch.nonzero(torch.bincount(graph.in_degrees())[100:]):
            thread_sum += torch.bincount(graph.in_degrees())[100+threshold[0]]
        print("---- The number of nodes with degree > 100: ", thread_sum.item())
        in_degrees = graph.in_degrees().numpy()
        out_degrees = graph.out_degrees().numpy()
        degrees = in_degrees + out_degrees
        unique_degrees, counts = np.unique(degrees, return_counts=True)
        plt.rc('xtick', labelsize=18) 
        plt.rc('ytick', labelsize=18) 
        plt.rcParams.update({'font.size': 22})
        plt.figure(figsize=(12, 6))
        # plt.plot(torch.arange(len(graph.in_degrees())),  graph.in_degrees())
        plt.bar(unique_degrees[:100], counts[:100], width=1.5, color='#6A579C')
        plt.xlabel('Node Degree')
        plt.ylabel('Edge Count')
        # plt.title('Graph Degree Distribution')
        plt.grid(True)
        plt.savefig("./figures/degree_distribution_{}.pdf".format(target_dataset), dpi=1500)

    # # add self-loop - ogbn-arxiv
    # print(f"Total edges before adding self-loop {graph.number_of_edges()}")
    # graph = graph.remove_self_loop().add_self_loop()
    # print(f"Total edges after adding self-loop {graph.number_of_edges()}")
    in_feats = nfeat.shape[1]
    
    # n_classes = (labels.max() + 1).item()

    # Create csr/coo/csc formats before launching sampling processes
    # graph.create_formats_()
    for i in range(int(args.n_runs)):
        if args.partition != 0:
            print("Do the graph partitioning")
            graph, labels, train_idx, val_idx, test_idx = dgl_partition(graph, labels, train_idx, val_idx, test_idx, args.partition)
        else:
            print("no graph partition")
        num_feat = torch.arange(graph.number_of_nodes()).to(device)
        labels = labels.to(device)

        data = train_idx, val_idx, test_idx, in_feats, labels, n_classes, num_feat, graph
    
    # Pack data
    num_feat = torch.arange(graph.number_of_nodes())
    data = train_idx, val_idx, test_idx, in_feats, labels, n_classes, num_feat, graph
    print("DGL Data packed!")
    
    return dgl_unpack_data(data, args)

def dgl_unpack_data(data, args):
    # train_nid (0-196614), val_nid (196615-235937), test_nid (235938-2449028)
    # in_feats (100), labels (2449029), n_classes (47), nfeat (2449029)
    # g is the whole graph with (num_nodes=2449029, num_edges=247436560)
    train_nid, val_nid, test_nid, in_feats, labels, n_classes, nfeat, g = data

    # Create PyTorch DataLoader for constructing blocks
    # fan_out = [5,10,15]
    if args.use_sample:
        sampler = dgl.dataloading.MultiLayerNeighborSampler(
            [int(fanout) for fanout in args.fan_out.split(',')])
    else:
        # sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args.num_layers)
        args.fan_out = 'NoFanOut'
    
    # batchsize is 1024, num_workers is 4
    train_loader = dgl.dataloading.DataLoader(
        g,
        train_nid,
        sampler,
        batch_size=args.batch,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers)
    
    sampler_all = dgl.dataloading.MultiLayerFullNeighborSampler(1)
    full_neighbor_loader = dgl.dataloading.DataLoader(
                g,
                torch.arange(g.num_nodes()),
                sampler_all,
                batch_size=args.batch,
                shuffle=False,
                drop_last=False,
                num_workers=args.num_workers)
    
    print("=== Dataloader (DGL) Done! === ")

    return train_loader, full_neighbor_loader, data

def custom_reordering(graph, is_degree=True):
    if is_degree:
        degrees = graph.in_degrees().numpy()
        # threshold for the degree settings
        degree_threshold = np.percentile(degrees, 80) 
        high_degree_vertex_indices = np.where(degrees >= degree_threshold)[0]
        reordered_indices = np.concatenate((high_degree_vertex_indices,
                                            np.setdiff1d(np.arange(graph.number_of_nodes()),
                                            high_degree_vertex_indices)))
        par_g = dgl.reorder_graph(graph, node_permute_algo='custom', permute_config={'nodes_perm': reordered_indices})
    
    # louvain and mertis
    else:
        node_perm, num_clusters, max_cluster = louvain_and_metis_reorder(graph, merege=None, plot_name=None)
        par_g = dgl.reorder_graph(graph, node_permute_algo='custom', permute_config={'nodes_perm': node_perm})
        
    return par_g

# CPU 
def louvain_and_metis_reorder(g, merge=None, plot_name=None):    
    nx_g = g.to_networkx().to_undirected()
    partition = community_louvain.best_partition(nx_g)
    modularity_score = community_louvain.modularity(partition, nx_g)
    partition_list = list(partition.items())
    df = pd.DataFrame(partition_list, columns=['vertex', 'partition'])
   
    partition = df['partition'].to_numpy()
    vertex = df['vertex'].to_numpy()
    num_clusters = max(partition) + 1


    # print(df)
    print('modularity score', modularity_score)
    print('number of clusters: ', num_clusters)

    nodes_per_cluster = np.zeros(num_clusters)
    for i in range(num_clusters):
        nodes_per_cluster[i] = np.sum(partition == i)

    print(nodes_per_cluster.astype(int))
    print('total nodes = ', np.sum(nodes_per_cluster))

    if plot_name is not None:
        _ = plt.hist(nodes_per_cluster, bins='auto', log=True)
        plt.title("Histogram of size of clusters in "+plot_name)
        plt.savefig(plot_name + ".pdf")

    node_perm = np.zeros(g.num_nodes())

    # don't merge any clusters
    if merge is None:    
        max_cluster = max(nodes_per_cluster)
        for i in range(num_clusters):
            curr_nodes = np.sum(partition == i)
            new_idx = np.arange(max_cluster * i, max_cluster * i + curr_nodes)
            curr_idx = partition == i
            node_perm[vertex[curr_idx]] = new_idx
            print("->", new_idx)

            subg = g.subgraph(nodes=vertex[curr_idx])
            if curr_nodes > 100:
                #  dgl.partition.metis_partition_assignment()
                pids = dgl.partition.metis_partition_assignment(
                    subg if subg.device == dgl.backend.cpu() else subg.to(dgl.backend.cpu()), 
                    int(min(128, curr_nodes/2)))
                pids = dgl.backend.asnumpy(pids)
                start_idx = max_cluster * i
                curr_pid = 0
                metis_vertex = vertex[curr_idx]
                while curr_pid < np.max(pids):
                    metis_idx = pids == curr_pid
                    new_idx = np.arange(start_idx, start_idx + int(np.sum(metis_idx)))
                    node_perm[metis_vertex[metis_idx]] = new_idx
                    curr_pid += 1
                    start_idx += int(np.sum(metis_idx))

        print('max = ', max_cluster)
        print('node perm = ', node_perm.astype(int))
        
    return node_perm.astype(int), num_clusters, max_cluster

def recursive_metis_reorder(graph, current_level, max_level, partition_list):
    """
    partition_list: [125, 140, 140]
    current_level: 1
    max_level: 3
    """
    if current_level > max_level:
        return graph

    # Use the number of partitions specified for the current level
    partitions = partition_list[current_level - 1]
    
    reordered_graph = dgl.reorder_graph(graph, 'metis', permute_config={'k': partitions})
    
    return recursive_metis_reorder(reordered_graph, current_level + 1, max_level, partition_list)

# A function to recursively partition a graph
# num_parts: List [140, 140, 125], num_parts[2], num_parts[1], num_parts[0]
def recursive_partition(graph, level, num_parts):
    if level == 0:
        return {f"level_{level}": graph}
    partitions = dgl.metis_partition(graph, num_parts[level])
    results = {}
    for part_id, subgraph in partitions.items():
        sub_partitions = recursive_partition(subgraph, level - 1, num_parts)
        for key, value in sub_partitions.items():
            results[f"level_{level}_part_{part_id}_{key}"] = value
    return results

def map_indices_to_partition(g, partition):
    # Original indices of nodes in the partition
    original_indices = partition.ndata['_ID']

    # Mapping global masks to partition-specific masks
    partition.ndata['train_mask'] = g.ndata['train_mask'][original_indices]
    partition.ndata['val_mask'] = g.ndata['val_mask'][original_indices]
    partition.ndata['test_mask'] = g.ndata['test_mask'][original_indices]
    partition.ndata['label'] = g.ndata['label'][original_indices]

    return partition

def dgl_partition(graph, labels, train_idx, val_idx, test_idx, partition):
    graph.ndata['label'] = labels
    num_nodes = graph.num_nodes()
    train_mask = np.zeros(num_nodes, dtype=bool)
    train_mask[train_idx] = True
    val_mask = np.zeros(num_nodes, dtype=bool)
    val_mask[val_idx] = True
    test_mask = np.zeros(num_nodes, dtype=bool)
    test_mask[test_idx] = True
    graph.ndata['train_mask'] = torch.tensor(train_mask)
    graph.ndata['val_mask'] = torch.tensor(val_mask)
    graph.ndata['test_mask'] = torch.tensor(test_mask)
    if partition != 0:
        if partition == -1:
            partition_with_subgraph = False
            if partition_with_subgraph:
                result = []
                print("Partition graph with multi-level METIS")
                ### Subgraphs
                par_g = recursive_partition(graph, level=2, num_parts=[2, 2, 2])
                for part_id, part in par_g.items():
                    result.append(map_indices_to_partition(graph, part))
                print("Partitioning done!")

            else:
                # par_g = custom_reordering(graph)
                ### Reorder
                print("Partition graph with multi-level METIS")
                par_g = recursive_metis_reorder(graph, 1, 3, [50, 60, 60])
                print("Partitioning done!")

        elif partition == -2:
            print("Partition graph by rcmk")
            par_g = dgl.reorder_graph(graph, 'rcmk')
            print("Partitioning done!")

        else:
            # print("Randomly permute the node order first")
            # nodes_perm = torch.randperm(graph.num_nodes())
            # par_g = dgl.reorder_graph(graph, 'custom', permute_config={'nodes_perm':nodes_perm})
            print("Partition graph by METIS into {} parts".format(partition))
            par_g = dgl.reorder_graph(graph, 'metis', permute_config={'k':partition})
            print("Partitioning done!")

    else:
        print("Randomly permute the node order")
        nodes_perm = torch.randperm(graph.num_nodes())
        par_g = dgl.reorder_graph(graph, 'custom', permute_config={'nodes_perm':nodes_perm})
    
    graph = par_g
    train_idx = torch.tensor(np.where(graph.ndata['train_mask'] == True))[0, :]
    val_idx = torch.tensor(np.where(graph.ndata['val_mask'] == True))[0, :]
    test_idx = torch.tensor(np.where(graph.ndata['test_mask'] == True))[0, :]
    labels = graph.ndata['label']

    return graph, labels, train_idx, val_idx, test_idx
