import torch as th

from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.loader import NeighborLoader, DataLoader
import torch_geometric.transforms as T
from torch_geometric.utils import add_self_loops, to_undirected

from ogb.nodeproppred import DglNodePropPredDataset
import dgl
from dgl.dataloading import NeighborSampler
from dgl.distributed import DistGraph, DistDataLoader, node_split
import numpy as np
import matplotlib.pyplot as plt
import community as community_louvain
import pandas as pd

import os

def dist_graph_loader(target_dataset, root, device, args):
    print("Prepare distributed graph data loader for {} dataset".format(target_dataset))
    
    # load data
    graph, label = load_ogb(target_dataset, root)

    # initialize distributed contexts
    # dgl.distributed.initialize('ip_config.txt')
    # th.distributed.init_process_group(backend='gloo')

    # Partition a graph for distributed training and store the partitions on files.
    dgl.distributed.partition_graph(graph, 'dict_graph', 2, 
        num_hops=3, 
        part_method='metis',
        out_path='output/', 
        balance_edges=True)
    
    # load distributed graph
    g = DistGraph('dict_graph', './output/dict_graph.json')
    pb = g.get_partition_book()
    train_nid = dgl.distributed.node_split(g.ndata['train_mask'], pb, force_even=True)
    val_nid = dgl.distributed.node_split(g.ndata['val_mask'], pb, force_even=True)
    test_nid = dgl.distributed.node_split(g.ndata['test_mask'], pb, force_even=True)
    # Create sampler
    sampler = NeighborSampler(g, [int(fanout) for fanout in args.fan_out.split(',')],
                          dgl.distributed.sample_neighbors,
                          device)
    dataloader = DistDataLoader(
        dataset=train_nid.numpy(),
        batch_size=args.batch,
        collate_fn=sampler.sample_blocks,
        shuffle=True,
        drop_last=False)
    
    return dataloader, g, label



def load_reddit(self_loop=True):
    from dgl.data import RedditDataset

    # load reddit data
    data = RedditDataset(self_loop=self_loop)
    g = data[0]
    g.ndata['features'] = g.ndata.pop('feat')
    g.ndata['labels'] = g.ndata.pop('label')
    return g, data.num_classes

def load_ogb(name, root='dataset'):
    
    print('load', name)
    data = DglNodePropPredDataset(name=name, root=root)
    print('finish loading', name)
    splitted_idx = data.get_idx_split()
    graph, labels = data[0]
    labels = labels[:, 0]

    graph.ndata['features'] = graph.ndata.pop('feat')
    graph.ndata['labels'] = labels
    in_feats = graph.ndata['features'].shape[1]
    num_labels = len(th.unique(labels[th.logical_not(th.isnan(labels))]))

    # Find the node IDs in the training, validation, and test set.
    train_nid, val_nid, test_nid = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
    train_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    train_mask[train_nid] = True
    val_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    val_mask[val_nid] = True
    test_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    test_mask[test_nid] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    print('finish constructing', name)
    return graph, num_labels

def inductive_split(g):
    """Split the graph into training graph, validation graph, and test graph by training
    and validation masks.  Suitable for inductive models."""
    train_g = g.subgraph(g.ndata['train_mask'])
    val_g = g.subgraph(g.ndata['train_mask'] | g.ndata['val_mask'])
    test_g = g
    return train_g, val_g, test_g

def dgl_graph_loader(target_dataset, root, device, args):
    # load data
    data = DglNodePropPredDataset(name=target_dataset, root=root)
    # Split the dataset into train, validation and test set
    splitted_idx = data.get_idx_split()
    train_idx, val_idx, test_idx = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
    graph, labels = data[0]

    # add reverse edges
    # print('add reversed edges')
    srcs, dsts = graph.all_edges()
    
    if target_dataset != 'ogbn-paper100M':
        graph.add_edges(dsts, srcs)
    # graph.add_edges(dsts, srcs)

    nfeat = graph.ndata.pop('feat')

    
    labels = labels[:, 0]
    print("train idx shape (DGL): ", train_idx.shape)
    print("nfeat shape (DGL): ", nfeat.shape)
    print("labels shape (DGL): ", labels.shape)
    print("degree (DGL): ", graph.in_degrees())

    if args.plot:
        plt.rc('xtick', labelsize=20) 
        plt.rc('ytick', labelsize=20) 
        plt.rcParams.update({'font.size': 22})
        plt.figure(figsize=(12, 6))
        plt.plot(th.arange(len(graph.in_degrees())),  graph.in_degrees())
        plt.xlabel('Node Index')
        plt.ylabel('# degree')
        plt.title('Graph Degree Distribution')
        plt.grid(True)
        plt.savefig("./figures/degree_distribution_{}.pdf".format(target_dataset), dpi=1500)

    # # add self-loop - ogbn-arxiv
    # print(f"Total edges before adding self-loop {graph.number_of_edges()}")
    # graph = graph.remove_self_loop().add_self_loop()
    # print(f"Total edges after adding self-loop {graph.number_of_edges()}")

    in_feats = nfeat.shape[1]
    n_classes = (labels.max() + 1).item()

    # Create csr/coo/csc formats before launching sampling processes
    graph.create_formats_()
    for i in range(int(args.n_runs)):
        if args.partition != 0:
            print("Do the graph partitioning")
            graph, labels, train_idx, val_idx, test_idx = dgl_partition(graph, labels, train_idx, val_idx, test_idx, args.partition)
        else:
            print("no graph partition")
        num_feat = th.arange(graph.number_of_nodes()).to(device)
        labels = labels.to(device)

        data = train_idx, val_idx, test_idx, in_feats, labels, n_classes, num_feat, graph
    
    # Pack data
    num_feat = th.arange(graph.number_of_nodes())
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
                th.arange(g.num_nodes()),
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
    graph.ndata['train_mask'] = th.tensor(train_mask)
    graph.ndata['val_mask'] = th.tensor(val_mask)
    graph.ndata['test_mask'] = th.tensor(test_mask)

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
            else:
                # par_g = custom_reordering(graph)
                ### Reorder
                print("Partition graph with multi-level METIS")
                par_g = recursive_metis_reorder(graph, 1, 3, [125, 140, 140])
        elif partition == -2:
            print("Partition graph by rcmk")
            par_g = dgl.reorder_graph(graph, 'rcmk')

        else:
            print("Randomly permute the node order first")
            nodes_perm = th.randperm(graph.num_nodes())
            par_g = dgl.reorder_graph(graph, 'custom', permute_config={'nodes_perm':nodes_perm})
            print("Partition graph by METIS into {} parts".format(partition))
            par_g = dgl.reorder_graph(graph, 'metis', permute_config={'k':partition})

    else:
        print("Randomly permute the node order")
        nodes_perm = th.randperm(graph.num_nodes())
        par_g = dgl.reorder_graph(graph, 'custom', permute_config={'nodes_perm':nodes_perm})
    
    graph = par_g
    train_idx = th.tensor(np.where(graph.ndata['train_mask'] == True))[0, :]
    val_idx = th.tensor(np.where(graph.ndata['val_mask'] == True))[0, :]
    test_idx = th.tensor(np.where(graph.ndata['test_mask'] == True))[0, :]
    labels = graph.ndata['label']

    return graph, labels, train_idx, val_idx, test_idx

# load ogbn-xxx graph data - pyg version
def graph_loader(target_dataset, root, args):
    
    if target_dataset == 'ogbn-products':
        dataset = PygNodePropPredDataset(name=target_dataset, root=root, transform=T.ToUndirected())
        # Get the first graph object, which in this case is the entire dataset
        data = dataset[0]

        # Add self-loops to the edges and convert to undirected
        data.edge_index = add_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]
        data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)

        # Data preparation
        nfeat = data.x
        labels = data.y.squeeze()

        # Extracting the number of features and classes
        in_feats = nfeat.size(1)
        n_classes = int(labels.max()) + 1

        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
        if args.use_sample:
            train_loader = NeighborLoader(data, input_nodes=train_idx,
                                        shuffle=True, num_workers=os.cpu_count() - 2,
                                        batch_size=args.batch, num_neighbors=[args.sample] * args.neighbors)

        else:
            # Batch size of 1 because we have only one graph
            train_loader = DataLoader([data], batch_size=args.batch, shuffle=True)
        
        total_loader = NeighborLoader(data, input_nodes=None, num_neighbors=[-1],
                                batch_size=args.batch, shuffle=False,
                                num_workers=os.cpu_count() - 2)

    elif target_dataset == 'ogbn-arxiv':
        dataset = PygNodePropPredDataset(name=target_dataset, root=root)
        # Get the first graph object, which in this case is the entire dataset
        data = dataset[0]
        split_idx = dataset.get_idx_split() 
                
        train_idx = split_idx['train']
        valid_idx = split_idx['valid']
        test_idx = split_idx['test']
        
        if args.use_sample:
            train_loader = NeighborLoader(data, input_nodes=train_idx,
                                        shuffle=True, num_workers=os.cpu_count() - 2,
                                        batch_size=args.batch, num_neighbors=[args.sample] * args.neighbors)

        else:
            # Batch size of 1 because we have only one graph
            train_loader = DataLoader([data], batch_size=args.batch, shuffle=True)
        
        total_loader = NeighborLoader(data, input_nodes=None, num_neighbors=[-1],
                                batch_size=args.batch, shuffle=False,
                                num_workers=os.cpu_count() - 2)
        
    return dataset, data, train_loader, total_loader, train_idx, valid_idx, test_idx, split_idx
