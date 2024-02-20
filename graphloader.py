import torch as th

from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.loader import NeighborLoader, DataLoader
import torch_geometric.transforms as T
from torch_geometric.utils import add_self_loops, to_undirected

from ogb.nodeproppred import DglNodePropPredDataset
import dgl
import numpy as np

import os

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
    graph.add_edges(dsts, srcs)
    nfeat = graph.ndata.pop('feat')

    labels = labels[:, 0]
    print("train idx shape (DGL): ", train_idx.shape)
    print("nfeat shape (DGL): ", nfeat.shape)
    print("labels shape (DGL): ", labels.shape)

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

    if partition > 0:
        print("Randomly permute the node order")
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
