import dgl
import dgl.nn.pytorch as dglnn
from dgl.data import AsNodePredDataset

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist

import os
import time

from tt_utils import *
from utils import Logger
from gnn_model import SAGE
from graphloader import dgl_graph_loader

import nevergrad as ng

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'

def compute_acc(pred, labels):
    return (torch.argmax(pred, dim=1) == labels).float().sum() / len(pred)

def evaluate(model, g, nfeat, labels, val_nid, test_nid, device, full_neighbor_loader):
    model.eval()
    with torch.no_grad():
        pred = model.inference(g, device, full_neighbor_loader)
    model.train()
    labels = labels.to(device)
    return compute_acc(pred[val_nid], labels[val_nid]), compute_acc(pred[test_nid], labels[test_nid]), pred

def load_subtensor(nfeat, labels, seeds, input_nodes):
    """
    Extracts features and labels for a set of nodes.
    """
    batch_inputs = nfeat[input_nodes]
    batch_labels = labels[seeds]
    return batch_inputs, batch_labels

def create_3d_weight_parameters(p_shape, q_shape, tt_rank, distribution='normal'):
    shape_1 = (1, p_shape[0], tt_rank[0] * q_shape[0])
    shape_2 = (1, p_shape[1], tt_rank[0] * tt_rank[1] * q_shape[1])
    shape_3 = (1, p_shape[2], tt_rank[1] * q_shape[2])

    # Define the 3D array parameter
    weights_1 = ng.p.Array(shape=shape_1)
    weights_2 = ng.p.Array(shape=shape_2)
    weights_3 = ng.p.Array(shape=shape_3)
    
    # Set the initialization distribution
    if distribution == 'normal':
        weights_1.set_mutation(sigma=1.0)  # Standard deviation for normal distribution
        weights_2.set_mutation(sigma=1.0)  # Standard deviation for normal distribution
        weights_3.set_mutation(sigma=1.0)  # Standard deviation for normal distribution
    elif distribution == 'uniform':
        weights_1.set_bounds(lower=0.0, upper=1.0)  # Bounds for uniform distribution
        weights_2.set_bounds(lower=0.0, upper=1.0)  # Bounds for uniform distribution
        weights_3.set_bounds(lower=0.0, upper=1.0)  # Bounds for uniform distribution

    return weights_1, weights_2, weights_3

def train(train_loader, model, loss_fcn, optimizer, lr_scheduler, nfeat, labels, device, args, iter_tput, log=None):
    # Loop over the dataloader to sample the computation dependency graph as a list of blocks.
    ave_forward_throughput=[]
    ave_backward_throughput=[]
    
    # Enable dataloader cpu affinitization for cpu devices (no effect on gpu)
    # with train_loader.enable_cpu_affinity():
    for step, (input_nodes, seeds, blocks) in enumerate(train_loader):
        tic_step = time.time()
        
        # copy block to gpus
        blocks = [blk.int().to(device) for blk in blocks]
        # Load the input features as well as output labels
        batch_inputs, batch_labels = load_subtensor(nfeat, labels, seeds, input_nodes)
        # batch_inputs = batch_inputs.to(device)
        batch_labels = batch_labels.to(device)

        # Compute loss and prediction
        # model.embed_layer.cache_populate()
        batch_pred = model(blocks, batch_inputs)
        
        # One of the optimization steps
        loss = loss_fcn(batch_pred, batch_labels)
        
        # Forward throughput
        fwd_throughput= len(seeds)/(time.time()-tic_step)
        ave_forward_throughput.append(fwd_throughput)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Backward throughput
        bwd_throughput= len(seeds)/(time.time()-tic_step)
        ave_backward_throughput.append(bwd_throughput)
        lr_scheduler.step(loss)
        iter_tput.append(len(seeds) / (time.time() - tic_step))

        if step % args.log_every == 0:
            acc = compute_acc(batch_pred, batch_labels)
            gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
            if args.logging:
                log.logger.debug('Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'.format(
                    step, loss.item(), acc.item(), np.mean(iter_tput[3:]), gpu_mem_alloc))
            else:
                print('Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'.format(
                    step, loss.item(), acc.item(), np.mean(iter_tput[3:]), gpu_mem_alloc))
    
    return loss, iter_tput

def give_record(per_epoch_time, total_time, iter_tput, args, tt_rank, log=None):
    if args.logging:
        log.logger.info('Per Epoch Time(s): {:.4f}'.format(per_epoch_time))
        log.logger.info('Running Time(s): {:.4f}'.format(total_time))
        log.logger.info('Avg overall throughput is {:.4f}'.format(np.mean(iter_tput[:])))
        log.logger.info('TT Ranks: {}'.format(tt_rank))
        
    else:
        print('Per Epoch Time(s): {:.4f}'.format(per_epoch_time))
        print('Running Time(s): {:.4f}'.format(total_time))
        print('Avg overall throughput is {:.4f}'.format(np.mean(iter_tput[:])))
        print('TT Ranks: {}'.format(tt_rank))

# weights_1, weights_2, weights_3, tt_rank, data, args
def model_tuner(tt_rank, data, args):
    train_nid, val_nid, test_nid, in_feats, labels, n_classes, nfeat, g = data
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
        tt_rank = [tt_rank[0],tt_rank[0]],
        p_shapes = [int(i) for i in args.p_shapes.split(',')],
        q_shapes = [int(i) for i in args.q_shapes.split(',')],
        init = args.init, 
        graph = g, 
        device = args.device,
        embed_name = args.emb_name,
        access_counts=args.access_counts,
        use_cached=args.use_cached,
        cache_size = args.cache_size,
        sparse = args.sparse,
        batch_count = args.batch_count,
        # weights_init = (weights_1, weights_2, weights_3),
    )
    model = model.to(device)
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.8,
                                                                  patience=800, verbose=True)
    # Training loop
    iter_tput = []
    total_time = 0.0

    tic = time.time()
    ## Turn this into a nevergrad tunning step
    loss, iter_tput = train(train_loader, 
                model, 
                loss_fcn, 
                optimizer, 
                lr_scheduler, 
                nfeat, 
                labels, 
                device, 
                args, 
                iter_tput, 
                log=None)
    toc = time.time()
    per_epoch_time = toc - tic
    total_time += (toc - tic)
    avg_througput = np.mean(iter_tput[5:]).item()
    
    # give_record(per_epoch_time, total_time, iter_tput, args, args.tt_rank, log)
    
    # Nevergrad could minimize the [loss, time, throughput]
    return 1 / avg_througput

def run_single(train_loader, full_neighbor_loader, data, args):
    # Logging
    start_time = int(round(time.time()*1000))
    timestamp = time.strftime('%Y%m%d-%H%M%S',time.localtime(start_time/1000))
    # Setup the saved log file, with time and filename
    saved_log_path = './logs/'
    saved_log_name = saved_log_path + 'FinalScale-{}-{}-4090-{}-r3-{}.log'.format(args.model, args.dataset, args.batch, timestamp)

    if args.logging:
        log = Logger(saved_log_name, level='debug')
        log.logger.debug("[Running GraphSAGE Model == Hidden: {}, Layers: {} ==]".format(args.num_hidden, args.num_layers))
        log.logger.debug("[Dataset: {}]".format(args.dataset))
    else:
        log = None

    # Unpack data
    #train_nid, val_nid, test_nid, in_feats, labels, n_classes, nfeat, g = data

    # Define model and optimizer
    tt_rank = ng.ops.Int(deterministic=True)(ng.p.Array(shape=(1,), lower=2, upper=256))
    # tt_rank = [int(i) for i in args.tt_rank.split(',')]
    p_shapes = [int(i) for i in args.p_shapes.split(',')]
    q_shapes = [int(i) for i in args.q_shapes.split(',')]
    # weights_1, weights_2, weights_3 = create_3d_weight_parameters(p_shapes, q_shapes, tt_rank, distribution='normal')
    # instru = ng.p.Instrumentation(weights_1, weights_2, weights_3)
    instru = ng.p.Instrumentation(tt_rank)
    budget = 20
    # names = ["RandomSearch", "TwoPointsDE", "CMA", "PSO", "ScrHammersleySearch"]
    # names = ["RandomSearch", "TwoPointsDE"]
    names = ["CMA", "PSO", "ScrHammersleySearch"]
    for name in names:
        optim = ng.optimizers.registry[name](parametrization=instru, budget=budget, num_workers=1)
        for u in range(budget):
            x1 = optim.ask()
            # x2 = optim.ask()
            # x3 = optim.ask()
            y1 = model_tuner(*x1.args, data, args)
            # y2 = model_tuner(*x2.args, tt_rank, data, args)
            # y3 = model_tuner(*x3.args, tt_rank, data, args)
            optim.tell(x1, y1)
            # optim.tell(x2, y2)
            # optim.tell(x3, y3)
    recommendation = optim.recommend()
    print("* ", name, " provides a vector of parameters with test error ",
          model_tuner(*recommendation.args, data, args))
    
    return recommendation
        

if __name__ == '__main__':
    args = parse_args()
    
    if args.device != 'cpu' and torch.cuda.is_available():
        device = torch.device(args.device)
        devices = list(map(int, args.num_gpus.split(',')))
        nprocs = len(devices)
        print('Using {} GPUs'.format(nprocs))
    else:
        device = torch.device('cpu')
        print("Training with CPU")

    # load ogbn-products data - dgl version
    # /home/shenghao/gnn_related/dataset
    target_dataset = args.dataset
    if args.data_dir == None:
        root = os.path.join(os.environ['HOME'], args.workspace, 'gnn_related', 'dataset')
    else:
        root = args.data_dir

    # Local single GPU training
    train_loader, full_neighbor_loader, data = dgl_graph_loader(target_dataset, root, device, args)
    print('Begin with Single GPU. Init from {}'.format(args.init))
    
    start_time = int(round(time.time()*1000))
    recommendation = run_single(train_loader, full_neighbor_loader, data, args)
    # print('Loss {:.4f}'.format(loss.item()))
    end_time = int(round(time.time()*1000))
    print('Recommendation: ', recommendation)
    print('Tunning Time: ', end_time - start_time)
