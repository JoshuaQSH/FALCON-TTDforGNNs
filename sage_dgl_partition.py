import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl.nn.pytorch as dglnn
import time
import argparse
import tqdm
from ogb.nodeproppred import DglNodePropPredDataset

import os
from tt_utils import *
from utils import Logger, gpu_timing, memory_usage

from dgl_sage import SAGE
from graphloader import dgl_graph_loader



def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)

def evaluate(model, g, nfeat, labels, val_nid, test_nid, device, full_neighbor_loader):
    """
    Evaluate the model on the validation set specified by ``val_mask``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_mask : A 0-1 mask indicating which nodes do we actually compute the accuracy for.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
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

def train(train_loader, model, loss_fcn, optimizer, lr_scheduler, nfeat, labels, device, epoch, args, iter_tput, log=None):
    # Loop over the dataloader to sample the computation dependency graph as a list of blocks.
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
        batch_pred = model(blocks, batch_inputs)
        loss = loss_fcn(batch_pred, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step(loss)
        iter_tput.append(len(seeds) / (time.time() - tic_step))
        if step % args.log_every == 0:
            acc = compute_acc(batch_pred, batch_labels)
            gpu_mem_alloc = th.cuda.max_memory_allocated() / 1000000 if th.cuda.is_available() else 0
            if args.logging:
                log.logger.debug('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'.format(
                    epoch, step, loss.item(), acc.item(), np.mean(iter_tput[3:]), gpu_mem_alloc))
            else:
                print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'.format(
                    epoch, step, loss.item(), acc.item(), np.mean(iter_tput[3:]), gpu_mem_alloc))

def run(train_loader, full_neighbor_loader, data, args):
    # Logging
    start_time = int(round(time.time()*1000))
    timestamp = time.strftime('%Y%m%d-%H%M%S',time.localtime(start_time/1000))
    
    # Setup the saved log file, with time and filename
    saved_log_path = './logs/'
    if args.use_sample:
        is_sample = 'sample'
    else:
        is_sample = 'full'
    saved_log_name = saved_log_path + '{}-{}-{}-{}-batch-{}-{}.log'.format(args.model, args.device, is_sample, args.fan_out, args.batch, timestamp)
    if args.logging:
        log = Logger(saved_log_name, level='debug')
        log.logger.debug("[Running GraphSAGE Model == Hidden: {}, Layers: {} ==]".format(args.num_hidden, args.num_layers))
        log.logger.debug("[Dataset: {}]".format(args.dataset))
    else:
        log = None

    # Unpack data
    train_nid, val_nid, test_nid, in_feats, labels, n_classes, nfeat, g = data

    # Define model and optimizer
    device = th.device(args.device)
    model = SAGE(
        num_nodes = g.number_of_nodes(), 
        in_feats = in_feats, 
        n_hidden = args.num_hidden, 
        n_classes = n_classes, 
        n_layers = args.num_layers, 
        activation = F.relu, 
        dropout = args.dropout, 
        use_tt = args.use_tt, 
        tt_rank = args.tt_rank, 
        dist = args.init, 
        graph = g, 
        device = args.device,
        embed_name = args.emb_name)
    model = model.to(device)
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    lr_scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.8,
                                                                  patience=800, verbose=True)
    
    # Training loop
    avg = 0
    iter_tput = []
    best_eval_acc = 0.0
    best_test_acc = 0.0
    total_time = 0.0
    for epoch in range(args.epochs):
        tic = time.time()
        train(train_loader, model, loss_fcn, optimizer, lr_scheduler, nfeat, labels, device, epoch, args, iter_tput, log=log)
        toc = time.time()
        total_time += (toc - tic)
        if args.logging:
            log.logger.info('Epoch Time(s): {:.4f}'.format(toc - tic))
        else:
            print('Epoch Time(s): {:.4f}'.format(toc - tic))
        
        if epoch >= 5:
            avg += toc - tic
        if epoch % args.eval_every == 0 and epoch != 0:
            eval_acc, test_acc, pred = evaluate(model, g, nfeat, labels, val_nid, test_nid, device, full_neighbor_loader)
            if args.save_pred:
                np.savetxt(args.save_pred + '%02d' % epoch, pred.argmax(1).cpu().numpy(), '%d')

            if args.logging:
                log.logger.info('Eval Acc {:.4f} Test Acc {:.4f}'.format(eval_acc, test_acc))
            else:
                print('Eval Acc {:.4f} Test Acc {:.4f}'.format(eval_acc, test_acc))
            
            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc
                best_test_acc = test_acc

                if args.save_model:
                    if args.use_tt:
                        file_emb = 'emb_{}_tt{}_part{}.pt'.format(args.init, args.tt_rank, args.partition)
                        file_pred = 'pred_{}_tt{}_part{}.pt'.format(args.init, args.tt_rank, args.partition)
                    else:
                        file_emb = 'emb_{}_baseline.pt'.format(args.init)
                        file_pred = 'pred_{}_baseline.pt'.format(args.init)
                    th.save(model.state_dict(), file_emb)
                    th.save(pred, file_pred)
                    print('Save model params')
            if args.logging:
                log.logger.info('Best Eval Acc {:.4f} Test Acc {:.4f}'.format(best_eval_acc, best_test_acc))
            else:
                print('Best Eval Acc {:.4f} Test Acc {:.4f}'.format(best_eval_acc, best_test_acc))
    if args.logging:
        log.logger.info('Avg epoch time: {:.4f}'.format(total_time / args.epochs))
        log.logger.info('End2End Time(s): {:.4f}'.format(total_time))
    else:
        print('Avg epoch time: {:.4f}'.format(total_time / args.epochs))
        print('End2End Time(s): {:.4f}'.format(total_time))

    return best_test_acc
        

if __name__ == '__main__':
    args = parse_args()
    
    if args.device == 'cuda':
        device = th.device('cuda:0')
    else:
        device = th.device('cpu')

    # load ogbn-products data - dgl version
    target_dataset = args.dataset
    root = os.path.join(os.environ['HOME'], 'gnn_related', 'dataset')

    train_loader, full_neighbor_loader, data = dgl_graph_loader(target_dataset, root, device, args)
    print('Init from {}'.format(args.init))
    
    best_test_acc = run(train_loader, full_neighbor_loader, data, args)
    print('The Best Test Acc {:.4f}'.format(best_test_acc))

    # Run 10 times, n_runs == 10
    # test_accs = []
    # for i in range(int(args.n_runs)):
    #     test_accs.append(run(train_loader, full_neighbor_loader, data, args).cpu().numpy())
    #     print('Average test accuracy:', np.mean(test_accs), '±', np.std(test_accs))