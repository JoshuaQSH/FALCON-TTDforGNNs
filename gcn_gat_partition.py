import argparse
import math
import time
import os
import random
import sys
import math

import dgl
import scipy
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

from gnn_model import GCN, GAT
from tt_utils import *
from graphloader import dgl_graph_loader
from utils import Logger, gpu_timing, memory_usage, calculate_access_percentages, plot_access_percentages

# sys.path.insert(0, '/home/shenghao/FBTT-Embedding')
# sys.path.insert(0, '/home/shenghao/home-3090/FBTT-Embedding')
# from tt_embeddings_ops import TTEmbeddingBag
from FBTT.tt_embeddings_ops import TTEmbeddingBag
from Efficient_TT.efficient_tt import Eff_TTEmbedding
# from Efficient_TT.breakdown.efficient_tt import Eff_TTEmbedding



# device = None
# in_feats, n_classes = None, None

# GCN so far
def gen_model(args, in_feats, n_classes):
    model = None
    if args.model == 'gcn': 
        if args.use_labels:
            model = GCN(
                in_feats + n_classes, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout, args.use_linear)
        else:
            model = GCN(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout, args.use_linear)
    elif args.model == 'gat':
        model = GAT(in_feats, n_classes, args.num_hidden, args.num_layers, args.num_heads, F.relu, args.dropout)
        # model = GAT_new(in_feats, args.num_hidden, n_classes, heads=[8, 1])
    
    return model


def cross_entropy(x, labels):
    epsilon = 1 - math.log(2)
    # y = F.cross_entropy(x, labels[:, 0], reduction="none")
    y = F.cross_entropy(x, labels[:], reduction="none")
    y = th.log(epsilon + y) - math.log(epsilon)
    return th.mean(y)

def BCE(x, labels):
    m = nn.Sigmoid()
    x = th.argmax(m(x),dim=1)
    loss_fn = nn.BCELoss(reduction="mean")
    y = loss_fn(x.type(th.FloatTensor), labels.type(th.FloatTensor))
    return y

def compute_acc(pred, labels, evaluator):
    return evaluator.eval({"y_pred": pred.argmax(dim=-1, keepdim=True), "y_true": labels.unsqueeze(1)})["acc"]


def add_labels(feat, labels, idx, n_classes):
    onehot = th.zeros([feat.shape[0], n_classes]).to(device)
    # hack here
    # onehot[idx, labels[idx, 0]] = 1
    onehot[idx, labels[idx]] = 1
    return th.cat([feat, onehot], dim=-1)


def adjust_learning_rate(optimizer, lr, epoch):
    if epoch <= 50:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr * epoch / 50


def train(epoch, model, graph, emb_layer, labels, train_idx, optimizer, n_classes, evaluator, train_loader, log, args):    
    
    # For the throughput testing
    ave_forward_throughput=[]
    ave_backward_throughput=[]
    tic_step = time.time()
    seeds = th.arange(graph.number_of_nodes()).to(labels.device)
    model.train()
    if args.use_tt:
        ids = th.arange(graph.number_of_nodes()).to(labels.device)
        offsets = th.arange(graph.number_of_nodes() + 1).to(labels.device)
        feat = emb_layer(ids, offsets)
    else:
        ids = th.arange(graph.number_of_nodes()).to(labels.device)
        feat = emb_layer(ids)
    
    if args.use_labels:
        mask_rate = 0.5
        mask = th.rand(train_idx.shape) < mask_rate

        train_labels_idx = train_idx[mask]
        train_pred_idx = train_idx[~mask]

        feat = add_labels(feat, labels, train_labels_idx, n_classes)
    else:
        mask_rate = 0.5
        mask = th.rand(train_idx.shape) < mask_rate

        train_pred_idx = train_idx[mask]

    optimizer.zero_grad()


    # Model forward
    # FBTT pred - torch.Size([169343, 40])
    # Eff pred - torch.Size([169343, 40])
    pred = model(graph, feat)

    # Forward throughput
    fwd_throughput = len(seeds)/(time.time()-tic_step)

    loss = cross_entropy(pred[train_pred_idx], labels[train_pred_idx])
    # loss = BCE(pred[train_pred_idx], labels[train_pred_idx])

    loss.backward()
    optimizer.step()

    # Backward throughput
    bwd_throughput = len(seeds)/(time.time()-tic_step)

    acc = compute_acc(pred[train_idx], labels[train_idx], evaluator)
    gpu_mem_alloc = th.cuda.max_memory_allocated() / 1000000 if th.cuda.is_available() else 0
    
    iter_tput_per_epoch = len(seeds) / (time.time() - tic_step)

    if args.logging:
        log.logger.debug('Epoch {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'.format(
            epoch, loss.item(), acc, iter_tput_per_epoch, gpu_mem_alloc))
    else:
        print('Epoch {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'.format(
            epoch, loss.item(), acc, iter_tput_per_epoch, gpu_mem_alloc))

    return loss, pred, acc, fwd_throughput, bwd_throughput, iter_tput_per_epoch


@th.no_grad()
def evaluate(model, graph, emb_layer, labels, train_idx, val_idx, test_idx, evaluator, n_classes, args):
    model.eval()

    if args.use_tt:
        ids = th.arange(graph.number_of_nodes()).to(labels.device)
        offsets = th.arange(graph.number_of_nodes() + 1).to(labels.device)
        feat = emb_layer(ids, offsets)
    else:
        ids = th.arange(graph.number_of_nodes()).to(labels.device)
        feat = emb_layer(ids)

    if args.use_labels:
        feat = add_labels(feat, labels, train_idx, n_classes)

    pred = model(graph, feat)
    train_loss = cross_entropy(pred[train_idx], labels[train_idx])
    val_loss = cross_entropy(pred[val_idx], labels[val_idx])
    test_loss = cross_entropy(pred[test_idx], labels[test_idx])

    return (
        compute_acc(pred[train_idx], labels[train_idx], evaluator),
        compute_acc(pred[val_idx], labels[val_idx], evaluator),
        compute_acc(pred[test_idx], labels[test_idx], evaluator),
        train_loss,
        val_loss,
        test_loss,
    )


def run(args, device, data, train_loader, evaluator, dist=None):

    # Add logging
    # Setup the saved log file, with time and filename
    saved_log_path = '../../../logs/'
    start_time = int(round(time.time()*1000))
    timestamp = time.strftime('%Y%m%d-%H%M%S',time.localtime(start_time/1000))

    if args.logging:
        # saved_log_name = saved_log_path + '{}-{}-{}-{}.log'.format(args.model, args.dataset, args.batch, timestamp)
        saved_log_name = saved_log_path + 'Baseline-{}-{}-4090-r3-{}.log'.format(args.model, args.batch, timestamp)
        # saved_log_name = saved_log_path + 'Final-{}-4090-r3-{}.log'.format(args.model, timestamp)

        log = Logger(saved_log_name, level='debug')
        log.logger.debug("[Running GraphSAGE Model == Hidden: {}, Layers: {} ==]".format(args.num_hidden, args.num_layers))
        log.logger.debug("[Dataset: {}]".format(args.dataset))
    else:
        log = None

    # define model and optimizer
    train_idx, val_idx, test_idx, in_feats, labels, n_classes, nfeat, graph = data

    model = gen_model(args, in_feats, n_classes)
    model = model.to(device)

    if not args.use_tt:
        embed_layer = th.nn.Embedding(graph.number_of_nodes(), in_feats)
    else:
        tt_rank = [int(i) for i in args.tt_rank.split(',')]
        p_shapes = [int(i) for i in args.p_shapes.split(',')]
        q_shapes = [int(i) for i in args.q_shapes.split(',')]

        if args.emb_name == "fbtt":
            print("Using FBTT")
            embed_layer = TTEmbeddingBag(
                    num_embeddings=graph.number_of_nodes(),
                    embedding_dim=in_feats,
                    tt_ranks=tt_rank,
                    tt_p_shapes=p_shapes,
                    tt_q_shapes=q_shapes,
                    sparse=args.sparse,
                    use_cache=args.use_cached,
                    cache_size=args.cache_size,
                    weight_dist="normal",
                    )

        elif args.emb_name == "eff":
            print("Using Efficient TT")
            # eff_tag == 0: both efficient
            # eff_tag == 1: forward TT, backward efficient
            # eff_tag == 2: forward efficient, backward TT
            embed_layer = Eff_TTEmbedding(
                    num_embeddings = graph.number_of_nodes(),
                    embedding_dim = in_feats,
                    tt_p_shapes=p_shapes,
                    tt_q_shapes=q_shapes,
                    tt_ranks = tt_rank,
                    weight_dist = "uniform",
                    batch_size = 12
                    ).to(device)

        if dist == 'eigen':
            eigen_vals, eigen_vecs = get_eigen(graph, in_feats, 'ogbn-arxiv')
            eigen_vecs = th.tensor(eigen_vecs * np.sqrt(eigen_vals).reshape((1, len(eigen_vals))), dtype=th.float32)
            emb_pad = np.zeros(shape=(55*55*56, 128)).astype(np.float32)
            emb_pad[:eigen_vecs.shape[0], :] = eigen_vecs

            tt_cores, _ = tt_matrix_decomp(
                    emb_pad,
                    [1, tt_rank[0], tt_rank[1], 1],
                    p_shapes,
                    q_shapes
                )
            for i in range(3):
                embed_layer.tt_cores[i].data = tt_cores[i]

        elif dist == 'ortho':
            tt_cores = get_ortho(
                [1, tt_rank[0], tt_rank[1], 1],
                p_shapes,
                q_shapes # [8,4,4] for ogbn-arxiv
            )
            for i in range(3):
                embed_layer.tt_cores[i].data = th.tensor(tt_cores[i])
            

    embed_layer = embed_layer.to(device)
    graph = graph.to(device)

    ### Model's params
    params = list(model.parameters()) + list(embed_layer.parameters())
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    # optimizer = optim.SGD(params, lr=args.lr, weight_decay=args.wd)

    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=100, verbose=True, min_lr=1e-3
    )
    
    ### Model info
    # print(model)
    # print(embed_layer)
    # print(f"Number of params: {count_parameters(args, embed_layer, in_feats, n_classes)}")

    # training loop
    total_time = 0
    best_val_acc, best_test_acc, best_val_loss = 0, 0, float("inf")
    iter_tput, ave_fwd_throughput, ave_bwd_throughput = [], [], []

    accs, train_accs, val_accs, test_accs = [], [], [], []
    losses, train_losses, val_losses, test_losses = [], [], [], []

    for epoch in range(1, args.epochs + 1):
        tic = time.time()

        adjust_learning_rate(optimizer, args.lr, epoch)

        loss, pred, acc, fwd_throughput, bwd_throughput, iter_tput_per_epoch = train(epoch, model, graph, embed_layer, labels, train_idx, optimizer, n_classes, evaluator, train_loader, log, args)
        # acc = compute_acc(pred[train_idx], labels[train_idx], evaluator)

        # Record the throughput
        ave_fwd_throughput.append(fwd_throughput)
        ave_bwd_throughput.append(bwd_throughput)
        iter_tput.append(iter_tput_per_epoch)

        train_acc, val_acc, test_acc, train_loss, val_loss, test_loss = evaluate(
            model, graph, embed_layer, labels, train_idx, val_idx, test_idx, evaluator, n_classes, args
        )

        lr_scheduler.step(loss)

        toc = time.time()
        total_time += toc - tic

        if args.logging:
            log.logger.info('Epoch Time(s): {:.4f}'.format(toc - tic))
        else:
            print('Epoch Time(s): {:.4f}'.format(toc - tic))
        
        # if val_acc > best_val_acc:
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_test_acc = test_acc

        # if epoch % args.log_every == 0:
        #     print(f"Epoch: {epoch}/{args.epochs}")
        #     print(
        #         f"Loss: {loss.item():.4f}, Acc: {acc:.4f}\n"
        #         f"Train/Val/Test loss: {train_loss:.4f}/{val_loss:.4f}/{test_loss:.4f}\n"
        #         f"Train/Val/Test/Best val/Best test acc: {train_acc:.4f}/{val_acc:.4f}/{test_acc:.4f}/{best_val_acc:.4f}/{best_test_acc:.4f}"
        #     )

        for l, e in zip(
            [accs, train_accs, val_accs, test_accs, losses, train_losses, val_losses, test_losses],
            [acc, train_acc, val_acc, test_acc, loss, train_loss, val_loss, test_loss],
        ):
            l.append(e)

        if epoch == args.epochs and args.store_emb:
            emb = np.array(embed_layer.cpu().weight.data)
            np.save('gcn_full_emb_{}.npy'.format(n_running), emb)
            embed_layer.to(device)

    if args.logging:
        log.logger.info('Avg epoch time: {:.4f}'.format(total_time / args.epochs))
        log.logger.info('End2End Time(s): {:.4f}'.format(total_time))
        log.logger.info('Avg forward throughput is {:.4f}'.format(np.mean(ave_fwd_throughput)))
        log.logger.info('Avg backward throughput is {:.4f}'.format(np.mean(ave_bwd_throughput)))
        log.logger.info('Avg overall throughput is {:.4f}'.format(np.mean(iter_tput[2:])))
        log.logger.info('Test acc {:.4f}'.format(best_test_acc))
    
    else:
        print(f"Avg epoch time: {total_time / args.epochs}")
        print('End2End Time(s): {:.4f}'.format(total_time))
        print('Avg forward throughput is {:.4f}'.format(np.mean(ave_fwd_throughput)))
        print('Avg backward throughput is {:.4f}'.format(np.mean(ave_bwd_throughput)))
        print('Avg overall throughput is {:.4f}'.format(np.mean(iter_tput[2:])))
        print(f"Test acc: {best_test_acc}")

        # print(f"Number of params: {count_parameters(args, embed_layer, in_feats, n_classes)}")

    return best_test_acc


def count_parameters(args, emb_layer, in_feats, n_classes):
    model = gen_model(args, in_feats, n_classes)
    #return sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
    from_model = sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
    from_embed = sum([np.prod(p.size()) for p in emb_layer.parameters() if p.requires_grad])
    return from_model + from_embed


if __name__ == "__main__":
    args = parse_args()

    if args.device != 'cpu' and th.cuda.is_available():
        device = th.device(args.device)
    else:
        device = th.device('cpu')
    
    # load ogbn-products data - dgl version
    target_dataset = args.dataset
    if args.dataset_dir == None:
        root = os.path.join(os.environ['HOME'], args.workspace, 'gnn_related', 'dataset')
    else:
        root = args.dataset_dir
    
    train_loader, full_neighbor_loader, data = dgl_graph_loader(target_dataset, root, device, args)
    print('Init from {}'.format(args.init))
    evaluator = Evaluator(name=target_dataset)
    
    best_test_acc = run(args, device, data, train_loader, evaluator, dist=args.init)
    print('The Best Test Acc {:.4f}'.format(best_test_acc))

    # for i in range(int(args.n_runs)):
    #     test_accs.append(run(args, device, data, i, evaluator, dist=args.init))
    #     print('Average test accuracy:', np.mean(test_accs), '±', np.std(test_accs))
    