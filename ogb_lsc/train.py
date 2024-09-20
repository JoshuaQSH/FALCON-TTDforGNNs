import time
import random
import os

import dgl
import dgl.function as fn
import dgl.nn as dglnn

import numpy as np
import ogb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm, trange
from ogb.lsc import MAG240MDataset, MAG240MEvaluator
from ogb.lsc import DglPCQM4Mv2Dataset, PCQM4Mv2Evaluator
from ogb.lsc import WikiKG90Mv2Dataset


import sys
sys.path.insert(0, '/home/shenghao/home-3090/tensor-train-for-gcn/TT4GNN')
# sys.path.insert(0, '/home/shenghao/tensor-train-for-gcn/TT4GNN')
from tt_utils import *
from gnn_model import GNN, ExternalNodeCollator, RGAT
from FBTT.tt_embeddings_ops import TTEmbeddingBag

def collate_dgl(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    labels = torch.stack(labels)

    return batched_graph, labels

def run_wikiv2(args, dataset, train_loader, valid_loader, evaluator, device):
    pass

def train_pcaqm4m(model, device, data_loader, optimizer, epoch):
    model.train()
    loss_accum = 0
    reg_criterion = nn.L1Loss()
    
    for step, (bg, labels) in enumerate(data_loader):
        bg = bg.to(device)
        x = bg.ndata.pop("feat")
        edge_attr = bg.edata.pop("feat")
        labels = labels.to(device)

        pred = model(bg, x, edge_attr).view(-1,)
        optimizer.zero_grad()
        loss = reg_criterion(pred, labels)
        loss.backward()
        optimizer.step()
        
        train_mae = evaluator.eval({"y_pred": pred, "y_true": labels})["mae"]
        loss_accum += loss.detach().cpu().item()

        if step % 20 == 0:
            gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
            print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train MAE {:.4f} | GPU {:.1f} MB'.format(
                    epoch, step, loss.detach().cpu().item(), train_mae, gpu_mem_alloc))

    # loss, pred, acc, fwd_throughput, bwd_throughput, iter_tput_per_epoch, train_mae
    return loss_accum, loss_accum / (step + 1)

def eval_pcaqm4m(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []
    val_loss = 0
    reg_criterion = nn.L1Loss()

    for step, (bg, labels) in enumerate(tqdm(loader, desc="Iteration")):
        bg = bg.to(device)
        x = bg.ndata.pop("feat")
        edge_attr = bg.edata.pop("feat")
        labels = labels.to(device)
        
        with torch.no_grad():
            pred = model(bg, x, edge_attr).view(
                -1,
            )

        loss = reg_criterion(pred, labels)
        val_loss += loss.detach().cpu().item()

        y_true.append(labels.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    input_dict = {"y_true": y_true, "y_pred": y_pred}
    
    # val_loss, valid_mae
    return val_loss, evaluator.eval(input_dict)["mae"]

def run_pcaqm4m(args, train_loader, valid_loader, evaluator, device, dist=None):
    
    ### setting the hyperparameters
    shared_params = {
        "num_layers": args.num_layers,
        "emb_dim": args.num_hidden,
        "drop_ratio": args.dropout,
        "graph_pooling": "sum",
    }

    ### choosing model
    if args.model == "gin":
        model = GNN(gnn_type="gin", virtual_node=False, **shared_params).to(
            device
        )
    elif args.model == "gcn":
        model = GNN(gnn_type="gcn", virtual_node=False, **shared_params).to(
            device
        )
    else:
        raise ValueError("Invalid GNN type")
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"#Params: {num_params}")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=100, verbose=True, min_lr=1e-3
    )

    best_valid_mae = 1000

    for epoch in range(0, args.epochs):
         print("===== Epoch {} =====".format(epoch))
         print("Training...")
         loss, train_mae = train_pcaqm4m(model, device, train_loader, optimizer, epoch)
         
         print("Evaluating...")
         val_loss, valid_mae = eval_pcaqm4m(model, device, valid_loader, evaluator)
         print({"Train": train_mae, "Validation": valid_mae})
         if valid_mae < best_valid_mae:
             best_valid_mae = valid_mae

         lr_scheduler.step(loss)
    
    print(f"Best validation MAE so far: {best_valid_mae}")

def test_mag240m(model, dataset, g, feats, paper_offset, submission_path="/home/shenghao/gnn_related/"):
    
    valid_idx = torch.LongTensor(dataset.get_idx_split("valid")) + paper_offset
    test_idx = torch.LongTensor(dataset.get_idx_split("test")) + paper_offset
    test_idx = torch.LongTensor(dataset.get_idx_split("test")) + paper_offset

    label = dataset.paper_label
    print("Initializing data loader...")
    sampler = dgl.dataloading.MultiLayerNeighborSampler([160, 160])
    valid_collator = ExternalNodeCollator(
        g, valid_idx, sampler, paper_offset, feats, label
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_collator.dataset,
        batch_size=16,
        shuffle=False,
        drop_last=False,
        collate_fn=valid_collator.collate,
        num_workers=2,
    )
    test_collator = ExternalNodeCollator(
        g, test_idx, sampler, paper_offset, feats, label
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_collator.dataset,
        batch_size=16,
        shuffle=False,
        drop_last=False,
        collate_fn=test_collator.collate,
        num_workers=4,
    )
    
    model.eval()
    y_preds = []
    for i, (input_nodes, output_nodes, mfgs) in enumerate(test_dataloader):
        with torch.no_grad():
            mfgs = [g.to("cuda") for g in mfgs]
            x = mfgs[0].srcdata["x"]
            y = mfgs[-1].dstdata["y"]
            y_hat = model(mfgs, x)
            y_preds.append(y_hat.argmax(1).cpu())
    evaluator.save_test_submission(
        {"y_pred": torch.cat(y_preds)}, submission_path
    )

def eval_mag240m(model, device, data_loader):
    model.eval()
    correct = total = 0
    
    for i, (input_nodes, output_nodes, mfgs) in enumerate(tqdm(data_loader)):
        with torch.no_grad():
                mfgs = [g.to("cuda") for g in mfgs]
                x = mfgs[0].srcdata["x"]
                y = mfgs[-1].dstdata["y"]
                y_hat = model(mfgs, x)
                correct += (y_hat.argmax(1) == y).sum().item()
                total += y_hat.shape[0]
        acc = correct / total
        print("Validation accuracy:", acc)
        
    return acc

def train_mag240m(model, device, data_loader, optimizer, epoch):
    model.train()
    loss_accum = 0
    for step, (input_nodes, output_nodes, mfgs) in enumerate(data_loader):
        mfgs = [g.to("cuda") for g in mfgs]
        x = mfgs[0].srcdata["x"]
        y = mfgs[-1].dstdata["y"]
        y_hat = model(mfgs, x)
        loss = F.cross_entropy(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = (y_hat.argmax(1) == y).float().mean()
        loss_accum += loss.detach().cpu().item()
        
        if step % 20 == 0:
            gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
            print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train acc {:.4f} | GPU {:.1f} MB'.format(
                epoch, step, loss.detach().cpu().item(), acc, gpu_mem_alloc))
        return acc, loss_accum

def run_mag240m(args, dataset, train_loader, valid_loader, device):
    print("Initializing model...")
    model = RGAT(
        dataset.num_paper_features,
        dataset.num_classes,
        args.num_hidden, # 1024
        5,
        args.num_layers, # 2
        args.num_heads, # 4
        args.dropout,
        "paper",)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr) # 0.001
    sched = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.25)

    best_acc = 0

    for epoch in range(args.epochs):
        train_acc, loss_accum = train_mag240m(model, device, train_loader, optimizer, epoch)
        acc = eval_mag240m(model, device, valid_loader)
        if best_acc < acc:
            best_acc = acc
            print("Updating best model...")
            # torch.save(model.state_dict(), args.model_path)
        print("--- Best Acc {:.4f} ---".format(best_acc))
        sched.step()
    
    if args.testing:
        paper_offset = dataset.num_authors + dataset.num_institutions
        # test_mag240m(model, dataset, g, feats, paper_offset)

def load_mag240m_features(dataset):
    ei_writes = dataset.edge_index("author", "writes", "paper")
    ei_cites = dataset.edge_index("paper", "paper")
    ei_affiliated = dataset.edge_index("author", "institution")
    # We sort the nodes starting with the papers, then the authors, then the institutions.
    author_offset = 0
    inst_offset = author_offset + dataset.num_authors
    paper_offset = inst_offset + dataset.num_institutions
    paper_feat = dataset.paper_feat
    g = dgl.heterograph(
        {
            ("author", "write", "paper"): (ei_writes[0], ei_writes[1]),
            ("paper", "write-by", "author"): (ei_writes[1], ei_writes[0]),
            ("author", "affiliate-with", "institution"): (
                ei_affiliated[0],
                ei_affiliated[1],
            ),
            ("institution", "affiliate", "author"): (
                ei_affiliated[1],
                ei_affiliated[0],
            ),
            ("paper", "cite", "paper"): (
                np.concatenate([ei_cites[0], ei_cites[1]]),
                np.concatenate([ei_cites[1], ei_cites[0]]),
            ),
        }
    )
    
    # Required by full_feat
    author_feat = np.empty((dataset.num_authors, dataset.num_paper_features), dtype=np.float16)
    inst_feat = np.empty((dataset.num_institutions, dataset.num_paper_features), dtype=np.float16)
    BLOCK_COLS = 16
    with trange(0, dataset.num_paper_features, BLOCK_COLS) as tq:
        for start in tq:
            tq.set_postfix_str("Reading paper features...")
            g.nodes["paper"].data["x"] = torch.FloatTensor(
                paper_feat[:, start : start + BLOCK_COLS].astype("float32")
            )
            # Compute author features...
            tq.set_postfix_str("Computing author features...")
            g.update_all(fn.copy_u("x", "m"), fn.mean("m", "x"), etype="write-by")
            # Then institution features...
            tq.set_postfix_str("Computing institution features...")
            g.update_all(
                fn.copy_u("x", "m"), fn.mean("m", "x"), etype="affiliate-with"
            )
            tq.set_postfix_str("Writing author features...")
            author_feat[:, start : start + BLOCK_COLS] = (
                g.nodes["author"].data["x"].numpy().astype("float16")
            )
            tq.set_postfix_str("Writing institution features...")
            inst_feat[:, start : start + BLOCK_COLS] = (
                g.nodes["institution"].data["x"].numpy().astype("float16")
            )
            del g.nodes["paper"].data["x"]
            del g.nodes["author"].data["x"]
            del g.nodes["institution"].data["x"]

    g = dgl.to_homogeneous(g)
    assert torch.equal(
        g.ndata[dgl.NTYPE],
        torch.cat(
            [
                torch.full((dataset.num_authors,), 0),
                torch.full((dataset.num_institutions,), 1),
                torch.full((dataset.num_papers,), 2),
            ]
        ),
    )
    assert torch.equal(
        g.ndata[dgl.NID],
        torch.cat(
            [
                torch.arange(dataset.num_authors),
                torch.arange(dataset.num_institutions),
                torch.arange(dataset.num_papers),
            ]
        ),
    )

    g.edata["etype"] = g.edata[dgl.ETYPE].byte()
    
    del g.edata[dgl.ETYPE]
    del g.ndata[dgl.NTYPE]
    del g.ndata[dgl.NID]

    # Process feature
    full_feat = np.empty((dataset.num_authors + dataset.num_institutions + dataset.num_papers,
            dataset.num_paper_features), dtype=np.float16)
 
    BLOCK_ROWS = 100000
    for start in trange(0, dataset.num_authors, BLOCK_ROWS):
        end = min(dataset.num_authors, start + BLOCK_ROWS)
        full_feat[author_offset + start : author_offset + end] = author_feat[
            start:end
        ]
    for start in trange(0, dataset.num_institutions, BLOCK_ROWS):
        end = min(dataset.num_institutions, start + BLOCK_ROWS)
        full_feat[inst_offset + start : inst_offset + end] = inst_feat[
            start:end
        ]
    for start in trange(0, dataset.num_papers, BLOCK_ROWS):
        end = min(dataset.num_papers, start + BLOCK_ROWS)
        full_feat[paper_offset + start : paper_offset + end] = paper_feat[
            start:end
        ]

    return full_feat

def load_data(args, root):
    dataset, train_loader, valid_loader, test_loader, evaluator = None, None, None, None, None
    if args.dataset == "mag240m":
        print("==== Preparing MAG240M data ====")
        dataset = MAG240MDataset(root=root)
        
        # A hack here, should load the features from the directory
        preprocessed = False
        
        if preprocessed:
            graph_path = '/home/shenghao/gnn_related/dataset/MAG240MFILES/graph.dgl'
            full_feature_path = '/home/shenghao/gnn_related/dataset/MAG240MFILES/full.npy'
            
            print("Loading graph")
            (g,), _ = dgl.load_graphs(graph_path)
            g = g.formats(["csc"])

            # Not enough space storing the features, need to do it every time
            print("Loading features")
            paper_offset = dataset.num_authors + dataset.num_institutions
            num_nodes = paper_offset + dataset.num_papers
            num_features = dataset.num_paper_features
            feats = load_mag240m_features(dataset)
            # feats = np.memmap(
            #     full_feature_path,
            #     mode="r",
            #     dtype="float16",
            #     shape=(num_nodes, num_features),
            # )
        else:
            # graph_path = '/home/shenghao/gnn_related/dataset/MAG240MFILES/graph.dgl'
            # full_feature_path = '/home/shenghao/gnn_related/dataset/MAG240MFILES/full_graph_dataset/full.npy'
            graph_path = '/home/shenghao/home-3090/gnn_related/dataset/MAG240MFILES/graph.dgl'
            full_feature_path = '/home/shenghao/gnn_related/full_graph_dataset/full.npy'
            print("Loading graph")
            (g,), _ = dgl.load_graphs(graph_path)
            g = g.formats(["csc"])
            print("Loading features")
            paper_offset = dataset.num_authors + dataset.num_institutions
            num_nodes = paper_offset + dataset.num_papers
            num_features = dataset.num_paper_features
            feats = np.memmap(
                full_feature_path,
                mode="r",
                dtype="float16",
                shape=(num_nodes, num_features),
            )

        print("Loading masks and labels")
        train_idx = torch.LongTensor(dataset.get_idx_split("train")) + paper_offset
        valid_idx = torch.LongTensor(dataset.get_idx_split("valid")) + paper_offset

        label = dataset.paper_label
        evaluator = MAG240MEvaluator()
        print("Initializing dataloader...")
        # Should be alingned to the fanout, args.fan_out
        sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 25])
        train_collator = ExternalNodeCollator(
            g, train_idx, sampler, paper_offset, feats, label
        )
        valid_collator = ExternalNodeCollator(
            g, valid_idx, sampler, paper_offset, feats, label
        )

        train_loader = DataLoader(
            train_collator.dataset,
            batch_size=args.batch,
            shuffle=True,
            drop_last=False,
            collate_fn=train_collator.collate,
            num_workers=args.num_workers,
        )
        valid_loader = DataLoader(
            valid_collator.dataset,
            batch_size=args.batch,
            shuffle=True,
            drop_last=False,
            collate_fn=valid_collator.collate,
            num_workers=args.num_workers,
        )

    elif args.dataset == 'pcaqm4m':
        print("==== Preparing PCQM4Mv2 data ====")
        dataset = DglPCQM4Mv2Dataset(root=root)
        split_idx = dataset.get_idx_split()
        evaluator = PCQM4Mv2Evaluator()
        train_loader = DataLoader(
            dataset[split_idx["train"]],
            batch_size=args.batch,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_dgl,
        )
        valid_loader = DataLoader(
            dataset[split_idx["valid"]],
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_dgl,
        )

    return dataset, train_loader, valid_loader, evaluator

"""
@pcaqm4m: run in 3090
@mag240m: run in 4090
@wikiv2: run in 3090
"""
if __name__ == "__main__":

    # graph-path ./graph.dgl
    # full-feature-path ./full.npy
    # model-path ./model.pt
    args = parse_args()
    target_dataset = args.dataset
    

    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")

    if args.data_dir == 'None':
        root = os.path.join(os.environ['HOME'], args.workspace, 'gnn_related', 'dataset')
    else:
        root = args.data_dir
    dataset, train_loader, valid_loader, evaluator = load_data(args, root)
    if args.dataset == 'pcaqm4m':
        run_pcaqm4m(args, train_loader, valid_loader, evaluator, device)
    elif args.dataset == 'mag240m':
        run_mag240m(args, dataset, train_loader, valid_loader, device)
    elif args.dataset == 'wikiv2':
        run_wikiv2(args, dataset, train_loader, valid_loader, evaluator, device)
    else:
        print("Invalid dataset")