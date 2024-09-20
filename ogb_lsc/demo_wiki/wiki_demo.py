import numpy as np

from ogb.lsc import WikiKG90Mv2Dataset, WikiKG90Mv2Evaluator
from ogb.linkproppred import LinkPropPredDataset, Evaluator
import torch
from torch.utils.data import Dataset, DataLoader
from dataloader import TrainDataset

"""
WikiKG90Mv2 is a Knowledge Graph (KG) extracted from the entire Wikidata knowledge base
- Each triple (head, relation, tail) represents an Wikidata claim, where head and tail are the Wikidata items, and relation is the Wikidata predicate
- 91,230,610 entities (Nodes), 1,387 relations, and 601,062,811 triples (Edges)
Task: Link Prediction
Given a set of training triples, predict a set of new test triples
For evaluation, for each test triple, (head, relation, tail), the model is asked to predict tail entity from (head, relation)
The goal is to rank the ground-truth tail entity as high in the rank as possible within the top 10, which is measured by Mean Reciprocal Rank (MRR)

#Nodes: 91,230,610, #Edges: 601,062,811

A smaller version:
@ogbl-wikikg2
#Node: 2,500,604, #Edges: 17,137,181
@ogbl-biokg
#Nodes: 93,773, #Edges: 5,088,434
"""

class WikiKGDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.num_triplets = dataset.train_hrt.shape[0]  # Number of triplets in the dataset

    def __len__(self):
        return self.num_triplets

    def __getitem__(self, idx):
        triplet = self.dataset.train_hrt[idx]  # Fetch the (head, relation, tail) triplet
        head, relation, tail = triplet
        return {
            'head': torch.tensor(head, dtype=torch.long),
            'relation': torch.tensor(relation, dtype=torch.long),
            'tail': torch.tensor(tail, dtype=torch.long)
        }

if __name__ == '__main__':
    root = '/home/shenghao/gnn_related/dataset'
    dataset = WikiKG90Mv2Dataset(root = root)
    print(dataset)
    print(dataset.num_entities)
    print(dataset.entity_feat)
    print(dataset.entity_feat.shape)
    print(dataset.num_relations)
    print(dataset.relation_feat)
    print(dataset.all_relation_feat)
    print(dataset.relation_feat.shape)
    print(dataset.train_hrt)
    print(dataset.valid_dict)
    print(dataset.test_dict(mode = 'test-dev'))
    print(dataset.test_dict(mode = 'test-challenge'))
    
    # print("======= For torch Dataloader =======")
    # torch_dataset = WikiKGDataset(dataset)
    # dataloader = DataLoader(torch_dataset, batch_size=128, shuffle=True, num_workers=4)
    # for batch in dataloader:
    #     heads = batch['head']
    #     relations = batch['relation']
    #     tails = batch['tail']

    
    print("======= a smaller wiki dataset =======")
    dataset_obgl = LinkPropPredDataset(name='ogbl-wikikg2', root='/home/shenghao/gnn_related/dataset')
    split_dict = dataset_obgl.get_edge_split()    
    nentity = dataset_obgl.graph['num_nodes']
    nrelation = int(max(dataset_obgl.graph['edge_reltype'])[0])+1
    split_dict = dataset_obgl.get_edge_split()
    train_triples = split_dict['train']
    print('#train: %d' % len(train_triples['head']))
    valid_triples = split_dict['valid']
    print('#valid: %d' % len(valid_triples['head']))
    test_triples = split_dict['test']
    print('#test: %d' % len(test_triples['head']))
    
    from collections import defaultdict
    train_count, train_true_head, train_true_tail = defaultdict(lambda: 4), defaultdict(list), defaultdict(list)
    for i in range(len(train_triples['head'])):
        head, relation, tail = train_triples['head'][i], train_triples['relation'][i], train_triples['tail'][i]
        train_count[(head, relation)] += 1
        train_count[(tail, -relation-1)] += 1
        train_true_head[(relation, tail)].append(head)
        train_true_tail[(head, relation)].append(tail)
    negative_sample_size = 128
    batch_size = 256
    for_dataloader = TrainDataset(train_triples, nentity, nrelation, 
                negative_sample_size, 'head-batch',
                train_count, train_true_head, train_true_tail)
    train_dataloader_head = DataLoader(
            TrainDataset(train_triples, nentity, nrelation, 
                negative_sample_size, 'head-batch',
                train_count, train_true_head, train_true_tail), 
            batch_size=batch_size,
            shuffle=True, 
            num_workers=2,
            collate_fn=TrainDataset.collate_fn
        )
        
    train_dataloader_tail = DataLoader(
            TrainDataset(train_triples, nentity, nrelation, 
                negative_sample_size, 'tail-batch',
                train_count, train_true_head, train_true_tail), 
            batch_size=batch_size,
            shuffle=True, 
            num_workers=2,
            collate_fn=TrainDataset.collate_fn
        )
    
    breakpoint()
    
    evaluator = WikiKG90Mv2Evaluator()

    t = np.random.randint(10000000, size = (10000,))
    t_pred_top10 = np.random.randint(10000000, size = (10000,10))

    rank = np.random.randint(10, size = (10000,))
    t_pred_top10[np.arange(len(rank)), rank] = t

    print(evaluator.eval({'h,r->t': {'t': t, 't_pred_top10': t_pred_top10}}))
    print(np.mean(1./(rank + 1)))

    t_pred_top10 = np.random.randint(10000000, size = (15000,10))
    evaluator.save_test_submission(
        input_dict = {'h,r->t': {'t_pred_top10': t_pred_top10}},
        dir_path = 'results',
        mode = 'test-dev',
    )