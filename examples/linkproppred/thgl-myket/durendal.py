import numpy as np
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
from tgb.linkproppred.evaluate import Evaluator
from torch_geometric.loader import TemporalDataLoader
from tqdm import tqdm
import timeit
from torch_geometric.data import Data, HeteroData
import torch
from tgb_modules.durendal import training_durendal_uta
from torch_geometric.transforms import RandomLinkSplit
from tgb.utils.utils import save_results
import sys
import argparse
import random
import os
import os.path as osp
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser('*** TGB: Durendal ***')
    parser.add_argument('--seed', type=int, help='Random seed', default=1)

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args, sys.argv 

DATA = "thgl-myket"

args, _ = get_args()

SEED = args.seed
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)

MODEL_NAME = 'DURENDAL_UTA'

# data loading
dataset = PyGLinkPropPredDataset(name=DATA, root="datasets")
train_mask = dataset.train_mask
val_mask = dataset.val_mask
test_mask = dataset.test_mask
data = dataset.get_TemporalData()
metric = dataset.eval_metric
evaluator = Evaluator(name=DATA)


# for saving the results...
results_path = f'{osp.dirname(osp.abspath(__file__))}/saved_results'
if not osp.exists(results_path):
    os.mkdir(results_path)
    print('INFO: Create directory {}'.format(results_path))
Path(results_path).mkdir(parents=True, exist_ok=True)
results_filename = f'{results_path}/{MODEL_NAME}_{DATA}_results.json'

print ("there are {} nodes and {} edges".format(dataset.num_nodes, dataset.num_edges))
print ("there are {} relation types".format(dataset.num_rels))


timestamp = data.t
head = data.src
tail = data.dst
edge_type = data.edge_type #relation

#! node type is a property of the dataset not the temporal data as temporal data has one entry per edge
node_type = dataset.node_type #node types
neg_sampler = dataset.negative_sampler

print ("shape of edge type is", edge_type.shape)
print ("shape of node type is", node_type.shape)

train_data = data[train_mask]
val_data = data[val_mask]
test_data = data[test_mask]
print ("finished loading PyG data")

BATCH_SIZE = 1000
edge_types = np.unique(edge_type)

def create_snapshot(data):
    edge_index_dict = {('node', str(r), 'node'): [[],[]] for r in edge_types}
    snap = HeteroData()
    if dataset.node_feat is not None:
        snap['node'].x = dataset.node_feat
    else:
        snap['node'].x = torch.Tensor([[1] for i in range(dataset.num_nodes)])
    for j in range(len(data)):
        src, dst, t, rel = data.src[j], data.dst[j], data.t[j], data.edge_type[j]
        edge_index_dict['node',f'{rel}','node'][0].append(src)
        edge_index_dict['node',f'{rel}','node'][1].append(dst)
    for edge_t, edge_index in edge_index_dict.items():
        snap[edge_t].edge_index = torch.Tensor(edge_index).long()
    return snap

def create_snapshots(data, snapshot_size):
    snapshots = []
    for snap_i in range(0, len(data), snapshot_size):
        snap = create_snapshot(data[snap_i:snap_i+snapshot_size])
        print(snap)
        snapshots.append(snap)
    return snapshots

@torch.no_grad()
def test(data_loader, past_dict_1, past_dict_2, split_mode):
    
    def create_test_snapshot(batch):
        """
        Create test snapshot using the positive and negative samples.

        For each data point we put an edge between the source node and the positive destination
        node, followed by edges between the source node and the negative destination nodes.

        Args:
            batch: batch data.

        Returns:
            test_snap: test snapshot.
        """
        test_snap = create_snapshot(batch)
        edge_type_keys = list(test_snap.edge_index_dict.keys())
        print(test_snap)
        for edge_type in edge_type_keys:
            edge_index = test_snap.edge_index_dict[edge_type]
            edge_label = []
            edge_label_index = []
            edge_label_per_edge = [1] + [0 for i in range(len(neg_batch_list[0]))]
            for i in range(edge_index.shape[1]):
                edge_label.extend(edge_label_per_edge)
                src_node = test_snap[edge_type].edge_index[0, i]
                src_list = ([src_node for j in range(len(edge_label_per_edge))])
                dst_list = [src_node] + list(neg_batch_list[i])
                edge_label_index.append(torch.Tensor([src_list, dst_list]))
            test_snap[edge_type].edge_label = torch.Tensor(edge_label).long()
            test_snap[edge_type].edge_label_index = torch.cat(edge_label_index, dim=1).long()
        return test_snap, edge_type_keys
    
    perf_list = []
    for batch in tqdm(data_loader):
        src, pos_dst, t, msg, rel = batch.src, batch.dst, batch.t, batch.msg, batch.edge_type
        neg_batch_list = neg_sampler.query_batch(src.detach().cpu().numpy(), pos_dst.detach().cpu().numpy(), t.detach().cpu().numpy(), rel.detach().cpu().numpy(), split_mode=split_mode)
        test_snap, edge_type_keys = create_test_snapshot(batch)
        pred_dict, past_dict_1, past_dict_2 =\
                model(test_snap.x_dict, test_snap.edge_index_dict, test_snap,\
                    1, past_dict_1, past_dict_2) # snap_i = 1 so that it is not zero
        
        for edge_type in edge_type_keys:    
            h = pred_dict[edge_type]
            pred_cont = torch.sigmoid(h).cpu().detach().numpy()
            edge_index = test_snap.edge_index_dict[edge_type]
            for i in range(edge_index.shape[1]):
                pos_i = i * (len(neg_batch_list[0]) + 1)
                input_dict = {
                    "y_pred_pos": np.array([pred_cont[pos_i]]),
                    "y_pred_neg": np.array(pred_cont[pos_i+1:pos_i+len(neg_batch_list[0])+1]),
                    "eval_metric": [metric],
                }
                perf_list.append(evaluator.eval(input_dict)[metric])
    perf_metrics = float(np.mean(perf_list))

    return perf_metrics


train_snapshots = create_snapshots(train_data, BATCH_SIZE)
model, past_dict_1, past_dict_2, durendalopt = training_durendal_uta(train_snapshots,  hidden_conv_1=256, hidden_conv_2=128)


val_loader = TemporalDataLoader(val_data, batch_size=BATCH_SIZE)
test_loader = TemporalDataLoader(test_data, batch_size=BATCH_SIZE)


dataset.load_val_ns()


# testing ...
start_val = timeit.default_timer()
perf_metric_test = test(val_loader, past_dict_1, past_dict_2, split_mode='val')
end_val = timeit.default_timer()

print(f"INFO: val: Evaluation Setting: >>> ONE-VS-MANY <<< ")
print(f"\tval: {metric}: {perf_metric_test: .4f}")
val_time = timeit.default_timer() - start_val
print(f"\tval: Elapsed Time (s): {val_time: .4f}")


dataset.load_test_ns()

# testing ...
start_test = timeit.default_timer()
perf_metric_test = test(test_loader, past_dict_1, past_dict_2, split_mode='test')
end_test = timeit.default_timer()

print(f"INFO: Test: Evaluation Setting: >>> ONE-VS-MANY <<< ")
print(f"\tTest: {metric}: {perf_metric_test: .4f}")
test_time = timeit.default_timer() - start_test
print(f"\tTest: Elapsed Time (s): {test_time: .4f}")

save_results({'model': MODEL_NAME,
            'data': DATA,
            'run': 1,
            'seed': SEED,
            metric: perf_metric_test,
            'test_time': test_time,
            'tot_train_val_time': 'NA'
            }, 
    results_filename)