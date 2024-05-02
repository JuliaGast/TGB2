"""
Temporal Knowledge Graph Reasoning Based on Evolutional Representation Learning
Reference:
- https://github.com/Lee-zix/RE-GCN
Zixuan Li, Xiaolong Jin, Wei Li, Saiping Guan, Jiafeng Guo, Huawei Shen, Yuanzhuo Wang and Xueqi Cheng. Temporal 
Knowledge Graph Reasoning Based on Evolutional Representation Learning. SIGIR 2021.
"""
import sys
import timeit
import os
import sys
import os.path as osp
from pathlib import Path
import numpy as np
import torch
import random
from tqdm import tqdm
# internal imports
tgb_modules_path = osp.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(tgb_modules_path)
from tgb_modules.rrgcn import RecurrentRGCNREGCN
from tgb.utils.utils import set_random_seed, get_args_regcn, split_by_time, build_sub_graph, save_results, reformat_ts
from tgb.linkproppred.evaluate import Evaluator
from tgb.linkproppred.dataset import LinkPropPredDataset 

def test(model, history_list, test_list, num_rels, num_nodes, use_cuda, model_name, static_graph, mode, split_mode):
    """
    :param model: model used to test
    :param history_list:    all input history snap shot list, not include output label train list or valid list
    :param test_list:   test triple snap shot list
    :param num_rels:    number of relations
    :param num_nodes:   number of nodes
    :param use_cuda:
    :param model_name:
    :param mode:
    :param split_mode: 'test' or 'val' to state which negative samples to load
    :return mrr
    """
    print("Testing for mode: ", split_mode)
    if split_mode == 'test':
        timesteps_to_eval = test_timestamps_orig
    else:
        timesteps_to_eval = val_timestamps_orig

    idx = 0

    if mode == "test":
        # test mode: load parameter form file 
        if use_cuda:
            checkpoint = torch.load(model_name, map_location=torch.device(args.gpu))
        else:
            checkpoint = torch.load(model_name, map_location=torch.device('cpu'))        
        # use best stat checkpoint:
        print("Load Model name: {}. Using best epoch : {}".format(model_name, checkpoint['epoch']))  
        print("\n"+"-"*10+"start testing"+"-"*10+"\n")
        model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    input_list = [snap for snap in history_list[-args.test_history_len:]] 
    perf_list_all = []
    for time_idx, test_snap in enumerate(tqdm(test_list)):
        if time_idx == 10:
            print(time_idx)
        history_glist = [build_sub_graph(num_nodes, num_rels, g, use_cuda, args.gpu) for g in input_list]    
        test_triples_input = torch.LongTensor(test_snap).cuda() if use_cuda else torch.LongTensor(test_snap)
        timesteps_batch =timesteps_to_eval[time_idx]*np.ones(len(test_triples_input[:,0]))

        neg_samples_batch = neg_sampler.query_batch(test_triples_input[:,0], test_triples_input[:,2], 
                                timesteps_batch, edge_type=test_triples_input[:,1], split_mode=split_mode)
        pos_samples_batch = test_triples_input[:,2]

        _, perf_list = model.predict(history_glist, num_rels, static_graph, test_triples_input, use_cuda, neg_samples_batch, pos_samples_batch, 
                                    evaluator, METRIC)  # TODO:  num_rels, static_graph different!

        perf_list_all.extend(perf_list)
        # reconstruct history graph list
        input_list.pop(0)
        input_list.append(test_snap)
        idx += 1
    
    mrr = np.mean(perf_list_all) 
    return mrr



def run_experiment(args, n_hidden=None, n_layers=None, dropout=None, n_bases=None):
    # load configuration for grid search the best configuration
    if n_hidden:
        args.n_hidden = n_hidden
    if n_layers:
        args.n_layers = n_layers
    if dropout:
        args.dropout = dropout
    if n_bases:
        args.n_bases = n_bases
    mrr = 0
    # 2) set save model path
    save_model_dir = f'{osp.dirname(osp.abspath(__file__))}/saved_models/'
    if not osp.exists(save_model_dir):
        os.mkdir(save_model_dir)
        print('INFO: Create directory {}'.format(save_model_dir))
    model_name= f'{MODEL_NAME}_{DATA}_{SEED}_{args.run_nr}'
    model_state_file = save_model_dir+model_name

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()

# TODO: what to do about this part: 
    # if args.add_static_graph:
    #     static_triples = np.array(_read_triplets_as_list("../data/" + args.dataset + "/e-w-graph.txt", {}, {}, load_time=False))
    #     num_static_rels = len(np.unique(static_triples[:, 1]))
    #     num_words = len(np.unique(static_triples[:, 2]))
    #     static_triples[:, 2] = static_triples[:, 2] + num_nodes 
    #     static_node_id = torch.from_numpy(np.arange(num_words + data.num_nodes)).view(-1, 1).long().cuda(args.gpu) \
    #         if use_cuda else torch.from_numpy(np.arange(num_words + data.num_nodes)).view(-1, 1).long()
    # else:
    num_static_rels, num_words, static_triples, static_graph = 0, 0, [], None

    # create stat
    model = RecurrentRGCNREGCN(args.decoder, #TODO: this has slightly different args than CEN
                          args.encoder,
                        num_nodes,
                        int(num_rels/2),
                        num_static_rels, # DIFFERENT
                        num_words, # DIFFERENT
                        args.n_hidden,
                        args.opn,
                        sequence_len=args.train_history_len,
                        num_bases=args.n_bases,
                        num_basis=args.n_basis,
                        num_hidden_layers=args.n_layers,
                        dropout=args.dropout,
                        self_loop=args.self_loop,
                        skip_connect=args.skip_connect,
                        layer_norm=args.layer_norm,
                        input_dropout=args.input_dropout,
                        hidden_dropout=args.hidden_dropout,
                        feat_dropout=args.feat_dropout,
                        aggregation=args.aggregation, # DIFFERENT
                        weight=args.weight, # DIFFERENT
                        discount=args.discount, # DIFFERENT
                        angle=args.angle, # DIFFERENT
                        use_static=args.add_static_graph, # DIFFERENT
                        entity_prediction=args.entity_prediction, 
                        relation_prediction=args.relation_prediction,
                        use_cuda=use_cuda,
                        gpu = args.gpu,
                        analysis=args.run_analysis) # DIFFERENT

    if use_cuda:
        torch.cuda.set_device(args.gpu)
        model.cuda()

    # if args.add_static_graph: # TODO: what to do about this part: 
    #     static_graph = build_sub_graph(len(static_node_id), num_static_rels, static_triples, use_cuda, args.gpu)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    if args.test and os.path.exists(model_state_file):
        mrr = test(model, 
                    train_list+valid_list, 
                    test_list, 
                    num_rels, 
                    num_nodes, 
                    use_cuda, 
                    model_state_file, 
                    static_graph, 
                    "test", 
                    "test")
        return mrr
    elif args.test and not os.path.exists(model_state_file):
        print("--------------{} not exist, Change mode to train and generate stat for testing----------------\n".format(model_state_file))
        return 0
    else:
        print("----------------------------------------start training----------------------------------------\n")
        best_mrr = 0
        for epoch in range(args.n_epochs):

            model.train()
            losses = []
            idx = [_ for _ in range(len(train_list))]
            random.shuffle(idx)

            for train_sample_num in tqdm(idx):
                if train_sample_num == 0: continue
                output = train_list[train_sample_num:train_sample_num+1]
                if train_sample_num - args.train_history_len<0:
                    input_list = train_list[0: train_sample_num]
                else:
                    input_list = train_list[train_sample_num - args.train_history_len:
                                        train_sample_num]

                # generate history graph
                history_glist = [build_sub_graph(num_nodes, num_rels, snap, use_cuda, args.gpu) for snap in input_list]
                output = [torch.from_numpy(_).long().cuda() for _ in output] if use_cuda else [torch.from_numpy(_).long() for _ in output]
                loss_e, loss_r, loss_static = model.get_loss(history_glist, output[0], static_graph, use_cuda)
                loss = args.task_weight*loss_e + (1-args.task_weight)*loss_r + loss_static

                losses.append(loss.item())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
                optimizer.step()
                optimizer.zero_grad()

            print("Epoch {:04d} | Ave Loss: {:.4f} | Best MRR {:.4f} | Model {} "
                  .format(epoch, np.mean(losses),  best_mrr, model_name))
            
            #! checking GPU usage
            free_mem, total_mem = torch.cuda.mem_get_info()
            print ("--------------GPU memory usage-----------")
            print ("there are ", free_mem, " free memory")
            print ("there are ", total_mem, " total available memory")
            print ("there are ", total_mem - free_mem, " used memory")
            print ("--------------GPU memory usage-----------")

            # validation
            if epoch and epoch % args.evaluate_every == 0:
                mrr = test(model, train_list, 
                            valid_list, 
                            num_rels, 
                            num_nodes, 
                            use_cuda, 
                            model_state_file, 
                            static_graph, 
                            mode="train", split_mode='val')
            
                if mrr < best_mrr:
                    if epoch >= args.n_epochs:
                        break
                else:
                    best_mrr = mrr
                    torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)

        # mrr = test(model, 
        #         train_list+valid_list,
        #         test_list, 
        #         num_rels, 
        #         num_nodes, 
        #         use_cuda,
        #         model_state_file, 
        #         static_graph, 
        #         mode="test", split_mode='test')

        return best_mrr
# ==================
# ==================
# ==================

start_overall = timeit.default_timer()

# set hyperparameters
args, _ = get_args_regcn()
args.dataset = 'tkgl-yago'

SEED = args.seed  # set the random seed for consistency
set_random_seed(SEED)

DATA=args.dataset
MODEL_NAME = 'REGCN'


# load data
dataset = LinkPropPredDataset(name=DATA, root="datasets", preprocess=True)

num_rels = dataset.num_rels
num_nodes = dataset.num_nodes 
subjects = dataset.full_data["sources"]
objects= dataset.full_data["destinations"]
relations = dataset.edge_type

timestamps_orig = dataset.full_data["timestamps"]
timestamps = reformat_ts(timestamps_orig) # stepsize:1
all_quads = np.stack((subjects, relations, objects, timestamps), axis=1)

train_data = all_quads[dataset.train_mask]
val_data = all_quads[dataset.val_mask]
test_data = all_quads[dataset.test_mask]

val_timestamps_orig = list(set(timestamps_orig[dataset.val_mask])) # needed for getting the negative samples
val_timestamps_orig.sort()
test_timestamps_orig = list(set(timestamps_orig[dataset.test_mask])) # needed for getting the negative samples
test_timestamps_orig.sort()

train_list = split_by_time(train_data)
valid_list = split_by_time(val_data)
test_list = split_by_time(test_data)

# evaluation metric
METRIC = dataset.eval_metric
evaluator = Evaluator(name=DATA)
neg_sampler = dataset.negative_sampler
#load the ns samples 
dataset.load_val_ns()
dataset.load_test_ns()

val_mrr, test_mrr = 0, 0
if args.grid_search:
    print("TODO: implement hyperparameter grid search")
# single run
else:
    #TODO: differentiate between train, valid, test
    start_train = timeit.default_timer()
    if args.test == False:
        print('start training')
        val_mrr = run_experiment(args)
    start_test = timeit.default_timer()
    args.test = True
    print('start testing')
    test_mrr = run_experiment(args)


test_time = timeit.default_timer() - start_test
all_time = timeit.default_timer() - start_train
print(f"\tTest: Elapsed Time (s): {test_time: .4f}")
print(f"\Train and Test: Elapsed Time (s): {all_time: .4f}")

print(f"\tTest: {METRIC}: {test_mrr: .4f}")
print(f"\tValid: {METRIC}: {val_mrr: .4f}")

# saving the results...
results_path = f'{osp.dirname(osp.abspath(__file__))}/saved_results'
if not osp.exists(results_path):
    os.mkdir(results_path)
    print('INFO: Create directory {}'.format(results_path))
Path(results_path).mkdir(parents=True, exist_ok=True)
results_filename = f'{results_path}/{MODEL_NAME}_{DATA}_results.json'

save_results({'model': MODEL_NAME,
              'data': DATA,
              'run': args.run_nr,
              'seed': SEED,
              f'val {METRIC}': float(val_mrr),
              f'test {METRIC}': float(test_mrr),
              'test_time': test_time,
              'tot_train_val_time': all_time
              }, 
    results_filename)

sys.exit()