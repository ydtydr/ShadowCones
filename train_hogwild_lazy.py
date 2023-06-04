#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
from HTorch.optimizers import RiemannianSGD, RiemannianAdam
from HTorch import HParameter, HTensor
from data_utils.data_handler import DataHandler
from data_utils.relations import Relations
import argparse
import time
from ConeModel import UmbralCone, PeumbralCone
import torch.multiprocessing as mp
# import threading
from data_utils.cone_loader import ConeDataLoader
torch.set_default_tensor_type('torch.DoubleTensor')

def train_epoch(rank, args, model, optimizer, train_set, loss_func, epoch_freq):
    if args.optimizer in ['rsgd', 'radam']:
        model.emb.weight = HParameter(model.emb.weight.data)
        model.emb.weight.manifold = model.emb.manifold
        model.emb.weight.curvature = model.emb.curvature
    train_loader = ConeDataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=False, num_workers=0)
    if epoch_freq == 0:
        lr_epoch = args.lr * args.burnin_multiplier
    else:
        lr_epoch = args.lr
    optimizer = optimizer(params=model.parameters(), lr=lr_epoch, weight_decay=args.weight_decay)
    for epoch in range(args.eval_freq):
        for i, data in enumerate(train_loader):
            chunk_size = data.size(-1) // 2 # can be a hyper-parameter
            optimizer.zero_grad()
            energy0 = model(data[..., :chunk_size], reverse=False)
            energy1 = model(data[..., chunk_size:], reverse=True)
            loss = loss_func(energy0)  + loss_func(energy1)
            loss.backward()
            ## need lock or not? sparse case, hogwild lock-free
            optimizer.step()
            if not args.source == 'infinity':
                model.proj_away(model.emb.weight)

def evaluate(model, pos_loader, neg_loader, method = "partial"):
    tp = 0.
    with torch.no_grad():
        model.eval()
        for i, data in enumerate(pos_loader, 0):
            parents, children = data[0], data[1]
            parents_emb = model.emb(parents)
            children_emb = model.emb(children)
            if method == "partial":
                tp += model.partial(parents_emb, children_emb).sum()
            else:
                tp += (model.energy(parents_emb, children_emb)<=0.).sum()
        fn = len(pos_loader.dataset) - tp
        fp = 0.
        for i, data in enumerate(neg_loader, 0):
            parents, children = data[0], data[1]
            parents_emb = model.emb(parents)
            children_emb = model.emb(children)
            if method == "partial":
                fp += model.partial(parents_emb, children_emb).sum()
            else:
                fp += (model.energy(parents_emb, children_emb)<=0.).sum()
    precision = 100 * tp / (tp + fp + 1e-6)
    recall = 100 * tp / (tp + fn + 1e-6)  
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return {'precision': float('%.1f' % precision), 'recall': float('%.1f' % recall), 'f1': float('%.1f' % f1)}

def tune(radius_list, model, pos_loader, neg_loader, method = "partial", radius = True):
    n_r = len(radius_list)
    res_list = []
    tp_list = [0.] * n_r
    fn_list = [0.] * n_r
    fp_list = [0.] * n_r
    with torch.no_grad():
        model.eval()
        for i, data in enumerate(pos_loader, 0):
            parents, children = data[0], data[1]
            for ith_r in range(n_r):
                if radius:
                    model.radius = radius_list[ith_r]
                else:
                    model.level = radius_list[ith_r]
                parents_emb = model.emb(parents)
                children_emb = model.emb(children)
                if method == "partial":
                    tp_list[ith_r] += model.partial(parents_emb, children_emb).sum()
                else:
                    tp_list[ith_r] += (model.energy(parents_emb, children_emb)<=0.).sum()
        fn_list = [len(pos_loader.dataset) - tp_list[ith_r] for ith_r in range(n_r)]
        for i, data in enumerate(neg_loader, 0):
            parents, children = data[0], data[1]
            for ith_r in range(n_r):
                if radius:
                    model.radius = radius_list[ith_r]
                else:
                    model.level = radius_list[ith_r]
                parents_emb = model.emb(parents)
                children_emb = model.emb(children)
                if method == "partial":
                    fp_list[ith_r] += model.partial(parents_emb, children_emb).sum()
                else:
                    fp_list[ith_r] += (model.energy(parents_emb, children_emb)<=0.).sum()
    precisions = [100 * tp_list[ith_r] / (tp_list[ith_r] + fp_list[ith_r] + 1e-6) for ith_r in range(n_r)]
    recalls = [100 * tp_list[ith_r] / (tp_list[ith_r] + fn_list[ith_r] + 1e-6) for ith_r in range(n_r)]
    f1s = [2 * precisions[ith_r] * recalls[ith_r] / (precisions[ith_r] + recalls[ith_r] + 1e-6) for ith_r in range(n_r)]
    return precisions, recalls, f1s

def argument_parser():
    parser = argparse.ArgumentParser(description='Hyperbolic Umbral and Penumbral Cone')
    ############# Dataset configurations
    parser.add_argument('-dataset', type=str, choices=['mammal', 'noun', 'MCG', 'hearst'], default='mammal',
                        help='dataset (mammal | verb | noun)')
    parser.add_argument('-train_non_basic_percent', type=int, choices=[0, 1, 2, 5, 10, 25, 50, 90], 
                        default=10, help="percentage of non basic edges in training set: (0 | 10 | 25 | 50 | 90)")
    ############# model configurations
    parser.add_argument('-dim', type=int, default=2, help="dimension of hyperbolic embeddings")
    parser.add_argument('-curvature', default=-1.0, help="curvature of hyperbolic space, set to None for trainable curvature")
    parser.add_argument('-model', type=str, choices=['umbral', 'penumbral'], default='umbral',
                        help='cone model class to use: (umbral | penumbral)')
    parser.add_argument('-source', type=str, default='infinity', help='source of light: (infinity | origin | float)')
    parser.add_argument('-radius', type=float, default=0.05, help="object / source radius")
    parser.add_argument('-margin', type=float, default=0.001, help="margin in energy function, measures how far to push negatives")
    parser.add_argument('-sub_apex_dist', type=float, default=0.0001, help="sub_apex_dist for training")
    parser.add_argument('-energy_type', type=str, choices=['angle', 'distance'], default='distance',
                        help='energy type in cone model: (angle | distance)')
    parser.add_argument('-sparse', default=True, help="turn on sparse embedding")
    parser.add_argument('-num_processes', type=int, default=1, help="num_processes in hogwild")
    ############# train/eval configurations
    parser.add_argument('-optimizer', type=str, choices=['rsgd', 'radam', 'sgd', 'adam'], default='rsgd',
                        help='training optimizer: (rsgd | radam)')
    parser.add_argument('-loss_type', type=str, choices=['vanilla', 'contrastive'], default='contrastive',
                        help='loss type for training cone model: (vanilla | contrastive)')
    parser.add_argument('-lr', type=float, default=0.01, help="learning rate for optimization")
    parser.add_argument('-burnin_multiplier', type=float, default=0.01, help='burnin_multiplier')
    parser.add_argument('-weight_decay', type=float, default=0.0, help="weight_decay for optimization")
    parser.add_argument('-epoch', type=int, default=300, help="training epochs")
    parser.add_argument('-burnin_epoch', type=int, default=20, help="burnin epochs at beginning for better initialization")
    parser.add_argument('-neg_size', type=int, default=10, help="negative sampling size")
    parser.add_argument('-batch_size', type=int, default=16, help="training batch_size")
    parser.add_argument('-neg_sampl_strategy', type=str, choices=['true_neg', 'all'], default='true_neg',
                        help='all or non-connected nodes used for negative sampling')
    parser.add_argument('-where_not_to_sample', type=str, choices=['ancestors', 'children', 'both'], default='children',
                        help='where_not_to_sample')
    parser.add_argument('-eval_method', type=str, choices=['partial', 'energy'], default='partial',
                        help='eval_method method: (partial | energy)')
    parser.add_argument('-eval_freq', type=int, default=20, help="eval_freq during training")
    ############# debug configurations
    parser.add_argument('-seed', type=int, default=43, help="random seed for reproducing results")
    parser.add_argument('-debug', type=int, default=0, help="debug mode")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = argument_parser()
    args.debug = False if args.debug==0 else True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.dataset in ['MCG', 'hearst']:
        args.data_dir = f'data_utils/data/{args.dataset}/{args.dataset}_closure.tsv'
    else:
        args.data_dir = f'data_utils/data/maxn/{args.dataset}_closure.tsv'
    # ### Load data
#     basic_edge_filepath = args.data_dir + '.train_0percent'
#     full_transitive_filepath = args.data_dir + '.full_transitive'
#     full_neg_filepath = args.data_dir + '.full_neg'
    train_data_dir = args.data_dir + f".train_{args.train_non_basic_percent}percent"
    val_pos_data_dir = args.data_dir + ".valid"
    val_neg_data_dir = args.data_dir + ".valid_neg"
    test_pos_data_dir = args.data_dir + ".test"
    test_neg_data_dir = args.data_dir + ".test_neg"
    train_data = Relations(train_data_dir, reverse=False)
    val_data = (Relations(val_pos_data_dir, reverse=False), Relations(val_neg_data_dir, reverse=False))
    test_data = (Relations(test_pos_data_dir, reverse=False), Relations(test_neg_data_dir, reverse=False))
    data_handler = DataHandler(train_data=train_data, val_data=val_data, test_data=test_data, num_negative=args.neg_size, 
                               batch_size=args.batch_size, neg_sampl_strategy=args.neg_sampl_strategy,
                               where_not_to_sample=args.where_not_to_sample, num_workers=0, num_processes=args.num_processes
                              )
    ###########################
    ### Initialize model
    ###########################
    size = len(data_handler.indices_set)
    args.curvature = torch.nn.Parameter(torch.tensor(-1.0)) if args.curvature==None else float(args.curvature)
    if args.debug:
        print("number of nodes", size)
    if args.model == 'umbral':
        hyp_cone = UmbralCone(source = args.source, radius = args.radius, 
                              size = size, dim = args.dim, sparse=args.sparse, curvature = args.curvature, 
                              margin = args.margin, sub_apex_dist=args.sub_apex_dist, energy_type=args.energy_type)
        hyp_cone_eval = UmbralCone(source = args.source, radius = args.radius+0.05, 
                              size = size, dim = args.dim, sparse=args.sparse, curvature = args.curvature, 
                              margin = args.margin, sub_apex_dist=args.sub_apex_dist, energy_type=args.energy_type)
    else:
        hyp_cone = PeumbralCone(source = args.source, radius = args.radius, 
                              size = size, dim = args.dim, sparse=args.sparse, curvature = args.curvature, 
                              margin = args.margin, sub_apex_dist=args.sub_apex_dist, energy_type=args.energy_type)
        hyp_cone_eval = PeumbralCone(source = args.source, radius = args.radius, 
                              size = size, dim = args.dim, sparse=args.sparse, curvature = args.curvature, 
                              margin = args.margin, sub_apex_dist=args.sub_apex_dist, energy_type=args.energy_type)
    optimizer = {'sgd': RiemannianSGD, 'adam': RiemannianAdam, 'rsgd': RiemannianSGD, 'radam': RiemannianAdam}[args.optimizer]
    ###########################
    ### Training hyper-parameters
    ###########################
    data_handler.prepare_train_data()
    val_pos_loader, val_neg_loader, test_pos_loader, test_neg_loader = data_handler.prepare_val_test_loader()
    hyp_cone.share_memory() # gradients are allocated lazily, so they are not shared here
    mp.set_start_method('spawn', force=True)
    loss_func = {'vanilla': hyp_cone.loss, 'contrastive': hyp_cone.loss_cross}[args.loss_type]
    best_metric = {'precision': -1.0, 'recall': -1.0, 'f1': -1.0}
    epoch_best = -1.0
    for epoch_freq in range(args.epoch//args.eval_freq):
        time1 = time.time()
        trainset_mp = data_handler.prepare_trainset_mp()
        hyp_cone.train()
        processes = []
        for rank in range(args.num_processes):
            p = mp.Process(target=train_epoch, args=(rank, args, hyp_cone, optimizer, trainset_mp[rank], loss_func, epoch_freq))
#             p = threading.Thread(target=train_epoch, args=(rank, args, hyp_cone, optimizer, trainset_mp[rank], loss_func, epoch_freq))
            # We first train the model across `num_processes` processes
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        if args.debug:
            print(f"epoch_freq {epoch_freq} training time: {'%.2f' % (time.time() - time1)}")
        time2 = time.time()
        ## can we test training accuracy? particularly for basic edges
        val_triplet = evaluate(hyp_cone, val_pos_loader, val_neg_loader, method = args.eval_method)
        if val_triplet['f1'] > best_metric['f1']:
            best_metric = val_triplet
            epoch_freq_best = epoch_freq
            hyp_cone_eval.emb.weight.data.copy_(hyp_cone.emb.weight.data)
        if args.debug:
            print(f"epoch_freq {epoch_freq} evaluate time: {'%.2f' % (time.time() - time2)}, val: {best_metric}")
    best_metric['epoch_freq_best'] = epoch_freq_best
    best_metric['radius'] = args.radius
    print(f"best val: {best_metric}")    
    ###########################
    ### Tuning radius
    ###########################
    time1 = time.time()
    if args.model == 'penumbral' and not args.source=='origin':
        height_list = np.array([2.0, 5.0, 10.0, 20.0, 21.0, 25.0])
        val_precision, val_recall, val_f1 = tune(height_list, hyp_cone_eval, val_pos_loader, val_neg_loader,
                                                 method=args.eval_method, radius=False)
        val_pos = val_f1.index(max(val_f1))
        print({'tuned val: height': ('%.1f' % height_list[val_pos]), 'precision': ('%.1f' % val_precision[val_pos].item()), 'recall': ('%.1f' % val_recall[val_pos].item()), 'f1': ('%.1f' % val_f1[val_pos].item())})
        hyp_cone_eval.level = height_list[val_pos]
        test_triplet = evaluate(hyp_cone_eval, test_pos_loader, test_neg_loader, method = args.eval_method)
    else:
        radius_list = np.array([0.01, 0.05, 0.06, 0.1, 0.2, 0.3])
        val_precision, val_recall, val_f1 = tune(radius_list, hyp_cone_eval, val_pos_loader, val_neg_loader,
                                                 method=args.eval_method, radius=True)
        val_pos = val_f1.index(max(val_f1))
        print({'tuned val: radius': ('%.1f' % radius_list[val_pos]), 'precision': ('%.1f' % val_precision[val_pos].item()), 'recall': ('%.1f' % val_recall[val_pos].item()), 'f1': ('%.1f' % val_f1[val_pos].item())})
        hyp_cone_eval.radius = radius_list[val_pos]
        test_triplet = evaluate(hyp_cone_eval, test_pos_loader, test_neg_loader, method = args.eval_method)
    print({'test precision': test_triplet['precision'], 'recall': test_triplet['recall'], 'f1': test_triplet['f1']})
    time2 = time.time()
    if args.debug:
        print(f"tuning time, val: {'%.2f' % (time2 - time1)}")
