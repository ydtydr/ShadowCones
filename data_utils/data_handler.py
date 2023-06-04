#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as np
from numpy import random as np_random
import torch
from collections import defaultdict
from gensim import utils
from data_utils.GensimBasicKeyedVectors import Vocab, KeyedVectorsBase
from data_utils.cone_loader import ConeTrainSet, ConeTestSet, ConeDataLoader

class DataHandler(utils.SaveLoad):
    def __init__(self,
                 train_data,
                 val_data,
                 test_data,
                 dim=50,  # Number of dimensions of the trained model.
                 seed=0,
                 logger=None,
                 KeyedVectorsClass=KeyedVectorsBase,
                 num_workers = 1,
                 num_processes = 1,
                 num_negative = 16,  # Number of negative samples to use.
                 batch_size = 16, # batchsize
                 ### How to sample negatives for an edge (u,v)
                 neg_sampl_strategy='true_neg',  # 'all' (all nodes used for negative sampling) or 'true_neg' (only not connected nodes)
                                                 # 'all_non_leaves' or 'true_neg_non_leaves'
                 where_not_to_sample='ancestors',  # both or ancestors or children. Has no effect if neg_sampl_strategy = 'all'.
                 neg_edges_attach='child',  # How to form negative edges: 'parent' (u,v') or 'child' (u', v) or 'both'
                 always_v_in_neg=True,  # always include the true edge (u,v) as negative.
                 neg_sampling_power=0.0,  # 0 for uniform, 1 for unigram, 0.75 for word2vec
                 ):
        """Initialize and train a DAG embedding model from an iterable of relations.

        Parameters
        ----------
        train_data : iterable of (str, str)
            Iterable of relations, e.g. a list of tuples, or a Relations instance streaming from a file.
            Note that the relations are treated as ordered pairs, i.e. a relation (a, b) does not imply the
            opposite relation (b, a). In case the relations are symmetric, the data should contain both relations
            (a, b) and (b, a).
        """

        self.logger = logger

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.kv = KeyedVectorsClass()

        self.dim = dim
        self.only_leaves_updated = False

        self.where_not_to_sample = where_not_to_sample
        assert self.where_not_to_sample in ['both', 'ancestors', 'children']
        self.neg_edges_attach = neg_edges_attach
        assert self.neg_edges_attach in ['parent', 'child', 'both']

        self.always_v_in_neg = always_v_in_neg
        
        self.num_workers = num_workers
        self.num_processes = num_processes
        self.batch_size = batch_size
        self.num_negative = num_negative
        self._neg_sampling_power = neg_sampling_power
        assert self._neg_sampling_power >= 0 and self._neg_sampling_power <= 2
        self.neg_sampl_strategy = neg_sampl_strategy
        assert self.neg_sampl_strategy in ['all', 'true_neg', 'all_non_leaves', 'true_neg_non_leaves']

        self._np_rand = np_random.RandomState(seed)
        self._load_relations_and_indexes()


    def _load_relations_and_indexes(self):
        """Load relations from the train data and build vocab."""
        self.kv.vocab = {} # word -> (index, count)
        self.kv.index2word = [] # index -> word
        self.all_relations = []  # List of all relation pairs
        self.adjacent_nodes = defaultdict(set)  # Mapping from node index to its neighboring node indices
        if '_non_leaves' in self.neg_sampl_strategy:
            self.non_leaves_indices_set = set()
            self.non_leaves_adjacent_nodes = defaultdict(set)

        # self.logger.info("Loading relations from train data..")
        for relation in self.train_data:
            if len(relation) != 2:
                raise ValueError('Relation pair "%s" should have exactly two items' % repr(relation))
            # Ignore self-edges
            assert relation[0] != relation[1]

            for item in relation:
                if item not in self.kv.vocab:
                    self.kv.vocab[item] = Vocab(count=1, index=len(self.kv.index2word))
                    self.kv.index2word.append(item)

            # Like in https://github.com/facebookresearch/poincare-embeddings
            self.kv.vocab[relation[0]].count += 1

            node_1, node_2 = relation # Edge direction : node1 -> node2, swapped in the csv file, but correctly read in PoincareRelations.
            node_1_index, node_2_index = self.kv.vocab[node_1].index, self.kv.vocab[node_2].index

            if self.where_not_to_sample in ['both', 'children']:
                self.adjacent_nodes[node_1_index].add(node_2_index)
            if self.where_not_to_sample in ['both', 'ancestors']:
                self.adjacent_nodes[node_2_index].add(node_1_index)

            if '_non_leaves' in self.neg_sampl_strategy:
                self.non_leaves_indices_set.add(node_1_index)

            self.all_relations.append((node_1_index, node_2_index))

        for node_idx in range(len(self.kv.index2word)):
            self.adjacent_nodes[node_idx].add(node_idx) # Do not sample current node

        if '_non_leaves' in self.neg_sampl_strategy:
            for node_idx in range(len(self.kv.index2word)):
                self.non_leaves_adjacent_nodes[node_idx].add(node_idx)
                for adj_node_idx in self.adjacent_nodes[node_idx]:
                    if adj_node_idx in self.non_leaves_indices_set:
                        self.non_leaves_adjacent_nodes[node_idx].add(adj_node_idx)

        # self.logger.info("Loaded %d relations from train data, %d nodes",
        #             len(self.all_relations), len(self.kv.vocab))

        self.indices_set = set((range(len(self.kv.index2word))))  # Set of all node indices

        freq_array = np.array([self.kv.vocab[self.kv.index2word[i]].count
                                for i in range(len(self.kv.index2word))], dtype=np.float64)
        unigrams_at_power_array = np.power(freq_array, self._neg_sampling_power)

        self._node_probabilities = unigrams_at_power_array / unigrams_at_power_array.sum()
        self._node_probabilities_cumsum = np.cumsum(self._node_probabilities)

        if '_non_leaves' in self.neg_sampl_strategy:
            self.non_leaves_indices_array = np.array(list(self.non_leaves_indices_set))
            unigrams_at_power_array_non_leaves = unigrams_at_power_array[self.non_leaves_indices_array]
            self._node_probabilities_non_leaves = unigrams_at_power_array_non_leaves / \
                                                  unigrams_at_power_array_non_leaves.sum()
            self._node_probabilities_cumsum_non_leaves = np.cumsum(self._node_probabilities_non_leaves)

    def _sample_negatives(self, node_index, connected_node=None):
        """Return a sample of negatives for the given node.

        Parameters
        ----------
        node_index : int
            Index of the positive node for which negative samples are to be returned.
            connected_node: the one positive example for contrastive learning

        Returns
        -------
        numpy.array
            Array of shape (self.num_negative,) containing indices of negative nodes for the given node index.

        """
        k = self.num_negative  # num negatives

        if self.neg_sampl_strategy == 'all' or len(self.adjacent_nodes[node_index]) == len(self.indices_set): # root node
            uniform_0_1_numbers = self._np_rand.random_sample(self.num_negative)
            negs = list(np.searchsorted(self._node_probabilities_cumsum, uniform_0_1_numbers))
        elif self.neg_sampl_strategy == 'all_non_leaves':
            uniform_0_1_numbers = self._np_rand.random_sample(self.num_negative)
            negs = list(np.searchsorted(self._node_probabilities_cumsum_non_leaves, uniform_0_1_numbers))
            negs = self.non_leaves_indices_array[negs]

        elif self.neg_sampl_strategy == 'true_neg':
            n = len(self.indices_set)
            a = len(self.adjacent_nodes[node_index])

            gamma = float(n) / (n - a)

            if gamma > n / (max(k,1) * math.log(n)):
                # Very expensive branch: O(n + k*log(n)). Should be avoided when possible.
                
                valid_negatives = np.array(list(self.indices_set - self.adjacent_nodes[node_index])) # O(n). Includes node_index.
                valid_node_probs = self._node_probabilities[valid_negatives]
                valid_node_probs = valid_node_probs / valid_node_probs.sum()
                valid_node_cumsum = np.cumsum(valid_node_probs) # O(n)
                uniform_0_1_numbers = self._np_rand.random_sample(k)
                valid_negative_indices = np.searchsorted(valid_node_cumsum, uniform_0_1_numbers) # O(k * log n)
                negs = list(valid_negatives[valid_negative_indices])
                
            else:
                # Less expensive branch: O(n / (n-a) * k * log n)
                # we sample gamma * k negatives and hope to find at least k true negatives
                negatives = []
                remain_to_sample = k
                while remain_to_sample > 0:
                    num_to_sample = int(gamma * remain_to_sample)
                    uniform_0_1_numbers = self._np_rand.random_sample(num_to_sample)
                    new_potential_negatives =\
                        np.searchsorted(self._node_probabilities_cumsum, uniform_0_1_numbers)  # O(gamma * k * log n)

                    # time complexity O(gamma * k),
                    # but len(new_good_negatives) is in expectation (1 - a/n) * len(new_potential_candidates)
                    new_good_negatives = [x for x in new_potential_negatives
                                          if x not in self.adjacent_nodes[node_index]]
                    num_new_good_negatives =  min(len(new_good_negatives), remain_to_sample)
                    negatives.extend(new_good_negatives[0 : num_new_good_negatives])
                    remain_to_sample -= num_new_good_negatives
                negs = negatives

        elif self.neg_sampl_strategy == 'true_neg_non_leaves':
            n = len(self.non_leaves_indices_set)
            a = len(self.non_leaves_adjacent_nodes[node_index])

            gamma = float(n) / (n - a)

            if gamma > n / (max(k,1) * math.log(n)):
                # Very expensive branch: O(n + k*log(n)). Should be avoided when possible.

                valid_negatives = np.array(list(self.non_leaves_indices_set - self.non_leaves_adjacent_nodes[node_index])) # O(n). Includes node_index.
                valid_node_probs = self._node_probabilities_non_leaves[valid_negatives]
                valid_node_probs = valid_node_probs / valid_node_probs.sum()
                valid_node_cumsum = np.cumsum(valid_node_probs) # O(n)
                uniform_0_1_numbers = self._np_rand.random_sample(k)
                valid_negative_indices = np.searchsorted(valid_node_cumsum, uniform_0_1_numbers) # O(k * log n)
                negs = list(valid_negatives[valid_negative_indices])
            else:
                # Less expensive branch: O(n / (n-a) * k * log n)
                # we sample gamma * k negatives and hope to find at least k true negatives
                negatives = []
                remain_to_sample = k
                while remain_to_sample > 0:
                    num_to_sample = int(gamma * remain_to_sample)
                    uniform_0_1_numbers = self._np_rand.random_sample(num_to_sample)
                    new_potential_negatives =\
                        np.searchsorted(self._node_probabilities_cumsum_non_leaves, uniform_0_1_numbers)  # O(gamma * k * log n)
                    new_potential_negatives = self.non_leaves_indices_array[new_potential_negatives]

                    # time complexity O(gamma * k),
                    # but len(new_good_negatives) is in expectation (1 - a/n) * len(new_potential_candidates)
                    new_good_negatives = [x for x in new_potential_negatives
                                          if x not in self.non_leaves_adjacent_nodes[node_index]]
                    num_new_good_negatives =  min(len(new_good_negatives), remain_to_sample)
                    negatives.extend(new_good_negatives[0 : num_new_good_negatives])
                    remain_to_sample -= num_new_good_negatives
                negs = negatives

        # Should we always include 'v' as negative in all batches ?
        if self.always_v_in_neg:
            negs[0] = connected_node
        return negs
    
    def generate_pair(self, key):
        """key in ['val', 'test']"""
        if key == 'val':
            pos_data = self.val_data[0]
            neg_data = self.val_data[1]
        elif key == 'test':
            pos_data = self.test_data[0]
            neg_data = self.test_data[1]
        else:
            raise NotImplemented
        pos_relations_parents = []
        pos_relations_children = []
        for node_parent, node_child in pos_data:
            assert node_parent != node_child
            node_parent_idx = self.kv.vocab[node_parent].index
            node_child_idx = self.kv.vocab[node_child].index
            pos_relations_parents.append(node_parent_idx)
            pos_relations_children.append(node_child_idx)
        neg_relations_parents = []
        neg_relations_children = []
        for node_parent, node_child in neg_data:
            assert node_parent != node_child
            node_parent_idx = self.kv.vocab[node_parent].index
            node_child_idx = self.kv.vocab[node_child].index
            neg_relations_parents.append(node_parent_idx)
            neg_relations_children.append(node_child_idx)
        pos_relations_parents = torch.tensor(pos_relations_parents)
        pos_relations_children = torch.tensor(pos_relations_children)
        neg_relations_parents = torch.tensor(neg_relations_parents)
        neg_relations_children = torch.tensor(neg_relations_children)
        return pos_relations_parents, pos_relations_children, neg_relations_parents, neg_relations_children
    
    def prepare_val_test_loader(self):
        pos_relations_parents, pos_relations_children, neg_relations_parents, neg_relations_children = self.generate_pair(key='val')
        val_pos = ConeTestSet(pos_relations_parents, pos_relations_children)
        val_neg = ConeTestSet(neg_relations_parents, neg_relations_children)
        pos_relations_parents, pos_relations_children, neg_relations_parents, neg_relations_children = self.generate_pair(key='test')
        test_pos = ConeTestSet(pos_relations_parents, pos_relations_children)
        test_neg = ConeTestSet(neg_relations_parents, neg_relations_children)
        val_pos_loader = ConeDataLoader(val_pos, batch_size=self.batch_size, shuffle=False, pin_memory=False, num_workers=self.num_workers)
        val_neg_loader = ConeDataLoader(val_neg, batch_size=self.batch_size, shuffle=False, pin_memory=False, num_workers=self.num_workers)
        test_pos_loader = ConeDataLoader(test_pos, batch_size=self.batch_size, shuffle=False, pin_memory=False, num_workers=self.num_workers)
        test_neg_loader = ConeDataLoader(test_neg, batch_size=self.batch_size, shuffle=False, pin_memory=False, num_workers=self.num_workers)
        return val_pos_loader, val_neg_loader, test_pos_loader, test_neg_loader
    
    def prepare_train_data(self):
        """
        still needs to implement pos and neg loader for evaulate training accuracy, and create a new function to test 
        performance on basic edges
        """
        relation_indices = list(range(len(self.all_relations)))
        num_batches = max(1, len(relation_indices) // self.batch_size)
        all_negatives_batch = []
        for i in range(num_batches):
            start_ind = i*self.batch_size
            end_ind = min((i+1)*self.batch_size, len(relation_indices))
            batch_indices = relation_indices[start_ind:end_ind]
            relations_batch = [self.all_relations[idx] for idx in batch_indices]
            # training format: [relation[0]=0, relation[1]=1, negatives ...]
            negatives_batch = [[relation[0]] + self._sample_negatives(relation[0], relation[1]) 
                               + [relation[1]] + self._sample_negatives(relation[1], relation[0]) for relation in relations_batch]
            all_negatives_batch.extend(negatives_batch)
        self.all_negatives_batch = torch.tensor(all_negatives_batch).squeeze()
        
    def prepare_train_loader(self):
        """
        still needs to implement pos and neg loader for evaulate training accuracy, and create a new function to test 
        performance on basic edges
        """
        train_dataset = ConeTrainSet(self.all_negatives_batch)
        trainloader = ConeDataLoader(train_dataset, 
                                     batch_size=self.batch_size, 
                                     shuffle=True,
                                     pin_memory=False,
                                     num_workers=self.num_workers)
        return trainloader
    
    def prepare_trainset_mp(self):
        num_rows = self.all_negatives_batch.size(0)
        # Generate a random permutation of indices
        perm = torch.randperm(num_rows)
        # Shuffle the tensor along the first dimension
        all_negatives_batch = self.all_negatives_batch[perm]
        rank_size = num_rows // self.num_processes
        trainset_mp = []
        for rank in range(self.num_processes):
            start = rank_size * rank
            if (rank+1) == self.num_processes:
                end = num_rows
            else:
                end = rank_size * (rank + 1)
            rank_negatives_batch = all_negatives_batch[start:end]
            trainset_mp.append(ConeTrainSet(rank_negatives_batch))
        return trainset_mp
        
    def custom_split(self, basic_edge_filepath, full_transitive_filepath, full_neg_filepath, ratio_list):
        """
        you can create train, val, test accordingly to a ratio using
        basic_edge_filepath, full_transitive_filepath, full_neg_filepath;
        for example, train should include all basic edges, augmented with x% non-basic edges
        val and test include a ratio of non-basic edges, default 5% for val and test
        all set are augmented with negative edges with #negative samples, should contain (u, v') and (u', v) samples
        inputs:
        basic_edge_filepath, full_transitive_filepath, full_neg_filepath and ratio_list
        ratio_list is a [x1, x2, x3] list indicate the split percentage of non-basic edges over train/val/test, for example
        ratio_list = [0.90, 0.05, 0.05], sum(ratio_list) <= 1
        returns:
        train_data, (val_pos, val_neg), (test_pos, test_neg)
        check and combine .split_wordnet_data.py file
        """
        raise NotImplemented
    
    def contrastive_batch(self, batch_size):
        """
        randomly sample contrastive_learning data of size batch_size * (2 + negative_size)
        note that: # How to form negative edges: 'parent' (u,v') or 'child' (u', v) or 'both'
        now for debugging only, use prepare_data
        """
        relation_indices = list(range(len(self.all_relations)))
        self._np_rand.shuffle(relation_indices)
        num_batches = len(relation_indices) // batch_size
        for i in range(num_batches):
            batch_indices = relation_indices[i*batch_size:(i+1)*batch_size]
            relations_batch = [self.all_relations[idx] for idx in batch_indices]
            # training format: [relation[0]=0, relation[1]=1, negatives ...]
            negatives_batch = [[relation[0]] + self._sample_negatives(relation[0], relation[1]) for relation in relations_batch]
            yield torch.tensor(np.array(negatives_batch).squeeze())
