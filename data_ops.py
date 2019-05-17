"""
data_ops

This file contains access to data and methods for assembly of data.

- Rakshit Agrawal, 2018
"""
import argparse
import os
import random
from collections import Counter, OrderedDict, defaultdict

import networkx as nx
import numpy as np
import pandas as pd
import scipy.io as sio
import tensorflow as tf

from log_control import *
from utils import Utils

kl = tf.keras.layers

DATA_DIR_DICT = {
    'seal': Utils.data_file('public_seal'),
    'snap': Utils.data_file('public_snap'),
    'snap_csv': Utils.data_file('public_snap'),
    'wsn': Utils.data_file('public_wsn')
}

DATA_PARSE = {
    'seal': 'mat',
    'snap': 'txt',
    'snap_csv': 'csv',
    'wsn': 'csv_wsn'
}


class DataHandling(object):

    def __init__(self,
                 dataset_name,
                 dataset_type='seal',
                 ordered_args=None):
        self.dataset_name = dataset_name
        self.dataset_type = dataset_type
        self.ordered_args = ordered_args

        self.g = None
        self.g_features = None

        if DATA_PARSE[self.dataset_type] == 'mat':
            self.adj_mat, self.g_features = self._load_adj_from_mat_file(dataset_name)
            self.g = nx.from_scipy_sparse_matrix(self.adj_mat, )
        elif DATA_PARSE[self.dataset_type] == 'txt':
            self.g = self._load_nxg_from_txt_file(dataset_name)
        elif DATA_PARSE[self.dataset_type] == 'csv':
            self.g = self._load_nxg_from_csv_edgelist(dataset_name)
        elif DATA_PARSE[self.dataset_type] == 'csv_wsn':
            self.g = self._load_nxg_from_csv_wsn(dataset_name)

        self.g = nx.convert_node_labels_to_integers(self.g, first_label=1)
        self.node_ref = {i: i for i in self.g.nodes}
        self.node_len = len(self.g.nodes)

        self.learning_dataset = None
        self.is_data_generated = False

        self.additional_embeddings = []

        if self.ordered_args.get('use_node_features', True):
            if self.g_features is not None:
                node_feature_layer = self._get_node_features_from_scipy_sparse_matrix(self.g_features)
                self.additional_embeddings.append(node_feature_layer)

        if len(self.additional_embeddings):
            logi("Additional embeddings will be used")

        self.baselines_only = False
        if self.ordered_args['for_baselines']:
            self.baselines_only = True

        self.metadata_file = None
        if self.ordered_args['visualize']:
            # Generate metadata file to visualize the data
            FIELDS = {'degree': nx.degree,
                      'centrality': nx.degree_centrality,
                      'triangles': nx.triangles}

            logi("Generating metadata for visualization using fields {}".format(FIELDS))

            field_dicts = {k: dict(v(self.g)) for k, v in FIELDS.items()}
            metadata_dict = defaultdict(dict)

            for field, fd in field_dicts.items():
                for k, v in fd.items():
                    metadata_dict[field][k] = v

            df = pd.DataFrame(metadata_dict)
            logi("Data frame shape: {}".format(df.shape))

            meta_filename = os.path.join(DATA_DIR_DICT[self.dataset_type], "meta_{}.tsv".format(self.dataset_name))

            df.to_csv(meta_filename, sep='\t', index_label='Node')
            logi("Metadata saved to {}".format(meta_filename))
            self.metadata_file = meta_filename

    def _load_adj_from_mat_file(self, dataset_name):
        data_file = os.path.join(DATA_DIR_DICT[self.dataset_type], '%s.mat' % dataset_name)
        file_data = sio.loadmat(data_file)

        if 'group' in file_data.keys():
            return file_data['net'], file_data['group']
        return file_data['net'], None

    def _load_nxg_from_csv_edgelist(self, dataset_name):
        data_file = os.path.join(DATA_DIR_DICT[self.dataset_type], '%s.csv' % dataset_name)
        graph = nx.Graph()
        data = pd.read_csv(data_file, header=None, index_col=None)
        for idx, row in data.iterrows():
            from_node, to_node = row[0], row[1]
            graph.add_edge(int(from_node), int(to_node))

        return graph

    def _load_nxg_from_csv_wsn(self, dataset_name):
        data_file = os.path.join(DATA_DIR_DICT[self.dataset_type], '%s.csv' % dataset_name)
        graph = nx.DiGraph()
        data = pd.read_csv(data_file, header=None, index_col=None, names=['from', 'to', 'rating'])
        for idx, row in data.iterrows():
            from_node, to_node, rating = row['from'], row['to'], row['rating']
            graph.add_edge(int(from_node), int(to_node), rating=rating)

        return graph

    def _load_nxg_from_txt_file(self, dataset_name):
        data_file = os.path.join(DATA_DIR_DICT[self.dataset_type], '%s.txt' % dataset_name)

        graph = nx.Graph()
        with open(data_file, 'r') as inp:
            for line in inp.readlines():
                if line.startswith("#"):
                    continue

                # Otherwise split by tab and get the from and to_nodes
                from_node, to_node = line.strip().split('\t')
                graph.add_edge(int(from_node), int(to_node))

        return graph

    def pad_data(self, data, vec_size, max_len):
        assert isinstance(data, np.ndarray)

        ret_vec = np.zeros((max_len, vec_size))
        if len(data):
            ret_vec[:min(len(data), max_len)] = data[:min(len(data), max_len)]
        return ret_vec

    def _nodes_to_ref(self, l):
        return map(lambda x: self.node_ref[x], l)

    def create_labeled_link_dataset(self, path_lengths, max_paths, shuffle=True, sample_size=None):
        """ Create the dataset. For each path length, create a separate list"""

        pos_data = set(self.g.edges)
        neg_data = set()
        for start in self.g.nodes:
            targets = random.sample(self.g.nodes, 10)
            targets = [i for i in targets if not i in nx.neighbors(self.g, start)]
            for i in targets:
                neg_data.add((start, i))

        logi("Raw\t Pos data: %d\t Neg data:%d" % (len(pos_data), len(neg_data)))
        if len(pos_data) < len(neg_data):
            neg_data = random.sample(neg_data, len(pos_data))
        logi("Balanced\t Pos data: %d\t Neg data:%d" % (len(pos_data), len(neg_data)))

        all_data = list(pos_data.union(neg_data))
        logi("Size of data: %d" % len(all_data))
        if shuffle:
            random.shuffle(all_data)

        node_pairs = []
        path_lists = {i: [] for i in path_lengths}
        labels = []

        if sample_size is not None:
            data = random.sample(all_data, sample_size) if sample_size < len(all_data) else all_data
        else:
            data = all_data

        max_path_len = max(path_lengths)
        for itr, (u, v) in enumerate(data):
            caches = {i: [] for i in path_lengths}
            label = 0
            try:
                for path in nx.shortest_simple_paths(self.g, source=u, target=v):
                    path_len = len(path)
                    if path_len > max_path_len:
                        break

                    path = self._nodes_to_ref(path)

                    if path_len == 1:
                        continue

                    if path_len == 2:
                        label = 1

                    else:
                        if path_len in caches.keys():
                            if len(caches[path_len]) < max_paths[path_len]:
                                caches[path_len].append(path)
                            else:
                                if path_len == max_path_len:
                                    break
                        else:
                            continue

            except nx.NetworkXNoPath as e:
                pass

            node_pairs.append(self._nodes_to_ref([u, v]))
            labels.append(label)
            assert len(node_pairs) == len(labels)
            for path_len in path_lengths:
                caches[path_len] = self.pad_data(np.asarray(caches[path_len]),
                                                 path_len, max_paths[path_len])
                path_lists[path_len].append(np.asarray(caches[path_len]))
                assert len(path_lists[path_len]) == len(labels)

            if np.random.random() > 0.999:
                logi("Label Distribution at step %d: %s" % (itr, Counter(labels)))

        logi("Final Label Distribution: %s" % (Counter(labels)))

        self.is_data_generated = True
        return node_pairs, path_lists, labels

    def create_labeled_wsn_dataset(self, path_lengths, max_paths, shuffle=True, sample_size=None):
        """ Create the dataset. For each path length, create a separate list

            For WSN, we do not need to collect any data from non-edge pairs.
            And since it is regression, there is no need for balance as well.
            So the rating of each edge is the label, and each edge is an entry.
            And from all the edges, we simply perform the split as needed.

        """
        all_data = list(self.g.edges)
        logi("Size of data: %d" % len(all_data))
        if shuffle:
            random.shuffle(all_data)

        node_pairs = []
        path_lists = {i: [] for i in path_lengths}
        # path_lists = defaultdict(list)
        labels = []

        if sample_size is not None:
            data = random.sample(all_data, sample_size) if sample_size < len(all_data) else all_data
        else:
            data = all_data

        max_path_len = max(path_lengths)
        for itr, (u, v) in enumerate(data):
            caches = {i: [] for i in path_lengths}
            # caches = defaultdict(list)
            label = 0  # Means no edge between the nodes
            try:
                for path in nx.shortest_simple_paths(self.g, source=u, target=v):
                    path_len = len(path)
                    if path_len > max_path_len:
                        break

                    path = self._nodes_to_ref(path)

                    if path_len == 1:
                        continue

                    if path_len == 2:
                        label = self.g[u][v]['rating']

                    else:
                        if path_len in caches.keys():
                            if len(caches[path_len]) < max_paths[path_len]:
                                caches[path_len].append(np.asarray(path, dtype=np.int32))
                            else:
                                if path_len == max_path_len:
                                    break
                        else:
                            continue

            except nx.NetworkXNoPath as e:
                pass

            node_pairs.append(self._nodes_to_ref([u, v]))
            labels.append(label)
            assert len(node_pairs) == len(labels)
            for path_len in path_lengths:
                caches[path_len] = self.pad_data(np.asarray(caches[path_len], dtype=np.int32),
                                                 path_len, max_paths[path_len])
                path_lists[path_len].append(np.asarray(caches[path_len], dtype=np.int32))
                assert len(path_lists[path_len]) == len(labels)

            if np.random.random() > 0.999:
                logi("Label Distribution at step %d: %s" % (itr, Counter(labels)))

        # logi("Final Label Distribution: \n", (np.histogram(labels, bins=20)))

        self.is_data_generated = True
        return node_pairs, path_lists, labels

    def create_labeled_link_dataset_for_baselines(self, path_lengths, max_paths, shuffle=True, sample_size=None):
        """ Create the dataset. For each path length, create a separate list"""

        pos_data = set(self.g.edges)
        neg_data = set()
        for start in self.g.nodes:
            targets = random.sample(self.g.nodes, 10)
            targets = [i for i in targets if not i in nx.neighbors(self.g, start)]
            for i in targets:
                neg_data.add((start, i))

        logi("Raw\t Pos data: %d\t Neg data:%d" % (len(pos_data), len(neg_data)))
        if len(pos_data) < len(neg_data):
            neg_data = random.sample(neg_data, len(pos_data))
        logi("Balanced\t Pos data: %d\t Neg data:%d" % (len(pos_data), len(neg_data)))

        all_data = list(pos_data.union(neg_data))
        logi("Size of data: %d" % len(all_data))
        if shuffle:
            random.shuffle(all_data)

        node_pairs = []
        path_lists = {}
        labels = []

        if sample_size is not None:
            data = random.sample(all_data, sample_size) if sample_size < len(all_data) else all_data
        else:
            data = all_data

        for itr, (u, v) in enumerate(data):
            caches = {i: [] for i in path_lengths}
            label = 1 if (u, v) in self.g.edges else 0

            node_pairs.append(self._nodes_to_ref([u, v]))
            labels.append(label)
            assert len(node_pairs) == len(labels)

            if np.random.random() > 0.999:
                logi("Label Distribution at step %d: %s" % (itr, Counter(labels)))

        logi("Final Label Distribution: %s" % (Counter(labels)))

        self.is_data_generated = True
        return node_pairs, path_lists, labels

    def create_labeled_wsn_dataset_for_baselines(self,
                                                 path_lengths, max_paths,
                                                 shuffle=True, sample_size=None):
        """ Create the dataset. For each path length, create a separate list

            For WSN, we do not need to collect any data from non-edge pairs.
            And since it is regression, there is no need for balance as well.
            So the rating of each edge is the label, and each edge is an entry.
            And from all the edges, we simply perform the split as needed.

        """
        all_data = list(self.g.edges)
        logi("Size of data: %d" % len(all_data))
        if shuffle:
            random.shuffle(all_data)

        node_pairs = []
        path_lists = {}
        labels = []

        if sample_size is not None:
            data = random.sample(all_data, sample_size) if sample_size < len(all_data) else all_data
        else:
            data = all_data

        for itr, (u, v) in enumerate(data):
            caches = {i: [] for i in path_lengths}
            # caches = defaultdict(list)
            label = self.g[u][v]['rating']

            node_pairs.append(self._nodes_to_ref([u, v]))
            labels.append(label)
            assert len(node_pairs) == len(labels)

            if np.random.random() > 0.999:
                logi("Label Distribution at step %d: %s" % (itr, Counter(labels)))

        # logi("Final Label Distribution: \n", (np.histogram(labels, bins=20)))

        self.is_data_generated = True
        return node_pairs, path_lists, labels

    def get_train_test_data(self, path_lengths,
                            max_paths,
                            shuffle=True,
                            problem='link',
                            sample_size=None,
                            split_point=0.9,
                            store=True):
        """ Create dataset for entire data at once,
            and retain in class object.
            This can then be re-partitioned for different split-points.
        """

        if self.learning_dataset is None:
            logi("Generating fresh dataset")
            if problem == 'link':

                if self.baselines_only:
                    self.learning_dataset = self.create_labeled_link_dataset_for_baselines(path_lengths,
                                                                                           max_paths, shuffle=shuffle,
                                                                                           sample_size=sample_size)
                else:
                    self.learning_dataset = self.create_labeled_link_dataset(path_lengths,
                                                                             max_paths, shuffle=shuffle,
                                                                             sample_size=sample_size)

            elif problem == 'wsn':
                if self.baselines_only:
                    self.learning_dataset = self.create_labeled_wsn_dataset_for_baselines(path_lengths,
                                                                                          max_paths,
                                                                                          shuffle=shuffle,
                                                                                          sample_size=sample_size)
                else:
                    self.learning_dataset = self.create_labeled_wsn_dataset(path_lengths,
                                                                            max_paths, shuffle=shuffle,
                                                                            sample_size=sample_size)

        else:
            logi("Using pre-generated dataset")

        logi("Split point: {}".format(split_point))

        node_pairs, path_lists, labels = self.learning_dataset
        X = [node_pairs] + path_lists.values()
        Y = labels

        split_index = int(len(labels) * split_point)

        train_x = [np.asarray(i[:split_index]) for i in X]
        test_x = [np.asarray(i[split_index:]) for i in X]

        train_y = np.asarray(Y[:split_index])
        test_y = np.asarray(Y[split_index:])

        if store:
            train_x_file = "data/npy/{}_train_x.npy".format(self.dataset_name)
            train_y_file = "data/npy/{}_train_y.npy".format(self.dataset_name)
            test_x_file = "data/npy/{}_test_x.npy".format(self.dataset_name)
            test_y_file = "data/npy/{}_test_y.npy".format(self.dataset_name)

            np.save(train_x_file, train_x)
            np.save(train_y_file, train_y)
            np.save(test_x_file, test_x)
            np.save(test_y_file, test_y)

        logi("Sizes: Train = {}, Test = {}".format(len(train_x), len(test_x)))

        return train_x, train_y, test_x, test_y

    def _get_node_features_from_scipy_sparse_matrix(self, features, add_zero=True,
                                                    layer_name='node_features'):
        """ Get keras embedding layer of features from sparse scipy matrix"""
        dense_features = features.toarray()
        feature_dim = dense_features.shape[1]
        if add_zero:
            dense_features = np.vstack([np.zeros((1, feature_dim)), dense_features])

        return dense_features


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LEAP data ops')

    parser.add_argument('--dataset', type=str, help="Dataset name", default="USAir")
    parser.add_argument('--dataset_type', type=str, help="Dataset type", default="seal")
    parser.add_argument('--path_lengths', type=list, help="Path lengths", default=[3, 4])
    parser.add_argument('--max_paths', type=dict, help="Max path size per length", default={3: 50, 4: 50})
    parser.add_argument('--store_data', type=bool, help="Store data", default=False)
    parser.add_argument('--visualize', type=bool, help="Visualize model", default=False)
    parser.add_argument('--problem', type=str, help="Type of problem", default='link')
    parser.add_argument('--sample_size', type=int, help="Sample size (2X links)", default=None)
    parser.add_argument('--for_baselines', type=bool, help="Perform for baselines", default=False)

    args = parser.parse_args()
    ordered_args = OrderedDict(vars(args))

    dh = DataHandling(dataset_name=ordered_args['dataset'], dataset_type=ordered_args['dataset_type'],
                      ordered_args=ordered_args)

    dx, dy, tdx, tdy = dh.get_train_test_data(path_lengths=ordered_args['path_lengths'],
                                              max_paths=ordered_args['max_paths'],
                                              store=ordered_args['store_data'],
                                              problem=ordered_args['problem'],
                                              sample_size=ordered_args['sample_size'])
