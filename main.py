"""
Main file for launching experiments

- Rakshit Agrawal, 2018
"""
import argparse
from collections import OrderedDict

from gensim.models import Word2Vec

import node2vec
from aggregators import Aggregators
from data_ops import DataHandling
from leap import ModelLearning
from log_control import logi
from utils import Utils


def run_multiple(ordered_args, datasets, methods, params, dataset_type=None, method_repeats=1, split_points=None):
    time_val = Utils.current_time_for_file_name()
    fname = "multiple_results_{}.csv".format(time_val)
    fname = Utils.results_file(fname)

    for dataset in datasets:
        dh = DataHandling(dataset_name=dataset, dataset_type=dataset_type, ordered_args=ordered_args)

        for split_point in split_points:

            dx, dy, tdx, tdy = dh.get_train_test_data(path_lengths=ordered_args['path_lengths'],
                                                      max_paths=ordered_args['max_paths'],
                                                      split_point=split_point,
                                                      problem=ordered_args['problem'],
                                                      store=ordered_args['store_data'],
                                                      sample_size=ordered_args['sample_size'])

            ag = Aggregators(node_count=dh.node_len + 1,
                             path_lengths=ordered_args['path_lengths'],
                             ordered_args=ordered_args,
                             additional_embeddings=dh.additional_embeddings)

            for method in methods:
                for m_iter in range(method_repeats):
                    build_method = ag.get_build_method(model_name=method)
                    for param_set in params:
                        res_dict = dict(dataset=dataset, method=method)
                        print(res_dict)
                        model = build_method(emb_dims=param_set['emb_dims'],
                                             dense_dims=param_set['dense_dims'],
                                             classifier_dims=param_set['classifier_dims'],
                                             dropout=param_set['dropout'],
                                             show_summary=ordered_args['show_summary'])

                        ml = ModelLearning(data=dh, ordered_args=ordered_args)

                        model = ml.compile_model(model=model,
                                                 optimizer=ordered_args['optimizer'],
                                                 lr=ordered_args['learning_rate'])

                        model = ml.train_model(model=model,
                                               data_x=dx,
                                               data_y=dy,
                                               validation_split=0.2,
                                               epochs=ordered_args['epochs'],
                                               metadata_file=dh.metadata_file,
                                               use_tensorboard=ordered_args['tensorboard'])

                        eval_res = ml.eval(model, tdx, tdy)

                        res_dict.update(eval_res)
                        res_dict.update(emb_dims=param_set['emb_dims'],
                                        dense_dims=param_set['dense_dims'],
                                        classifier_dims=param_set['classifier_dims'],
                                        dropout=param_set['dropout'])
                        res_dict.update(path_lengths=ordered_args['path_lengths'])
                        res_dict.update(split_point=split_point)
                        Utils.append_dict_to_csv([res_dict], fname, use_results_dir=False)

                        if ordered_args['visualize']:
                            viz_file = "viz_{}_{}_{}".format(method, dataset, time_val)
                            viz_file = Utils.results_file(viz_file)
                            ml.visualize(model=model, viz_file=viz_file)

                        if ordered_args['projection']:
                            viz_file = "projection_{}_{}_{}.tsv".format(method, dataset, time_val)
                            viz_file = Utils.results_file(viz_file)
                            ml.projection(model=model, viz_file=viz_file)


def run_multiple_with_emb(ordered_args, datasets, methods, params, dataset_type=None, method_repeats=1,
                          split_points=None):
    fname = "multiple_results_{}.csv".format(Utils.current_time_for_file_name())
    fname = Utils.results_file(fname)

    for dataset in datasets:
        dh = DataHandling(dataset_name=dataset, dataset_type=dataset_type, ordered_args=ordered_args)

        for split_point in split_points:

            dx, dy, tdx, tdy = dh.get_train_test_data(path_lengths=ordered_args['path_lengths'],
                                                      max_paths=ordered_args['max_paths'],
                                                      split_point=split_point,
                                                      problem=ordered_args['problem'],
                                                      store=ordered_args['store_data'],
                                                      sample_size=ordered_args['sample_size'])

            ag = Aggregators(node_count=dh.node_len + 1,
                             path_lengths=ordered_args['path_lengths'],
                             ordered_args=ordered_args)
            emb_cache = {}
            for method in methods:
                for m_iter in range(method_repeats):
                    build_method = ag.get_build_method(model_name=method)
                    method_name = "{}_n2v".format(method)

                    for param_set in params:
                        res_dict = dict(dataset=dataset, method=method_name)
                        print(res_dict)

                        emb_dims = param_set['emb_dims']
                        if emb_dims in emb_cache.keys():
                            known_emb = emb_cache[emb_dims]
                        else:
                            known_emb = get_n2v_embeddings(param_set['emb_dims'], dh.g)
                            emb_cache[emb_dims] = known_emb

                        model = build_method(emb_dims=param_set['emb_dims'],
                                             dense_dims=param_set['dense_dims'],
                                             classifier_dims=param_set['classifier_dims'],
                                             dropout=param_set['dropout'],
                                             show_summary=ordered_args['show_summary'],
                                             known_embeddings=known_emb)

                        ml = ModelLearning(data=dh, ordered_args=ordered_args)

                        model = ml.compile_model(model=model,
                                                 optimizer=ordered_args['optimizer'],
                                                 lr=ordered_args['learning_rate'])

                        model = ml.train_model(model=model,
                                               data_x=dx,
                                               data_y=dy,
                                               validation_split=0.2,
                                               epochs=ordered_args['epochs'],
                                               metadata_file=dh.metadata_file)

                        eval_res = ml.eval(model, tdx, tdy)

                        res_dict.update(eval_res)
                        res_dict.update(emb_dims=param_set['emb_dims'],
                                        dense_dims=param_set['dense_dims'],
                                        classifier_dims=param_set['classifier_dims'],
                                        dropout=param_set['dropout'])
                        res_dict.update(path_lengths=ordered_args['path_lengths'])
                        res_dict.update(split_point=split_point)
                        Utils.append_dict_to_csv([res_dict], fname, use_results_dir=False)


def learn_embeddings(walks, emb_dim):
    """
    Learn embeddings by optimizing the Skipgram objective using SGD.
    """
    walks = [map(str, walk) for walk in walks]
    model = Word2Vec(walks,
                     size=emb_dim,
                     window=10,
                     min_count=0, sg=1,
                     workers=6,
                     iter=5)
    emb = model.wv.get_keras_embedding()
    logi("Embeddings ready")
    return emb


def get_n2v_embeddings(emb_dim, nx_G):
    """
    Pipeline for representational learning for all nodes in a graph.
    """
    G = node2vec.Graph(nx_G, is_directed=False, p=1, q=1)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(10, 80)
    learn_embeddings(walks, emb_dim)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PALP learning')

    parser.add_argument('--dataset', type=str, help="Dataset name", default="USAir")
    parser.add_argument('--dataset_type', type=str, help="Dataset type", default="seal")
    parser.add_argument('--method', type=str, help="Aggregator name", default="avg_pool")
    parser.add_argument('--path_lengths', type=list, help="Path lengths", default=[3, 4])
    parser.add_argument('--max_paths', type=dict, help="Max path size per length", default={3: 50, 4: 50})
    parser.add_argument('--pl_start', type=int, help="Start path length", default=3)
    parser.add_argument('--pl_end', type=int, help="End path length", default=4)
    parser.add_argument('--max_fix', type=int, help="Fix val for all max paths", default=30)
    parser.add_argument('--store_data', type=bool, help="Store data", default=False)
    parser.add_argument('--emb_dims', type=int, help="Embedding dimensions", default=32)
    parser.add_argument('--dense_dims', type=int, help="Neural dimensions", default=32)
    parser.add_argument('--classifier_dims', type=int, help="Classifier dimensions", default=32)
    parser.add_argument('--dropout', type=float, help="Dropout", default=0.5)
    parser.add_argument('--learning_rate', type=float, help="Learning rate", default=0.001)
    parser.add_argument('--show_summary', type=bool, help="Show summary", default=True)
    parser.add_argument('--optimizer', type=str, help="Optimizer for learning", default='adam')
    parser.add_argument('--epochs', type=int, help="Epochs", default=30)
    parser.add_argument('--split_point', type=float, help="Split point", default=None)
    parser.add_argument('--bulk_setting', type=str, help="Setting to run", default='regular')
    parser.add_argument('--use_n2v', type=bool, help="Use node2vec?", default=False)
    parser.add_argument('--sample_size', type=int, help="Sample size (2X links)", default=None)
    parser.add_argument('--agg', type=str, help="Particular aggregator", default=None)
    parser.add_argument('--visualize', type=bool, help="Visualize model", default=False)
    parser.add_argument('--projection', type=bool, help="Projections of model", default=False)
    parser.add_argument('--tensorboard', type=bool, help="Use tensorboard", default=False)
    parser.add_argument('--repeats', type=int, help="Method repeats", default=1)
    parser.add_argument('--multi_split', type=bool, help="Perform multiple splits", default=False)
    parser.add_argument('--problem', type=str, help="Type of problem", default='link')
    parser.add_argument('--for_baselines', type=bool, help="Perform for baselines", default=False)
    parser.add_argument('--use_node_features', type=bool, help="Use node features", default=False)
    parser.add_argument('--regression_only', type=bool, help="Perform pure regression", default=False)

    args = parser.parse_args()
    ordered_args = OrderedDict(vars(args))

    REGULAR_DATASETS = ['USAir', 'NS', 'PB', 'Celegans', 'Ecoli', 'Yeast']
    LARGE_DATASETS = ['arxiv', 'facebook']

    REGULAR_SPLIT = 0.9
    LARGE_SPLIT = 0.5

    AGGS = ['avg_pool', 'dense_max', 'seq_of_seq', 'edge_conv']

    ordered_args['path_lengths'] = range(ordered_args['pl_start'], ordered_args['pl_end'] + 1)
    if not ordered_args['path_lengths'] == ordered_args['max_paths'].keys():
        ordered_args['max_paths'] = {i: ordered_args['max_fix'] for i in ordered_args['path_lengths']}

    bulk_setting = ordered_args['bulk_setting']
    method_repeats = ordered_args['repeats']

    d_type = 'seal'
    SPLIT_POINTS = [REGULAR_SPLIT, LARGE_SPLIT]

    if bulk_setting == 'regular':
        PARAMS = [
            dict(dense_dims=32, emb_dims=32, classifier_dims=32, dropout=0.5),
            dict(dense_dims=32, emb_dims=16, classifier_dims=32, dropout=0.5),
            dict(dense_dims=48, emb_dims=24, classifier_dims=32, dropout=0.5),
            dict(dense_dims=64, emb_dims=32, classifier_dims=32, dropout=0.5),
            dict(dense_dims=32, emb_dims=8, classifier_dims=64, dropout=0.5),
            dict(dense_dims=32, emb_dims=16, classifier_dims=64, dropout=0.5),
            dict(dense_dims=64, emb_dims=32, classifier_dims=64, dropout=0.5),
        ]
        d = REGULAR_DATASETS[:]
        d_type = ordered_args['dataset_type']
        a = AGGS[:]
        p = PARAMS[:]

        if ordered_args['split_point'] is None:
            ordered_args['split_point'] = REGULAR_SPLIT

    elif bulk_setting == 'large':
        PARAMS = [
            dict(dense_dims=32, emb_dims=32, classifier_dims=32, dropout=0.5),
            dict(dense_dims=48, emb_dims=48, classifier_dims=32, dropout=0.5),
            dict(dense_dims=64, emb_dims=64, classifier_dims=32, dropout=0.5),
            dict(dense_dims=100, emb_dims=128, classifier_dims=64, dropout=0.5),
        ]
        d = LARGE_DATASETS[:]
        d_type = ordered_args['dataset_type']
        a = AGGS[:]
        p = PARAMS[:]

        if ordered_args['split_point'] is None:
            ordered_args['split_point'] = LARGE_SPLIT

    elif bulk_setting == 'large_single':
        PARAMS = [
            dict(dense_dims=32, emb_dims=32, classifier_dims=32, dropout=0.5),
            dict(dense_dims=64, emb_dims=64, classifier_dims=32, dropout=0.5),
            dict(dense_dims=100, emb_dims=128, classifier_dims=64, dropout=0.5),
        ]
        d = [ordered_args['dataset']]
        d_type = ordered_args['dataset_type']
        a = AGGS[:]
        if ordered_args['agg'] is not None:
            a = [ordered_args['agg']]
        p = PARAMS[:]

        if ordered_args['split_point'] is None:
            ordered_args['split_point'] = LARGE_SPLIT

    elif bulk_setting == 'regular_single':
        PARAMS = [
            dict(dense_dims=32, emb_dims=32, classifier_dims=32, dropout=0.5),
            dict(dense_dims=100, emb_dims=64, classifier_dims=48, dropout=0.5),
            dict(dense_dims=128, emb_dims=64, classifier_dims=64, dropout=0.5),
            dict(dense_dims=100, emb_dims=128, classifier_dims=64, dropout=0.5),
        ]
        d = [ordered_args['dataset']]
        d_type = ordered_args['dataset_type']
        a = AGGS[:]
        if ordered_args['agg'] is not None:
            a = [ordered_args['agg']]
        p = PARAMS[:]

        if ordered_args['split_point'] is None:
            ordered_args['split_point'] = REGULAR_SPLIT

    elif bulk_setting == 'vlarge_single':
        PARAMS = [
            dict(dense_dims=32, emb_dims=32, classifier_dims=32, dropout=0.5),
            dict(dense_dims=64, emb_dims=64, classifier_dims=32, dropout=0.5),
            dict(dense_dims=100, emb_dims=128, classifier_dims=64, dropout=0.5),
        ]
        d = [ordered_args['dataset']]
        d_type = ordered_args['dataset_type']
        a = AGGS[:]
        if ordered_args['agg'] is not None:
            a = [ordered_args['agg']]
        p = PARAMS[:]

        if ordered_args['split_point'] is None:
            ordered_args['split_point'] = LARGE_SPLIT

    elif bulk_setting == 'wsn_single':
        PARAMS = [
            dict(dense_dims=80, emb_dims=64, classifier_dims=32, dropout=0.5),
            dict(dense_dims=48, emb_dims=64, classifier_dims=32, dropout=0.5),
            dict(dense_dims=48, emb_dims=32, classifier_dims=32, dropout=0.5),
        ]
        d = [ordered_args['dataset']]
        d_type = ordered_args['dataset_type']
        a = AGGS[:]
        if ordered_args['agg'] is not None:
            a = [ordered_args['agg']]
        p = PARAMS[:]

        if ordered_args['split_point'] is None:
            ordered_args['split_point'] = REGULAR_SPLIT

    if ordered_args['multi_split']:
        if ordered_args['problem'] == 'link':
            SPLIT_POINTS = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        elif ordered_args['problem'] == 'wsn':
            SPLIT_POINTS = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    else:
        SPLIT_POINTS = [ordered_args['split_point']]

    print("Staring for datasets: {}\n and methods: {}".format(d, a))

    use_n2v = ordered_args['use_n2v']

    print(ordered_args)
    if use_n2v:
        run_multiple_with_emb(ordered_args, datasets=d, methods=a, params=p, dataset_type=d_type,
                              method_repeats=method_repeats, split_points=SPLIT_POINTS)
    else:
        run_multiple(ordered_args, datasets=d, methods=a, params=p, dataset_type=d_type,
                     method_repeats=method_repeats, split_points=SPLIT_POINTS)
