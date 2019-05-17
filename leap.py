"""
leap

Path Aggregations for Link Prediction

This file controls the experiment code for
leap architecture

- Rakshit Agrawal, 2018
"""
import argparse
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import pearsonr
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, precision_score, recall_score, f1_score, \
    mean_squared_error

from aggregators import Aggregators
from data_ops import DataHandling
from log_control import logi

kc = tf.keras.callbacks


class ModelLearning(object):

    def __init__(self, data=None, ordered_args=None):
        self.model = None
        self.data = data
        self.ordered_args = ordered_args if not None else {}

        self.problem = self.ordered_args.get('problem','link')

    def compile_model(self, model,
                      optimizer='adam',
                      loss='binary_crossentropy',
                      lr=0.001,
                      metrics=None):

        if self.problem == 'link':
            loss = 'binary_crossentropy'
        elif self.problem == 'wsn':
            loss = 'mse'
        else:
            loss = 'categorical_crossentropy'

        if metrics is None:
            metrics = ['acc']

        if optimizer == 'adam':
            optimizer = tf.keras.optimizers.Adam(lr=lr)

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        self.model = model
        return model

    def train_model(self, model, data_x, data_y,
                    validation_split=0.2,
                    epochs=30, metadata_file=None,
                    use_tensorboard=False):
        cbs = []
        es_callback = kc.EarlyStopping(monitor='val_loss', patience=5)
        cbs.append(es_callback)

        if metadata_file is not None:
            tb_callback = kc.TensorBoard(log_dir='results/logs',
                                         histogram_freq=1,
                                         # write_grads=True,
                                         # embeddings_freq=1,
                                         # embeddings_metadata=metadata_file,
                                         # embeddings_layer_names=['embedding_layer'],
                                         # embeddings_data=[list(self.data.g.nodes)]*len(model.inputs)
                                         )
            logi("Tensorboard with embeddings: {}".format(tb_callback))
        else:
            tb_callback = kc.TensorBoard(log_dir='results/logs')
            logi("Tensorboard: {}".format(tb_callback))

        if use_tensorboard or metadata_file is not None:
            cbs.append(tb_callback)

        assert isinstance(model, tf.keras.Model)

        sess = tf.Session()
        tf.keras.backend.set_session(sess)
        model.fit(x=data_x, y=data_y,
                  epochs=epochs,
                  validation_split=validation_split,
                  shuffle=True, callbacks=cbs)

        self.model = model
        return model

    def eval(self, model, test_x, test_y):
        assert isinstance(model, tf.keras.Model)

        score = model.evaluate(test_x, test_y)

        print("Score from model: %s" % score)

        pred_y = model.predict(test_x)

        if self.problem == 'link':
            auc_score = roc_auc_score(y_score=pred_y, y_true=test_y)
            print("AUC score: %s" % auc_score)

            prf = precision_recall_fscore_support(y_true=test_y, y_pred=np.round(pred_y))
            print("PRF score: %r" % (str(prf)))

            prec = precision_score(y_true=test_y, y_pred=np.round(pred_y))
            rec = recall_score(y_true=test_y, y_pred=np.round(pred_y))
            f1 = f1_score(y_true=test_y, y_pred=np.round(pred_y))

            res_dict = dict(auc=auc_score,
                            precision=prec,
                            recall=rec,
                            f1_score=f1)

        elif self.problem == 'wsn':
            rmse = np.sqrt(mean_squared_error(y_true=test_y, y_pred=pred_y))
            pearson, pearson_p_val = pearsonr(test_y.flatten(), pred_y.flatten())

            res_dict = dict(rmse=rmse,
                            pearson=pearson,
                            pearson_p_val=pearson_p_val)
        else:
            res_dict = {}

        print(res_dict)

        return res_dict

    def visualize(self, model, viz_file):
        """ Visualize the model"""

        embeddings = model.get_layer('embedding_layer').get_weights()[0]
        vecs = embeddings.tolist()
        labels = list(range(len(vecs)))

        logi("Visualizing model using T-SNE")

        # TSNE
        tsne_model = TSNE(perplexity=20,
                          n_components=2,
                          init='random',
                          n_iter=2500,
                          # random_state=23
                          )
        new_values = tsne_model.fit_transform(vecs)

        x, y = zip(*new_values)
        plt.figure(figsize=(16, 16))
        for i in range(len(x)):
            plt.scatter(x[i], y[i])
            plt.annotate(labels[i],
                         xy=(x[i], y[i]),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')

        plt.savefig(viz_file)
        logi("Visualization saved to {}".format(viz_file))

    def projection(self, model, viz_file):
        """ Visualize the model"""

        embeddings = model.get_layer('embedding_layer').get_weights()[0]

        vec_df = pd.DataFrame(embeddings)

        vec_df.to_csv(viz_file, sep='\t', header=None, index=None)
        logi("Project data saved to {}".format(viz_file))


def run_learning(ordered_args):
    dh = DataHandling(dataset_name=ordered_args['dataset'],
                      dataset_type=ordered_args['dataset_type'],
                      ordered_args=ordered_args)

    dx, dy, tdx, tdy = dh.get_train_test_data(path_lengths=ordered_args['path_lengths'],
                                              max_paths=ordered_args['max_paths'],
                                              split_point=ordered_args['split_point'],
                                              problem=ordered_args['problem'],
                                              store=ordered_args['store_data'],
                                              sample_size=ordered_args['sample_size'])

    ag = Aggregators(node_count=dh.node_len + 1,
                     path_lengths=ordered_args['path_lengths'],
                     ordered_args=ordered_args)

    build_method = ag.get_build_method(model_name=ordered_args['method'])

    model = build_method(emb_dims=ordered_args['emb_dims'],
                         dense_dims=ordered_args['dense_dims'],
                         classifier_dims=ordered_args['classifier_dims'],
                         dropout=ordered_args['dropout'],
                         show_summary=ordered_args['show_summary'])

    ml = ModelLearning()

    model = ml.compile_model(model=model,
                             optimizer=ordered_args['optimizer'],
                             lr=ordered_args['learning_rate'])

    model = ml.train_model(model=model,
                           data_x=dx,
                           data_y=dy,
                           validation_split=0.2,
                           epochs=ordered_args['epochs'])

    ml.eval(model, tdx, tdy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='leap learning')

    parser.add_argument('--dataset', type=str, help="Dataset name", default="USAir")
    parser.add_argument('--dataset_type', type=str, help="Dataset type", default="seal")
    parser.add_argument('--method', type=str, help="Aggregator name", default="avg_pool")
    parser.add_argument('--path_lengths', type=list, help="Path lengths", default=[3, 4])
    parser.add_argument('--max_paths', type=dict, help="Max path size per length", default={3: 50, 4: 50})
    parser.add_argument('--store_data', type=bool, help="Store data", default=False)
    parser.add_argument('--emb_dims', type=int, help="Embedding dimensions", default=32)
    parser.add_argument('--dense_dims', type=int, help="Neural dimensions", default=32)
    parser.add_argument('--classifier_dims', type=int, help="Classifier dimensions", default=32)
    parser.add_argument('--dropout', type=float, help="Dropout", default=0.5)
    parser.add_argument('--learning_rate', type=float, help="Learning rate", default=0.001)
    parser.add_argument('--show_summary', type=bool, help="Show summary", default=True)
    parser.add_argument('--optimizer', type=str, help="Optimizer for learning", default='adam')
    parser.add_argument('--epochs', type=int, help="Epochs", default=20)
    parser.add_argument('--split_point', type=float, help="Split point", default=0.9)
    parser.add_argument('--visualize', type=bool, help="Visualize data ", default=False)
    parser.add_argument('--problem', type=str, help="Type of problem", default='link')
    parser.add_argument('--sample_size', type=int, help="Sample size (2X links)", default=None)

    args = parser.parse_args()
    ordered_args = OrderedDict(vars(args))

    run_learning(ordered_args=ordered_args)
