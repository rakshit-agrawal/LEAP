"""
aggregators

This file maintains methods for path aggregation

- Rakshit Agrawal, 2018
"""

import numpy as np
import tensorflow as tf

kl = tf.keras.layers


class Aggregators(object):

    def __init__(self,
                 node_count,
                 path_lengths,
                 additional_embeddings=None,
                 ordered_args=None):

        self.node_count = node_count
        self.path_lengths = path_lengths
        self.additional_embeddings = additional_embeddings
        self.ordered_args = ordered_args if not None else {}

        # Set the problem for output layers
        self.problem = self.ordered_args.get('problem', 'link')

    def get_build_method(self, model_name):

        model_ref = {
            'avg_pool': self.build_mean_model,
            'dense_max': self.build_dense_max_model,
            'seq_of_seq': self.build_seq_of_seq_model,
            'edge_conv': self.build_edge_conv_model
        }

        return model_ref.get(model_name, None)

    def get_final_output_layer(self, n_classes=None):
        """ Create final layer of model based on problem type. """

        if n_classes is not None and isinstance(n_classes, int):
            return kl.Dense(n_classes, activation='softmax', name='final_val')

        if self.problem == 'link':
            return kl.Dense(1, activation='sigmoid', name='final_val')

        if self.problem == 'wsn':
            if self.ordered_args.get('regression_only',False):
                return kl.Dense(1, name='final_val')
            return kl.Dense(1, activation='tanh', name='final_val')

    def build_mean_model(self,
                         emb_dims=32,
                         dense_dims=32,
                         classifier_dims=32,
                         dropout=0.5,
                         known_embeddings=None,
                         show_summary=True):
        """ Build a mean model """

        if isinstance(dense_dims, dict):
            assert set(dense_dims.keys()) == set(self.path_lengths.keys())
        elif isinstance(dense_dims, int):
            dense_dims = {i: dense_dims for i in self.path_lengths}

        node_inp = kl.Input((2,), name='node_pair_input')

        node_feature_values = []

        if known_embeddings is None:
            emb = kl.Embedding(input_dim=self.node_count,output_dim=emb_dims,
                               name='embedding_layer')
        else:
            assert isinstance(known_embeddings, kl.Embedding)
            emb = known_embeddings

        node_emb = emb(node_inp)
        processed_node_pair = kl.Flatten()(node_emb)
        node_feature_values.append(processed_node_pair)

        if self.additional_embeddings is not None:
            # Add node features through more embedding layers
            if isinstance(self.additional_embeddings, list):
                assert all([isinstance(i, np.ndarray) for i in self.additional_embeddings])
            elif isinstance(self.additional_embeddings, np.ndarray):
                self.additional_embeddings = [self.additional_embeddings]
            else:
                raise ValueError("Unknown embedding type provided.")

            for i, emb_weights in enumerate(self.additional_embeddings):
                emb_layer = kl.Embedding(input_dim=emb_weights.shape[0],
                                         output_dim=emb_weights.shape[1],
                                         weights=[emb_weights],
                                         trainable=False,
                                         name='node_features_{}'.format(i+1))
                node_features = emb_layer(node_inp)
                processed_features = kl.Flatten()(node_features)
                node_feature_values.append(processed_features)

        path_inps = {}
        path_embs = {}
        processed_paths = {}

        for path_len in self.path_lengths:
            path_inps[path_len] = kl.Input((None, path_len), name='path_%d_input' % path_len)
            path_embs[path_len] = emb(path_inps[path_len])

            processed_paths[path_len] = kl.TimeDistributed(
                kl.Flatten(name='flatten_for_%d_paths' % path_len)
            )(path_embs[path_len])

            processed_paths[path_len] = kl.GlobalAveragePooling1D(name='final_mean_pool_for_%d_paths' % path_len)(
                processed_paths[path_len])

        combined = kl.Concatenate()(node_feature_values + processed_paths.values())

        d2_out = kl.Dense(classifier_dims, name='dense_on_combined')(combined)
        d2_out = kl.Dropout(dropout)(d2_out)
        out = self.get_final_output_layer()(d2_out)

        model = tf.keras.Model(inputs=[node_inp] + path_inps.values(), outputs=out)

        if show_summary:
            model.summary()

        return model

    def build_dense_max_model(self,
                              emb_dims=32,
                              dense_dims=32,
                              classifier_dims=32,
                              dropout=0.5,
                              known_embeddings=None,
                              show_summary=True):
        """ Build a dense max model """

        if isinstance(dense_dims, dict):
            assert set(dense_dims.keys()) == set(self.path_lengths.keys())
        elif isinstance(dense_dims, int):
            dense_dims = {i: dense_dims for i in self.path_lengths}

        node_inp = kl.Input((2,), name='node_pair_input')
        node_feature_values = []

        if known_embeddings is None:
            emb = kl.Embedding(input_dim=self.node_count, output_dim=emb_dims,
                               name='embedding_layer')
        else:
            assert isinstance(known_embeddings, kl.Embedding)
            emb = known_embeddings
        node_emb = emb(node_inp)
        processed_node_pair = kl.Flatten()(node_emb)
        node_feature_values.append(processed_node_pair)

        if self.additional_embeddings is not None:
            # Add node features through more embedding layers
            if isinstance(self.additional_embeddings, list):
                assert all([isinstance(i, kl.Embedding) for i in self.additional_embeddings])
            elif isinstance(self.additional_embeddings, kl.Embedding):
                self.additional_embeddings = [self.additional_embeddings]
            else:
                raise ValueError("Unkonwn embedding type provided.")

            for emb_layer in self.additional_embeddings:
                node_features = emb_layer(node_inp)
                processed_features = kl.Flatten()(node_features)
                node_feature_values.append(processed_features)

        path_inps = {}
        path_embs = {}
        processed_paths = {}

        for path_len in self.path_lengths:
            path_inps[path_len] = kl.Input((None, path_len), name='path_%d_input' % path_len)
            path_embs[path_len] = emb(path_inps[path_len])

            processed_paths[path_len] = kl.TimeDistributed(
                kl.Flatten(name='flatten_for_%d_paths' % path_len)
            )(path_embs[path_len])

            processed_paths[path_len] = kl.TimeDistributed(
                kl.Dense(dense_dims[path_len], name='dense_for_%d_paths' % path_len),
                name='td_dense_for_%d_paths' % path_len)(processed_paths[path_len])

            processed_paths[path_len] = kl.Dropout(dropout)(processed_paths[path_len])

            processed_paths[path_len] = kl.GlobalMaxPooling1D(name='final_max_pool_for_%d_paths' % path_len)(
                processed_paths[path_len])

        combined = kl.Concatenate()(node_feature_values + processed_paths.values())

        d2_out = kl.Dense(classifier_dims, name='dense_on_combined')(combined)
        d2_out = kl.Dropout(dropout)(d2_out)
        out = self.get_final_output_layer()(d2_out)

        model = tf.keras.Model(inputs=[node_inp] + path_inps.values(), outputs=out)

        if show_summary:
            model.summary()

        return model

    def build_seq_of_seq_model(self,
                               emb_dims=32,
                               dense_dims=32,
                               classifier_dims=32,
                               dropout=0.5,
                               known_embeddings=None,
                               show_summary=True
                               ):
        """ Build a sequence of sequence model """

        if isinstance(dense_dims, dict):
            assert set(dense_dims.keys()) == set(self.path_lengths.keys())
        elif isinstance(dense_dims, int):
            dense_dims = {i: dense_dims for i in self.path_lengths}

        node_inp = kl.Input((2,), name='node_pair_input')
        node_feature_values = []

        if known_embeddings is None:
            emb = kl.Embedding(input_dim=self.node_count, output_dim=emb_dims,
                               name='embedding_layer')
        else:
            assert isinstance(known_embeddings, kl.Embedding)
            emb = known_embeddings
        node_emb = emb(node_inp)
        processed_node_pair = kl.Flatten()(node_emb)
        node_feature_values.append(processed_node_pair)

        if self.additional_embeddings is not None:
            # Add node features through more embedding layers
            if isinstance(self.additional_embeddings, list):
                assert all([isinstance(i, kl.Embedding) for i in self.additional_embeddings])
            elif isinstance(self.additional_embeddings, kl.Embedding):
                self.additional_embeddings = [self.additional_embeddings]
            else:
                raise ValueError("Unkonwn embedding type provided.")

            for emb_layer in self.additional_embeddings:
                node_features = emb_layer(node_inp)
                processed_features = kl.Flatten()(node_features)
                node_feature_values.append(processed_features)

        path_inps = {}
        path_embs = {}
        processed_paths = {}

        for path_len in self.path_lengths:
            path_inps[path_len] = kl.Input((None, path_len), name='path_%d_input' % path_len)
            path_embs[path_len] = emb(path_inps[path_len])

            processed_paths[path_len] = kl.TimeDistributed(
                kl.LSTM(dense_dims[path_len], return_sequences=True, name='lstm_for_%d_paths' % path_len),
                name='td_lstm_for_%d_paths' % path_len)(path_embs[path_len])

            processed_paths[path_len] = kl.Dropout(dropout)(processed_paths[path_len])

            processed_paths[path_len] = kl.TimeDistributed(
                kl.GlobalMaxPool1D(name='global_max_pool_for_%d_paths' % path_len),
                name='td_for_global_max_pool_for_%d_paths' % path_len)(processed_paths[path_len])

            processed_paths[path_len] = kl.LSTM(dense_dims[path_len] * 2, return_sequences=True,
                                                name='lstm_for_%d_paths' % path_len)(processed_paths[path_len])

            processed_paths[path_len] = kl.GlobalMaxPooling1D(name='final_max_pool_for_%d_paths' % path_len)(
                processed_paths[path_len])

        combined = kl.Concatenate()(node_feature_values + processed_paths.values())

        d2_out = kl.Dense(classifier_dims, name='dense_on_combined')(combined)
        d2_out = kl.Dropout(dropout)(d2_out)
        out = self.get_final_output_layer()(d2_out)

        model = tf.keras.Model(inputs=[node_inp] + path_inps.values(), outputs=out)

        if show_summary:
            model.summary()

        return model

    def build_edge_conv_model(self,
                              emb_dims=32,
                              dense_dims=32,
                              classifier_dims=32,
                              dropout=0.5,
                              known_embeddings=None,
                              show_summary=True
                              ):
        """ Build an edge conv model """

        if isinstance(dense_dims, dict):
            assert set(dense_dims.keys()) == set(self.path_lengths.keys())
        elif isinstance(dense_dims, int):
            dense_dims = {i: dense_dims for i in self.path_lengths}

        node_inp = kl.Input((2,), name='node_pair_input')
        node_feature_values = []

        if known_embeddings is None:
            emb = kl.Embedding(input_dim=self.node_count, output_dim=emb_dims,
                               name='embedding_layer')
        else:
            assert isinstance(known_embeddings, kl.Embedding)
            emb = known_embeddings
        node_emb = emb(node_inp)
        processed_node_pair = kl.Flatten()(node_emb)
        node_feature_values.append(processed_node_pair)

        if self.additional_embeddings is not None:
            # Add node features through more embedding layers
            if isinstance(self.additional_embeddings, list):
                assert all([isinstance(i, kl.Embedding) for i in self.additional_embeddings])
            elif isinstance(self.additional_embeddings, kl.Embedding):
                self.additional_embeddings = [self.additional_embeddings]
            else:
                raise ValueError("Unkonwn embedding type provided.")

            for emb_layer in self.additional_embeddings:
                node_features = emb_layer(node_inp)
                processed_features = kl.Flatten()(node_features)
                node_feature_values.append(processed_features)

        path_inps = {}
        path_embs = {}
        processed_paths = {}

        for path_len in self.path_lengths:
            path_inps[path_len] = kl.Input((None, path_len), name='path_%d_input' % path_len)
            path_embs[path_len] = emb(path_inps[path_len])

            processed_paths[path_len] = kl.TimeDistributed(
                kl.Conv1D(filters=emb_dims, kernel_size=2, strides=1,
                          name='conv_for_%d_paths' % path_len),
                name='td_conv_for_%d_paths' % path_len)(path_embs[path_len])

            processed_paths[path_len] = kl.Dropout(dropout)(processed_paths[path_len])

            processed_paths[path_len] = kl.TimeDistributed(
                kl.GlobalMaxPool1D(name='global_max_pool_for_%d_paths' % path_len),
                name='td_for_global_max_pool_for_%d_paths' % path_len)(processed_paths[path_len])

            processed_paths[path_len] = kl.LSTM(dense_dims[path_len] * 2, return_sequences=True,
                                                name='lstm_for_%d_paths' % path_len)(processed_paths[path_len])

            processed_paths[path_len] = kl.GlobalMaxPooling1D(name='final_max_pool_for_%d_paths' % path_len)(
                processed_paths[path_len])

        combined = kl.Concatenate()(node_feature_values + processed_paths.values())

        d2_out = kl.Dense(classifier_dims, name='dense_on_combined')(combined)
        d2_out = kl.Dropout(dropout)(d2_out)
        out = self.get_final_output_layer()(d2_out)

        model = tf.keras.Model(inputs=[node_inp] + path_inps.values(), outputs=out)

        if show_summary:
            model.summary()

        return model


if __name__ == "__main__":
    # Test
    ag = Aggregators(node_count=200, path_lengths=[3, 4])

    model = ag.build_mean_model()
    model = ag.build_dense_max_model()
    model = ag.build_seq_of_seq_model()
    model = ag.build_edge_conv_model()
