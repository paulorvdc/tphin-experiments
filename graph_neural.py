from tensorflow.keras import backend as K
from ge import LINE
from ge import Struc2Vec
from ge import Node2Vec
from ge import DeepWalk
import pandas as pd
import networkx as nx
import numpy as np
from scipy.spatial.distance import cosine
import numpy as np
import pickle5 as pickle
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tphin_utils import metapath2vec
from tphin_utils import gcn
from tphin_utils import embedding_graph
from tphin_utils import regularization
from tphin_utils import type_code_graph
from tphin_utils import prepare_train_test
from tphin_utils import get_lstm
from tphin_utils import next_labels_cut

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def run_lstm(G_disturbed, cutted_dict, algorithm, interval, commodity, time_window, i, path, 
                type_feature='node_type', event_type='event', label_number_feature='type_code', embedding_feature='f'
            ):
    if algorithm == 'regularization':
        G_disturbed = regularization(G_disturbed, iterations=30, mi=0.75)
        G_disturbed = type_code_graph(G_disturbed)
        X_train, X_test, y_train = prepare_train_test(G_disturbed)
        X_train, X_test = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1])), np.reshape(
            X_test, (X_test.shape[0], 1, X_test.shape[1]))
        K.clear_session()
        model = get_lstm(512, 4)
        model.fit(X_train, y_train, epochs=20, batch_size=time_window)
        y_pred = np.argmax(model.predict(X_test), axis=1)
        pd.Series(y_pred).to_csv('{}/pred_iterative/lstm_{}_{}_{}_{}_{}.csv'.format(path,algorithm,interval,commodity,time_window,i), index=False)

    elif algorithm == 'deep_walk':
        model_deep_walk = DeepWalk(
            G_disturbed, walk_length=10, num_walks=80, workers=1)
        model_deep_walk.train(window_size=5, iter=3,
                              embed_size=512)  # train model
        embeddings_deep_walk = model_deep_walk.get_embeddings()  # get embedding vectors
        G_disturbed = embedding_graph(G_disturbed, embeddings_deep_walk)
        G_disturbed = type_code_graph(G_disturbed)
        X_train, X_test, y_train = prepare_train_test(G_disturbed)
        X_train, X_test = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1])), np.reshape(
            X_test, (X_test.shape[0], 1, X_test.shape[1]))
        K.clear_session()
        model = get_lstm(512, 4)
        model.fit(X_train, y_train, epochs=20, batch_size=time_window)
        y_pred = np.argmax(model.predict(X_test), axis=1)
        pd.Series(y_pred).to_csv('{}/pred_iterative/lstm_{}_{}_{}_{}_{}.csv'.format(path,algorithm,interval,commodity,time_window,i), index=False)

    elif algorithm == 'node2vec':
        model_node2vec = Node2Vec(
            G_disturbed, walk_length=10, num_walks=80, p=0.5, q=1, workers=1)
        model_node2vec.train(window_size=5, iter=3,
                             embed_size=512)  # train model
        embeddings_node2vec = model_node2vec.get_embeddings()  # get embedding vectors
        G_disturbed = embedding_graph(G_disturbed, embeddings_node2vec)
        G_disturbed = type_code_graph(G_disturbed)
        X_train, X_test, y_train = prepare_train_test(G_disturbed)
        X_train, X_test = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1])), np.reshape(
            X_test, (X_test.shape[0], 1, X_test.shape[1]))
        K.clear_session()
        model = get_lstm(512, 4)
        model.fit(X_train, y_train, epochs=20, batch_size=time_window)
        y_pred = np.argmax(model.predict(X_test), axis=1)
        pd.Series(y_pred).to_csv('{}/pred_iterative/lstm_{}_{}_{}_{}_{}.csv'.format(path,algorithm,interval,commodity,time_window,i), index=False)

    elif algorithm == 'struc2vec':
        model_struc2vec = Struc2Vec(
            G_disturbed, 10, 80, workers=2, verbose=40)  # init model
        model_struc2vec.train(window_size=5, iter=3,
                              embed_size=512)  # train model
        embeddings_struc2vec = model_struc2vec.get_embeddings()  # get embedding vectors
        G_disturbed = embedding_graph(G_disturbed, embeddings_struc2vec)
        G_disturbed = type_code_graph(G_disturbed)
        X_train, X_test, y_train = prepare_train_test(G_disturbed)
        X_train, X_test = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1])), np.reshape(
            X_test, (X_test.shape[0], 1, X_test.shape[1]))
        K.clear_session()
        model = get_lstm(512, 4)
        model.fit(X_train, y_train, epochs=20, batch_size=time_window)
        y_pred = np.argmax(model.predict(X_test), axis=1)
        pd.Series(y_pred).to_csv('{}/pred_iterative/lstm_{}_{}_{}_{}_{}.csv'.format(path,algorithm,interval,commodity,time_window,i), index=False)

    elif algorithm == 'metapath2vec':
        embeddings_metapath2vec = metapath2vec(G_disturbed)
        G_disturbed = embedding_graph(G_disturbed, embeddings_metapath2vec)
        G_disturbed = type_code_graph(G_disturbed)
        X_train, X_test, y_train = prepare_train_test(G_disturbed)
        X_train, X_test = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1])), np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
        K.clear_session()
        model = get_lstm(512, 4)
        model.fit(X_train, y_train, epochs=20, batch_size=time_window)
        y_pred = np.argmax(model.predict(X_test), axis=1)
        pd.Series(y_pred).to_csv('{}/pred_iterative/lstm_{}_{}_{}_{}_{}.csv'.format(path,algorithm,interval,commodity,time_window,i), index=False)

    elif algorithm == 'line':
        # init model,order can be ['first','second','all']
        model_line = LINE(G_disturbed, embedding_size=512, order='second')
        model_line.train(batch_size=8, epochs=20, verbose=0)  # train model
        embeddings_line = model_line.get_embeddings()  # get embedding vectors
        G_disturbed = embedding_graph(G_disturbed, embeddings_line)
        G_disturbed = type_code_graph(G_disturbed)
        X_train, X_test, y_train = prepare_train_test(G_disturbed)
        X_train, X_test = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1])), np.reshape(
            X_test, (X_test.shape[0], 1, X_test.shape[1]))
        K.clear_session()
        model = get_lstm(512, 4)
        model.fit(X_train, y_train, epochs=20, batch_size=time_window)
        y_pred = np.argmax(model.predict(X_test), axis=1)
        pd.Series(y_pred).to_csv('{}/pred_iterative/lstm_{}_{}_{}_{}_{}.csv'.format(path,algorithm,interval,commodity,time_window,i), index=False)

    elif algorithm == 'gcn':
        y_pred = gcn(G_disturbed, interval, i)
        pd.Series(y_pred).to_csv('{}/pred_iterative/lstm_{}_{}_{}_{}_{}.csv'.format(path,algorithm,interval,commodity,time_window,i), index=False)


algorithms = ['regularization', 'deep_walk', 'node2vec', 'struc2vec', 'metapath2vec', 'line', 'gcn']
path = "/media/pauloricardo/basement/commodities_usecase/"
intervals = ['week', 'month']
commodities = ['corn', 'soybean']
time_windows = [3, 6, 12]

label_codes = {
    'big_down': 0,
    'down': 1,
    'up': 2,
    'big_up': 3,
}

# next event label cut LSTM
for interval in intervals:
    for commodity in commodities:
        with open(path + commodity + "_" + interval + ".gpickle", "rb") as fh:
            G = pickle.load(fh)
        for time_window in time_windows:
            for i in range(10):
                G_cutted, cutted_dict = next_labels_cut(G, time_window=time_window, interval=interval)
                y_true = cutted_dict['event_trend'].neighbor.to_list()
                for idx in range(len(y_true)):
                    y_true[idx] = label_codes[y_true[idx]]
                pd.Series(y_true).to_csv('{}/pred_iterative/true_{}_{}_{}.csv'.format(path, interval, commodity, time_window), index=False)
                for algorithm in algorithms:
                    print('TEST: {}, {}, {}, {}, {}'.format(algorithm, interval, commodity, time_window, i))
                    run_lstm(G_cutted, cutted_dict, algorithm, interval, commodity, time_window, i, path)
