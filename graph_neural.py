from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from ephin_utils import metapath2vec
from ephin_utils import gcn
from ephin_utils import masked_accuracy
from ephin_utils import masked_softmax_cross_entropy
from ephin_utils import embedding_graph
from ephin_utils import restore_hin
from ephin_utils import get_knn_data
from ephin_utils import regularization
from ge import LINE
from ge import Struc2Vec
from ge import Node2Vec
from ge import DeepWalk
import pandas as pd
import networkx as nx
import numpy as np
from scipy.spatial.distance import cosine
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import random
import pickle5 as pickle
import time
import scipy.sparse
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


#from ephin_utils import disturbed_hin

# mlp


def next_labels_cut(G,
                    time_window=1, interval='week', edge_type='event_trend', type_feature='edge_type',
                    node_type_feature='node_type', label_feature='trend', date_feture='date', event_feature='event'
                    ):
    def keep_left(x, G):
        edge_split = x['type'].split('_')
        if G.nodes[x['node']]['node_type'] != edge_split[0]:
            x['node'], x['neighbor'] = x['neighbor'], x['node']
        return x

    # prepare data for type counting
    edges = list(G.edges)
    edge_types, edge_dates = [], []
    for node, neighbor in edges:
        edge_types.append(G[node][neighbor][type_feature])
        if G.nodes[node][node_type_feature] == event_feature:
            for edge in G.neighbors(node):
                if G.nodes[edge][node_type_feature] == date_feture:
                    edge_dates.append(edge)

    # get labels for test
    labels = pd.Series(edge_dates).unique()
    labels = labels[len(labels)-time_window:]

    edges = pd.DataFrame(edges)
    edges = edges.rename(columns={0: 'node', 1: 'neighbor'})
    edges['type'] = edge_types
    edges = edges.apply(keep_left, G=G, axis=1)
    events_to_cut = edges.groupby(
        by=['node', 'neighbor'], as_index=False).count()
    events_to_cut = events_to_cut['node'][events_to_cut['neighbor'].isin(
        labels)]

    to_cut = edges[edges['node'].isin(events_to_cut)]
    to_cut = to_cut[to_cut['type'] == edge_type]
    to_cut_dict = {edge_type: to_cut}

    # eliminar arestas, salvar grafo e arestas retiradas para avaliação
    G_disturbed = deepcopy(G)
    for key, tc_df in to_cut_dict.items():
        for index, row in tc_df.iterrows():
            G_disturbed.remove_edge(row['node'], row['neighbor'])
    return G_disturbed, to_cut_dict


def type_code_graph(G, type_feature='node_type', event_type='event', label_feature='trend', label_number_feature='type_code'):
    node_list = list(G.nodes())
    label_codes = {
        'big_down': 0,
        'down': 1,
        'up': 2,
        'big_up': 3,
    }
    for node in node_list:
        G.nodes[node][label_number_feature] = -1
        if G.nodes[node][type_feature] == event_type:
            for edge in G.neighbors(node):
                if G.nodes[edge][type_feature] == label_feature:
                    G.nodes[node][label_number_feature] = label_codes[edge]
    return G


def get_mlp_deep(dimX, dimY):
    # number of units to each layer on the deep architecture
    units = dimX + round(dimY/2)
    model = Sequential()
    model.add(Dense(units, input_dim=dimX, activation='relu'))
    model.add(Dense(units, activation='relu'))
    model.add(Dense(units, activation='relu'))
    model.add(Dense(units, activation='relu'))
    model.add(Dense(units, activation='relu'))
    model.add(Dense(dimY, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


def get_lstm(dimX, dimY):
    model = Sequential()
    model.add(CuDNNLSTM(256, input_shape=(1, dimX)))
    model.add(Dense(dimY, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


def prepare_train_test(G_disturbed, type_feature='node_type', label_number_feature='type_code', embedding_feature='f', event_type='event'):
    X_train, X_test, y_train = [], [], []
    for node in G_disturbed.nodes():
        if G_disturbed.nodes[node][type_feature] == event_type:
            if G_disturbed.nodes[node][label_number_feature] == -1:
                X_test.append(G_disturbed.nodes[node][embedding_feature])
            else:
                X_train.append(G_disturbed.nodes[node][embedding_feature])
                y_train.append(
                    G_disturbed.nodes[node][label_number_feature])
    y_train = to_categorical(y_train, num_classes=4)
    X_train, X_test = np.asarray(X_train), np.asarray(X_test)
    return X_train, X_test, y_train

def run_model(G_disturbed, cutted_dict, algorithm, interval, commodity, path, iteration, type_feature='node_type', event_type='event', label_number_feature='type_code', embedding_feature='f'):
    if algorithm == 'regularization':
        G_disturbed = regularization(G_disturbed, iterations=30, mi=0.75)
        G_disturbed = type_code_graph(G_disturbed)
        X_train, X_test, y_train = prepare_train_test(G_disturbed)
        K.clear_session()
        model = get_mlp_deep(512, 4)
        model.fit(X_train, y_train, epochs=20, batch_size=50)
        y_pred = np.argmax(model.predict(X_test), axis=1)
        pd.Series(y_pred).to_csv('{}/pred/pred_{}_{}_{}_{}.csv'.format(path,
                                                                       algorithm, interval, commodity, i), index=False)

    elif algorithm == 'deep_walk':
        model_deep_walk = DeepWalk(
            G_disturbed, walk_length=10, num_walks=80, workers=1)
        model_deep_walk.train(window_size=5, iter=3,
                              embed_size=512)  # train model
        embeddings_deep_walk = model_deep_walk.get_embeddings()  # get embedding vectors
        G_disturbed = embedding_graph(G_disturbed, embeddings_deep_walk)
        G_disturbed = type_code_graph(G_disturbed)
        X_train, X_test, y_train = prepare_train_test(G_disturbed)
        K.clear_session()
        model = get_mlp_deep(512, 4)
        model.fit(X_train, y_train, epochs=20, batch_size=50)
        y_pred = np.argmax(model.predict(X_test), axis=1)
        pd.Series(y_pred).to_csv('{}/pred/pred_{}_{}_{}_{}.csv'.format(path,
                                                                       algorithm, interval, commodity, i), index=False)

    elif algorithm == 'node2vec':
        model_node2vec = Node2Vec(
            G_disturbed, walk_length=10, num_walks=80, p=0.5, q=1, workers=1)
        model_node2vec.train(window_size=5, iter=3,
                             embed_size=512)  # train model
        embeddings_node2vec = model_node2vec.get_embeddings()  # get embedding vectors
        G_disturbed = embedding_graph(G_disturbed, embeddings_node2vec)
        G_disturbed = type_code_graph(G_disturbed)
        X_train, X_test, y_train = prepare_train_test(G_disturbed)
        K.clear_session()
        model = get_mlp_deep(512, 4)
        model.fit(X_train, y_train, epochs=20, batch_size=50)
        y_pred = np.argmax(model.predict(X_test), axis=1)
        pd.Series(y_pred).to_csv('{}/pred/pred_{}_{}_{}_{}.csv'.format(path,
                                                                       algorithm, interval, commodity, i), index=False)

    elif algorithm == 'struc2vec':
        model_struc2vec = Struc2Vec(
            G_disturbed, 10, 80, workers=2, verbose=40)  # init model
        model_struc2vec.train(window_size=5, iter=3,
                              embed_size=512)  # train model
        embeddings_struc2vec = model_struc2vec.get_embeddings()  # get embedding vectors
        G_disturbed = embedding_graph(G_disturbed, embeddings_struc2vec)
        G_disturbed = type_code_graph(G_disturbed)
        X_train, X_test, y_train = prepare_train_test(G_disturbed)
        K.clear_session()
        model = get_mlp_deep(512, 4)
        model.fit(X_train, y_train, epochs=20, batch_size=50)
        y_pred = np.argmax(model.predict(X_test), axis=1)
        pd.Series(y_pred).to_csv('{}/pred/pred_{}_{}_{}_{}.csv'.format(path,
                                                                       algorithm, interval, commodity, i), index=False)

    elif algorithm == 'metapath2vec':
        embeddings_metapath2vec = metapath2vec(G_disturbed)
        G_disturbed = embedding_graph(G_disturbed, embeddings_metapath2vec)
        G_disturbed = type_code_graph(G_disturbed)
        X_train, X_test, y_train = prepare_train_test(G_disturbed)
        K.clear_session()
        model = get_mlp_deep(512, 4)
        model.fit(X_train, y_train, epochs=20, batch_size=iteration)
        y_pred = np.argmax(model.predict(X_test), axis=1)
        pd.Series(y_pred).to_csv('{}/pred/pred_{}_{}_{}_{}.csv'.format(path,algorithm,interval,commodity,i), index=False)

    elif algorithm == 'line':
        # init model,order can be ['first','second','all']
        model_line = LINE(G_disturbed, embedding_size=512, order='second')
        model_line.train(batch_size=8, epochs=20, verbose=0)  # train model
        embeddings_line = model_line.get_embeddings()  # get embedding vectors
        G_disturbed = embedding_graph(G_disturbed, embeddings_line)
        G_disturbed = type_code_graph(G_disturbed)
        X_train, X_test, y_train = prepare_train_test(G_disturbed)
        K.clear_session()
        model = get_mlp_deep(512, 4)
        model.fit(X_train, y_train, epochs=20, batch_size=50)
        y_pred = np.argmax(model.predict(X_test), axis=1)
        pd.Series(y_pred).to_csv('{}/pred/pred_{}_{}_{}_{}.csv'.format(path,
                                                                       algorithm, interval, commodity, i), index=False)

    elif algorithm == 'gcn':
        y_pred = gcn(G_disturbed, interval, i)
        pd.Series(y_pred).to_csv('{}/pred/pred_{}_{}_{}_{}.csv'.format(path,
                                                                       algorithm, interval, commodity, i), index=False)

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


#algorithms = ['regularization', 'deep_walk', 'node2vec', 'struc2vec', 'metapath2vec', 'line', 'gcn']
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

"""
# next event label cut MLP
for interval in intervals:
    for commodity in commodities:
        with open(path + commodity + "_" + interval + ".gpickle", "rb") as fh:
            G = pickle.load(fh)
        for i in range(3, 4):
            G_cutted, cutted_dict = next_labels_cut(G, i=i, interval=interval)
            y_true = cutted_dict['event_trend'].neighbor.to_list()
            for idx in range(len(y_true)):
                y_true[idx] = label_codes[y_true[idx]]
            pd.Series(y_true).to_csv('{}/pred/true_{}_{}_{}.csv'.format(path,interval,commodity,i), index=False)
            for algorithm in algorithms:
                print('TEST: {0}, {1}, {2}, {3}'.format(algorithm, interval, commodity, i))
                run_model(G_cutted, cutted_dict, algorithm, interval, commodity, path, i)"""