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

def difference(start, end, interval):
    x = end - start
    r = {
            'week': int(x / np.timedelta64(1, 'W')),
            'fortnight': int(x / np.timedelta64(2, 'W')),
            'month': int(x / np.timedelta64(1, 'M'))
        }
    return r[interval]

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
def get_metric(metric, true, pred):
    if metric == 'acc':
        return accuracy_score(list(true), list(pred))
    elif metric == 'precision':
        return precision_score(list(true), list(pred), average='macro')
    elif metric == 'recall':
        return recall_score(list(true), list(pred), average='macro')
    elif metric == 'f1':
        return f1_score(list(true), list(pred), average='macro')

def regularization(G, dim=512, embedding_feature: str = 'embedding', iterations=15, mi=0.85):
    nodes = []
    # inicializando vetor f para todos os nodes
    for node in G.nodes():
        G.nodes[node]['f'] = np.array([0.0]*dim)
        if embedding_feature in G.nodes[node]:
            G.nodes[node]['f'] = G.nodes[node][embedding_feature]*1.0
        nodes.append(node)
    pbar = tqdm(range(0, iterations))
    for iteration in pbar:
        random.shuffle(nodes)
        energy = 0.0
        # percorrendo cada node
        for node in nodes:
            f_new = np.array([0.0]*dim)
            f_old = np.array(G.nodes[node]['f'])*1.0
            sum_w = 0.0
            # percorrendo vizinhos do onde
            for neighbor in G.neighbors(node):
                w = 1.0
                if 'weight' in G[node][neighbor]:
                    w = G[node][neighbor]['weight']
                w /= np.sqrt(G.degree[neighbor])
                f_new = f_new + w*G.nodes[neighbor]['f']
                sum_w = sum_w + w
            if sum_w == 0.0: sum_w = 1.0
            f_new /= sum_w
            G.nodes[node]['f'] = f_new*1.0
            if embedding_feature in G.nodes[node]:
                G.nodes[node]['f'] = G.nodes[node][embedding_feature] * \
                    mi + G.nodes[node]['f']*(1.0-mi)
            energy = energy + np.linalg.norm(f_new-f_old)
        iteration = iteration + 1
        message = 'Iteration '+str(iteration)+' | Energy = '+str(energy)
        pbar.set_description(message)
    return G

# put embeddings on graph
def embedding_graph(G, embeddings, embedding_feature='f'):
    for key, value in embeddings.items():
        G.nodes[key][embedding_feature] = value
    return G

def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)

def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)

import layers.graph as lg
import utils.sparse as us
from random import randint

def gcn(G, interval, i, label_feature='trend', type_feature='node_type', event_feature='event', label_number_feature='type_code', embedding_feature='f'):
    node_list = []
    for node in G.nodes():
      node_list.append(node)
    
    label_codes = {
            'big_down': 0,
            'down': 1,
            'up': 2,
            'big_up': 3,
        }
    for node in node_list:
        G.nodes[node][label_number_feature] = -1
        if G.nodes[node][type_feature] == event_feature:
            for edge in G.neighbors(node):
                if G.nodes[edge][type_feature] == label_feature:
                    G.nodes[node][label_number_feature] = label_codes[edge]
        
    adj = nx.adj_matrix(G,nodelist=node_list)
  
    # Get important parameters of adjacency matrix
    n_nodes = adj.shape[0]
  
    # Some preprocessing
    adj_tilde = adj + np.identity(n=adj.shape[0])
    d_tilde_diag = np.squeeze(np.sum(np.array(adj_tilde), axis=1))
    d_tilde_inv_sqrt_diag = np.power(d_tilde_diag, -1/2)
    d_tilde_inv_sqrt = np.diag(d_tilde_inv_sqrt_diag)
    adj_norm = np.dot(np.dot(d_tilde_inv_sqrt, adj_tilde), d_tilde_inv_sqrt)
    adj_norm_tuple = us.sparse_to_tuple(scipy.sparse.coo_matrix(adj_norm))
  
    # Features are just the identity matrix
    feat_x = np.identity(n=adj.shape[0])
    feat_x_tuple = us.sparse_to_tuple(scipy.sparse.coo_matrix(feat_x))
  
    # Preparing train data
    memberships = [m for m in nx.get_node_attributes(G, label_number_feature).values()]
    nb_classes = len(set(memberships))
    targets = np.array([memberships], dtype=np.int32).reshape(-1)
    one_hot_targets = np.eye(nb_classes)[targets]
  
    labels_to_keep = [i for i in range(len(node_list)) if memberships[i] != -1]
  
    y_train = np.zeros(shape=one_hot_targets.shape,
                      dtype=np.float32)
  
    train_mask = np.zeros(shape=(n_nodes,), dtype=np.bool)
    
    for l in labels_to_keep:
        y_train[l, :] = one_hot_targets[l, :]
        train_mask[l] = True
  
    # TensorFlow placeholders
    ph = {
        'adj_norm': tf.sparse_placeholder(tf.float32, name="adj_mat"),
        'x': tf.sparse_placeholder(tf.float32, name="x"),
        'labels': tf.placeholder(tf.float32, shape=(n_nodes, nb_classes)),
        'mask': tf.placeholder(tf.int32)}
  
    l_sizes = [1024, 1024, 512, nb_classes]
    
    name_text = str(interval) + '_' + str(i)
    
    o_fc1 = lg.GraphConvLayer(
        input_dim=feat_x.shape[-1],
        output_dim=l_sizes[0],
        name='fc1_'+name_text,
        activation=tf.nn.tanh)(adj_norm=ph['adj_norm'], x=ph['x'], sparse=True)
  
    o_fc2 = lg.GraphConvLayer(
        input_dim=l_sizes[0],
        output_dim=l_sizes[1],
        name='fc2_'+name_text,
        activation=tf.nn.tanh)(adj_norm=ph['adj_norm'], x=o_fc1)
  
    o_fc3 = lg.GraphConvLayer(
        input_dim=l_sizes[1],
        output_dim=l_sizes[2],
        name='fc3_'+name_text,
        activation=tf.nn.tanh)(adj_norm=ph['adj_norm'], x=o_fc2)
  
    o_fc4 = lg.GraphConvLayer(
        input_dim=l_sizes[2],
        output_dim=l_sizes[3],
        name='fc4_'+name_text,
        activation=tf.identity)(adj_norm=ph['adj_norm'], x=o_fc3)
  
  
    with tf.name_scope('optimizer'):
        loss = masked_softmax_cross_entropy(preds=o_fc4, labels=ph['labels'], mask=ph['mask'])
        accuracy = masked_accuracy(preds=o_fc4, labels=ph['labels'], mask=ph['mask'])
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
        opt_op = optimizer.minimize(loss)
  
    feed_dict_train = {ph['adj_norm']: adj_norm_tuple,
                      ph['x']: feat_x_tuple,
                      ph['labels']: y_train,
                      ph['mask']: train_mask}
  
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
  
    epochs = 20
    save_every = 50
  
    t = time.time()
    embedding_out, preds_out = [], []
    # Train model
    for epoch in range(epochs):
        _, train_loss, train_acc = sess.run(
            (opt_op, loss, accuracy), feed_dict=feed_dict_train)
  
        if True:
            val_loss, val_acc = sess.run((loss, accuracy), feed_dict=feed_dict_train)
  
            # # Print results
            # #print("Epoch:", '%04d' % (epoch + 1),
            #       "train_loss=", "{:.5f}".format(train_loss),
            #       "time=", "{:.5f}".format(time.time() - t))
  
            feed_dict_output = {ph['adj_norm']: adj_norm_tuple,
                                ph['x']: feat_x_tuple}
  
            #embeddings = sess.run(o_fc3, feed_dict=feed_dict_output)
            preds = sess.run(o_fc4, feed_dict=feed_dict_output)
            if epoch + 1 == epochs:
                #embedding_out = embeddings
                preds_out = np.argmax(preds, axis=1)
    y_pred = []
    for idx, node in enumerate(G.nodes()):
        #G.nodes[node][embedding_feature] = embedding_out[idx]
        if G.nodes[node][type_feature] == event_feature:
            if G.nodes[node][label_number_feature] == -1:
                y_pred.append(preds_out[idx])
    return y_pred

from bs4 import BeautifulSoup
def decode_html_text(x):
    x = BeautifulSoup(x, 'html.parser')
    return x.get_text()

from gensim.models import Word2Vec
from stellargraph.data import UniformRandomMetaPathWalk
from stellargraph import StellarGraph

def metapath2vec(graph, dimensions = 512, num_walks = 1, walk_length = 100, context_window_size = 10, 
                           num_iter = 1, workers = 1, node_type='node_type', edge_type='edge_type',
                           user_metapaths=[
                                   ['event','date','event'],['event','what','event'],['event','where','event'],
                                   ['event','who','event'],['event','why','event'],['event','how','event'],
                                   ['event','date','event','trend','event'],['event','what','event','trend','event'],
                                   ['event','where','event','trend','event'],['event','who','event','trend','event'],
                                   ['event','why','event','trend','event'],['event','how','event','trend','event'],
                               ]
                           ):
    s_graph = StellarGraph.from_networkx(graph, node_type_attr=node_type, edge_type_attr=edge_type)
    rw = UniformRandomMetaPathWalk(s_graph)
    walks = rw.run(
        s_graph.nodes(), n=num_walks, length=walk_length, metapaths=user_metapaths
    )
    
    print(f"Number of random walks: {len(walks)}")

    model = Word2Vec(
        walks,
        size=dimensions,
        window=context_window_size,
        min_count=0,
        sg=1,
        workers=workers,
        iter=num_iter,
    )
    
    def get_embeddings(model, graph):
        if model is None:
            print("model not train")
            return {}

        _embeddings = {}
        for word in graph.nodes():
            try:
                _embeddings[word] = model.wv[word]
            except:
                _embeddings[word] = np.zeros(dimensions)

        return _embeddings
    return get_embeddings(model, graph)

def get_code(x, restored):
    restored_value = [elem[1] for elem in restored if elem[0] == x[0]]
    return_dict = {
            'big_down': 0,
            'down': 1,
            'up': 2,
            'big_up': 3,
        }
    return return_dict[x[1]], return_dict[restored_value[0]]

def make_hin(df, 
             id_feature='EventId', date_feature='WeekYear', date_value_feature='Date', 
             what_feature='what', where_feature='where', who_feature='who', why_feature='why', how_feature='how',
             commodities_feature='WeekYearCornTrend'
             ):
    G = nx.Graph()
    for index,row in df.iterrows():
        node_id = 'Event' + str(row[id_feature])
        date_value = row[date_value_feature]
        node_date = row[date_feature]
        node_what = row[what_feature]
        node_where = row[where_feature]
        node_who = row[who_feature]
        node_why = row[why_feature]
        node_how = row[how_feature]
        # label
        node_commodities = row[commodities_feature]
        
        # event <-> date
        G.add_edge(node_id, node_date, edge_type='event_date', edge_value=date_value)
        G.nodes[node_id]['node_type'] = 'event'
        G.nodes[node_date]['node_type'] = 'date'
        # event <-> what
        if node_what is not None:
            G.add_edge(node_id, node_what, edge_type='event_what')
            G.nodes[node_what]['node_type'] = 'what'
        # event <-> where
        if node_where is not None:
            G.add_edge(node_id, node_where, edge_type='event_where')
            G.nodes[node_where]['node_type'] = 'where'
        # event <-> who
        if node_who is not None:
            G.add_edge(node_id, node_who, edge_type='event_who')
            G.nodes[node_who]['node_type'] = 'who'
        # event <-> why
        if node_why is not None:
            G.add_edge(node_id, node_why, edge_type='event_why')
            G.nodes[node_why]['node_type'] = 'why'
        # event <-> how
        if node_how is not None:
            G.add_edge(node_id, node_how, edge_type='event_how')
            G.nodes[node_how]['node_type'] = 'how'
        # event <-> trend
        if node_commodities is not None:
            G.add_edge(node_id, node_commodities, edge_type='event_trend')
            G.nodes[node_commodities]['node_type'] = 'trend'
        # embedding
        G.nodes[node_id]['embedding'] = row.embedding
    return G

def difference(start, end, interval):
    x = end - start
    r = {
            'week': int(x / np.timedelta64(1, 'W')),
            'fortnight': int(x / np.timedelta64(2, 'W')),
            'month': int(x / np.timedelta64(1, 'M'))
        }
    return r[interval]

def inner_connections(G, interval='week', embedding_feature='embedding', type_feature='edge_type', desired_type_feature='event_date', value_feature='edge_value', return_type_feature='event_event'):
    edges_to_add = []
    for node1, neighbor1 in G.edges:
        if embedding_feature in G.nodes[node1]:
            if G[node1][neighbor1][type_feature] == desired_type_feature:
                for node2, neighbor2 in G.edges:
                    if embedding_feature in G.nodes[node2]:
                        if G[node2][neighbor2][type_feature] == desired_type_feature:
                            temp_cosine = cosine(G.nodes[node1][embedding_feature], G.nodes[node2][embedding_feature])
                            if temp_cosine <= 0.5 and temp_cosine != 0.0:
                                if abs(difference(G[node1][neighbor1][value_feature], G[node2][neighbor2][value_feature], interval)) <= 3:
                                    edges_to_add.append((node1,node2))
    for new_edge in edges_to_add:
        G.add_edge(new_edge[0],new_edge[1],edge_type=return_type_feature)
    return G

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

from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

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

def period_year(date, interval):
    string = ''
    func_dict = {
            'week': string + str(date.week) + '-' + str(date.year),
            'month': string + str(date.month) + '-' + str(date.year),
        }
    return func_dict[interval]