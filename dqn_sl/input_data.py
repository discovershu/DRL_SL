import numpy as np
import sys
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import random
import pickle
from collections import Counter


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_data(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features

def get_neigh(adj):
    neigh = []
    for i in range(len(adj)):
        neigh_i = []
        for j in range(len(adj[i])):
            if adj[i][j] == 1:
                neigh_i.append(j)
        neigh.append(neigh_i)
    return neigh

def get_neigh_s(adj):
    adj = adj + adj.T
    neigh = []
    for i in range(len(adj)):
        neigh_i = []
        for j in range(len(adj[i])):
            if adj[i][j] > 0:
                neigh_i.append(j)
        neigh.append(neigh_i)
    return neigh


def get_random_op():
    num = random.sample(range(10), 3)
    num = np.asarray(num) + 1.0
    num = [num[0] / np.sum(num), num[1] / np.sum(num), num[2] / np.sum(num)]
    return num

def load_synthetic_data():
    random.seed(33333)
    adj = np.asarray([[0, 0, 0, 1, 1, 1, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                      [0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
                      [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    opinion = np.zeros([10, 10, 3])
    neigh = get_neigh(adj)
    for i in range(10):
        for j in range(10):
            if adj[i][j] == 1:
                opinion[i][j] = get_random_op()
    f = np.zeros([10, 3])
    for i in range(10):
        neigh_i = neigh[i]
        op = []
        for n in neigh_i:
            op.append(opinion[i][n])
        if op:
            f[i] = np.mean(op, axis=0)
    adj = adj + np.eye(10)
    return f, opinion, adj, neigh


def get_alphabeta(ome, W=2.0, a=0.5):
    r = ome[3]
    s = ome[4]
    alpha = r + W * a
    beta = s + W * (1 - a)
    return [alpha, beta]

def get_omega_obs(obs):
    W = 2.0
    r = Counter(obs)[1]
    s = Counter(obs)[0]
    u = W / (W + r + s)
    b = r / (W + r + s)
    d = s / (W + r + s)
    return [b, d, u, r, s]


def find_direct_neigh_edge(E, i, E_X):
    neigh = []
    nodes = E[i]
    for j in range(len(E)):
        if j != i:
            if E[j] not in E_X:
                if nodes[1] == E[j][0]:
                    neigh.append(j)
            # elif nodes[0] == E[j][1]:
            #     neigh.append(j)
    return neigh

def find_direct_neigh_node(E, i, E_X, V):
    neigh = []
    node = V[i]
    # E = np.asarray(E)
    for j in range(len(E)):
        if E[j] not in E_X:
            if node == E[j][0]:
                neigh.append(E[j][1])
    return neigh


def get_epinion_data_old(T):
    pkl_file = open("/network/rit/lab/ceashpc/xujiang/eopinion_data/tmp/nodes-1000-T-38-rate-0.1-testratio-0.2-swaprate-0.1-realization-0-data-X.pkl")
    [V, E, Obs, E_X] = pickle.load(pkl_file)
    t_Obs = {e: e_Obs[0:T] for e, e_Obs in Obs.items()}
    omega_rs = {}
    opinion = {}
    for edge in E:
        omega_rs[edge] = get_omega_obs(t_Obs[edge])
        opinion[edge] = get_alphabeta(omega_rs[edge])
    belief = np.zeros(len(E))
    uncertain = np.zeros(len(E))
    test_index = []
    for i in range(len(E)):
        belief[i] = omega_rs[E[i]][0]
        uncertain[i] = omega_rs[E[i]][2]
        if E[i] in E_X:
            test_index.append(i)
    return belief, uncertain, test_index


def get_epinion_data(T):
    pkl_file = open("/network/rit/lab/ceashpc/xujiang/eopinion_data/tmp/nodes-500-T-38-rate-0.1-testratio-0.2-swaprate-0.1-realization-0-data-X.pkl")
    [V, E, Obs, E_X] = pickle.load(pkl_file)
    t_Obs = {e: e_Obs[0:T] for e, e_Obs in Obs.items()}
    omega_rs = {}
    opinion = {}
    f = []
    for edge in E:
        omega_rs[edge] = get_omega_obs(t_Obs[edge])
        opinion[edge] = omega_rs[edge][0:3]
        f.append(opinion[edge])
    # input_file = open("./data/neigh_epinoin_1000.pkl")
    # neigh = pickle.load(input_file)
    # adj = np.load("./data/adj_epinoin_1000.npy")
    neigh, adj = get_neigh_adj()
    return np.asarray(f), opinion, adj, neigh, E, E_X

def get_epinion_data_node(T):
    pkl_file = open("/network/rit/lab/ceashpc/xujiang/eopinion_data/tmp/nodes-1000-T-38-rate-0.1-testratio-0.2-swaprate-0.1-realization-0-data-X.pkl")
    [V, E, Obs, E_X] = pickle.load(pkl_file)
    t_Obs = {e: e_Obs[0:T] for e, e_Obs in Obs.items()}
    omega_rs = {}
    opinion = {}
    f = []
    # E_X = np.load("./data/node/new_EX_1000.npy")
    for edge in E:
        omega_rs[edge] = get_omega_obs(t_Obs[edge])
        opinion[edge] = omega_rs[edge][0:3]
    neigh, adj = get_node_neigh_adj(V, E, E_X)
    # input_file = open("./data/node/neigh_epinion_500.pkl")
    # neigh = pickle.load(input_file)
    # adj = np.load("./data/node/adj_epinion_500.npy")
    opinion_all = np.zeros([len(V), len(V), 3])
    for i in V:
        neigh_i = neigh[i]
        op = []
        for n in neigh_i:
            edge_i = (i, n)
            op.append(opinion[edge_i])
            opinion_all[i][n] = opinion[edge_i]
        if op == []:
            f.append(np.asarray([0.0, 0.0, 0.0]))
        else:
            f.append(np.mean(op, axis=0))
        ## opinion
    E_X = np.load("./data/node/new_EX_1000.npy")
    E_X = E_X.tolist()
    adj = np.asarray(adj + np.eye(len(adj)), dtype=int)
    return np.asarray(f), opinion_all, adj, neigh, E_X


def get_epinion_data_policy(T):
    pkl_file = open("/network/rit/lab/ceashpc/xujiang/eopinion_data/tmp/nodes-1000-T-38-rate-0.1-testratio-0.2-swaprate-0.1-realization-0-data-X.pkl")
    [V, E, Obs, E_X] = pickle.load(pkl_file)
    t_Obs = {e: e_Obs[0:T] for e, e_Obs in Obs.items()}
    omega_rs = {}
    opinion = {}
    f = []
    # E_X = np.load("./data/node/new_EX_1000.npy")
    for edge in E:
        omega_rs[edge] = get_omega_obs(t_Obs[edge])
        opinion[edge] = omega_rs[edge][0:3]
    neigh, adj = get_node_neigh_adj(V, E, E_X)
    # input_file = open("./data/node/neigh_epinion_500.pkl")
    # neigh = pickle.load(input_file)
    # adj = np.load("./data/node/adj_epinion_500.npy")
    opinion_all = np.zeros([len(V), len(V), 3])

    E_X = np.load("./data/node/new_EX_1000.npy")
    E_X = E_X.tolist()
    neigh_ = get_neigh_s(adj)
    f = embedding_neigh(neigh_, opinion_all, k=10)
    np.save("./data/node/f_1000.npy", f)
    return np.asarray(f), opinion_all, adj, neigh, E_X

def get_neigh_adj():
    pkl_file = open("/network/rit/lab/ceashpc/xujiang/eopinion_data/tmp/nodes-500-T-38-rate-0.1-testratio-0.2-swaprate-0.1-realization-0-data-X.pkl")
    [V, E, Obs, E_X] = pickle.load(pkl_file)
    neigh = {}
    adj = np.zeros([len(E), len(E)])
    for i in range(len(E)):
        neigh_i = find_direct_neigh_edge(E, i, E_X)
        neigh[E[i]] = neigh_i
        for j in neigh_i:
            adj[i][j] = 1
    return neigh, adj


def get_node_neigh_adj(V, E, E_X):
    # pkl_file = open("/network/rit/lab/ceashpc/xujiang/eopinion_data/tmp/nodes-500-T-38-rate-0.1-testratio-0.2-swaprate-0.1-realization-0-data-X.pkl")
    # [V, E, Obs, E_X] = pickle.load(pkl_file)
    neigh = {}
    adj = np.zeros([len(V), len(V)])
    for i in range(len(V)):
        neigh_i = find_direct_neigh_node(E, i, E_X, V)
        neigh[V[i]] = neigh_i
        for j in neigh_i:
            adj[i][j] = 1
    # output_file = open("./data/node/neigh_epinion_500.pkl", "wb")
    # pickle.dump(neigh, output_file)
    # np.save("./data/node/adj_epinion_500", adj)
    return neigh, adj


def get_sample_data():
    adj = np.load("./data/sample/sub1_adj.npy")
    opinion = np.load("./data/sample/sub1_opinion.npy")
    opinion = opinion[:, :, :3]
    neigh = get_neigh(adj)
    N = len(adj)
    f = np.zeros([N, 3])
    for i in range(N):
        neigh_i = neigh[i]
        op = []
        for n in neigh_i:
            op.append(opinion[i][n])
        if op:
            f[i] = np.mean(op, axis=0)
    adj = adj + np.eye(N)
    E_X = [(0, 35)]
    return f, opinion, adj, neigh, E_X


def get_sample_data_s():
    adj = np.load("/network/rit/lab/ceashpc/xujiang/project/DQN_SL2/dqn_sl/data/sample/sub1_adj.npy")
    opinion = np.load("/network/rit/lab/ceashpc/xujiang/project/DQN_SL2/dqn_sl/data/sample/sub1_opinion.npy")
    opinion = opinion[:, :, :3]
    neigh_ = get_neigh_s(adj)
    neigh = get_neigh(adj)
    N = len(adj)
    f = embedding_neigh(neigh_, opinion, k=5)
    # adj = adj + np.eye(N)
    E_X = [(0, 35)]
    return f, opinion, adj, neigh, E_X


def embedding_neigh(neigh, opinion, k):
    emb = np.zeros([len(neigh), k*3])
    for i in range(len(emb)):
        p = i
        emb_i = []
        neigh_0 = []
        neigh_0.append(neigh[i])
        ppp = []
        j = 0
        while len(emb_i) < k:
            for n_i in neigh_0[j]:
                ppp.append(n_i)
                if np.sum(opinion[p][n_i]) > 0:
                    emb_i.append(opinion[p][n_i])
                neigh_0.append(neigh[n_i])
            p = ppp[j]
            j += 1
        emb_i = emb_i[:k]
        emb[i] = np.reshape(emb_i, [-1])
    return emb
if __name__ == '__main__':
    # get_epinion_data(38)
    # neigh, adj = get_neigh_adj()
    # output_file = open("./data/neigh_epinoin_1000.pkl", "wb")
    # pickle.dump(neigh, output_file)
    # np.save("./data/adj_epinoin_1000", adj)
    # input_file = open("./data/neigh_epinoin_1000.pkl")
    # n = pickle.load(input_file)
    # get_epinion_data_node(38)
    # pkl_file = open("/network/rit/lab/ceashpc/xujiang/eopinion/data/nodes-500-rate-0.2-testratio-0.2-swaprate-0.05-realization-0-data-X.pkl")
    # [V, E, Obs, E_X] = pickle.load(pkl_file)
    # get_epinion_data_node(38)
    get_epinion_data_policy(38)
    print(1)