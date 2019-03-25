import tensorflow as tf
import numpy as np
from dqn_sl.input_data import get_sample_data

def uncertain_op_m(w1, w2):
    b1 = w1[0]
    d1 = w1[1]
    u1 = w1[2]
    a1 = w1[3]

    b2 = w2[0]
    d2 = w2[1]
    u2 = w2[2]
    a2 = w2[3]

    b = b1 * b2
    d = b1 * d2
    u = d1 + u1 + b1 * u2
    a = a2

    w = [b, d, u, a]
    return w


def uncertain_op_m2(w1, w2):
    b1 = w1[0]
    d1 = w1[1]
    u1 = w1[2]

    b2 = w2[0]
    d2 = w2[1]
    u2 = w2[2]

    b = b1 * b2
    d = b1 * d2
    u = d1 + u1 + b1 * u2

    w = [b, d, u]
    return w

def uncertain_op_add(w1, w2):
    b1 = w1[0]
    d1 = w1[1]
    u1 = w1[2]
    a1 = w1[3]

    b2 = w2[0]
    d2 = w2[1]
    u2 = w2[2]
    a2 = w2[3]

    epsion = u1 + u2 - u1 * u2

    b = (b1 * u2 + b2 * u1) / epsion
    d = (d1 * u2 + d2 * u1) / epsion
    u = u1 * u2 / epsion
    a = a1

    w = [b, d, u, a]
    return w


def greed(epsion, neigh, state):
    rand_e = np.random.randint(1, 100, 1)
    if epsion * 100 > rand_e[0]:
        statue = True
    else:
        statue = False
    action = np.random.choice(neigh[state], 1)[0]
    return statue, action

def greed2(epsion, neigh, state, output):
    rand_e = np.random.randint(1, 100, 1)
    if epsion * 100 > rand_e[0]:
        action = np.random.choice(neigh[state], 1)[0]
    else:
        out = output[neigh[state]]
        m = np.argmax(out)
        action = neigh[state][m]
    return action

def neigh_index(neigh):
    ll = []
    for n in neigh:
        ll.append([n])
    return ll


def get_path_opinion(path, opinion):
    cu_op = opinion[path[0], path[1]]
    for i in range(1, len(path)-1):
        a = path[i]
        b = path[i+1]
        op2 = opinion[a][b]
        cu_op = uncertain_op_m2(cu_op, op2)
    return cu_op


if __name__ == '__main__':
    # path = [0, 1, 0, 1, 0, 1, 2, 35]
    # f, opinion, adj, neigh, E_X = get_sample_data()
    # pred_op = get_path_opinion(path, opinion)
    a = np.ones([10, 3])
    b = a[:, 2]
    c = np.argmax(b)
    print(1)