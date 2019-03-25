from dqn_sl.initializations import *
from dqn_sl.metrics import *
from dqn_sl.input_data import load_synthetic_data, get_epinion_data_node, get_sample_data
import tensorflow as tf
import time as time

# Parameters
learning_rate = 0.001
gamma = 0.5
decay_steps = 100
training_epochs = 1000
episodes = 1000
batch_size = 200
C = 10
epsion = 0.5

H1_num = 1024
H2_num = 512
input_dim = 3
categories = 1
node = 36
# Define placeholders

feature = tf.placeholder(tf.float32, [node, input_dim])
opinion_all = tf.placeholder(tf.float32, [node, node, 3])
cur_opinion = tf.placeholder(tf.float32, [3])
adj_opinion = tf.placeholder(tf.int32, [node, node])
action = tf.placeholder_with_default(0, shape=())
neigh_action = tf.placeholder(tf.int32, [None, 1])
neigh_state = tf.placeholder(tf.int32, [None, 1])


random_s = tf.placeholder_with_default(0.0, shape=())
target_s = tf.placeholder_with_default(0.0, shape=())
start_s = tf.placeholder_with_default(0.0, shape=())
n_statue = tf.placeholder_with_default(0.0, shape=())
path_statue = tf.placeholder_with_default(0.0, shape=())

state = tf.placeholder_with_default(0, shape=())
target_node = tf.placeholder_with_default(0, shape=())
start_node = tf.placeholder_with_default(0, shape=())

ground_truth = tf.placeholder(tf.float32, [3, 1])
W11 = tf.placeholder(tf.float32, [input_dim, H1_num])
W22 = tf.placeholder(tf.float32, [H1_num, H2_num])
W33 = tf.placeholder(tf.float32, [H2_num, categories])
B11 = tf.placeholder(tf.float32, [H1_num])
B22 = tf.placeholder(tf.float32, [H2_num])
B33 = tf.placeholder(tf.float32, [categories])

# Weights
W1 = weight_variable_glorot(input_dim, H1_num, name="w1")
W2 = weight_variable_glorot(H1_num, H2_num, name="w2")
W3 = weight_variable_glorot(H2_num, categories, name="w2")
B1 = weight_variable_bisa([H1_num], name='b1')
B2 = weight_variable_bisa([H2_num], name='b2')
B3 = weight_variable_bisa([categories], name='b3')

mask_f = adj_opinion[state] + adj_opinion[target_node] + adj_opinion[start_node]
mask_f = tf.cast(mask_f, dtype=tf.bool)
mask_f = tf.cast(mask_f, dtype=tf.float32)
mask_f = tf.reshape(mask_f, [-1, 1])

input_1 = feature * mask_f
output = tf.matmul(tf.nn.relu(tf.matmul(tf.nn.relu(tf.matmul(input_1, W1) + B1), W2) + B2), W3) + B3
max_q = tf.reduce_max(tf.gather_nd(output, neigh_state))

# if random_s == True:
#     action_next = action
# else:
#     action_next = tf.argmax(output * mask_output, axis=0)
#     action_next = action_next[0]
action_next = action
action_opinion = opinion_all[state][action_next]

b = tf.cond(start_s > tf.constant(0.0), lambda: action_opinion[0], lambda: cur_opinion[0] * action_opinion[0])
d = tf.cond(start_s > tf.constant(0.0), lambda: action_opinion[1], lambda: cur_opinion[0] * action_opinion[1])
u = tf.cond(start_s > tf.constant(0.0), lambda: action_opinion[2], lambda: action_opinion[1] + cur_opinion[2] + cur_opinion[0] * action_opinion[2])

# b = cur_opinion[0] * action_opinion[0]
# d = cur_opinion[0] * action_opinion[1]
# u = action_opinion[1] + cur_opinion[2] + cur_opinion[0] * action_opinion[2]

# vacuity
reward_old = 1.0 - u
reward_ = tf.cond(n_statue > tf.constant(0.0), lambda: -1.0, lambda: reward_old)
reward_2 = tf.cond(path_statue > tf.constant(0.0), lambda: -1.0, lambda: reward_)
reward = tf.cond(target_s > tf.constant(0.0), lambda: reward_old, lambda: reward_2)

mask_f2 = adj_opinion[action_next] + adj_opinion[target_node] + adj_opinion[start_node]
mask_f2 = tf.cast(mask_f2, dtype=tf.bool)
mask_f2 = tf.cast(mask_f2, dtype=tf.float32)
mask_f2 = tf.reshape(mask_f2, [-1, 1])
input_2 = feature * mask_f2

output2 = tf.matmul(tf.nn.relu(tf.matmul(tf.nn.relu(tf.matmul(input_2, W11) + B11), W22) + B22), W33) + B33
max_q2 = tf.reduce_max(tf.gather_nd(output2, neigh_action))

y_i = tf.cond(target_s > tf.constant(0.0), lambda: reward + 1.0, lambda: tf.add(reward, gamma * max_q2))
score = tf.square(y_i - max_q)

train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=score)

# input data

saver = tf.train.Saver()

# initialization variables
# f, opinion, adj, neigh, E_X = get_epinion_data_node(38)
f, opinion, adj, neigh, E_X = get_sample_data()

for o in range(1):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    init = tf.global_variables_initializer()
    with tf.Session(config=config) as sess:
        sess.run(init)
        for episode in range(episodes):
            p = np.mod(episode, len(E_X))
            start_ = E_X[p][0]
            target_ = E_X[p][1]
            state_n = start_
            path = []
            path.append(state_n)
            current_opinion = np.asarray([0, 0, 0])
            print("start node:", start_, "target node:", target_)
            for epoch in range(training_epochs):
                if np.mod(episode, C) == 0:
                    weight = sess.run([W1, W2, W3, B1, B2, B3])
                # random_state, action_r = greed(epsion, neigh, state_n)
                neigh_s = neigh_index(neigh[state_n])
                output_, input_1_ = sess.run([output, input_1], feed_dict={feature: f, opinion_all: opinion, adj_opinion: adj, state: state_n, target_node: target_, start_node: start_, neigh_state:neigh_s})
                action_r = greed2(epsion, neigh, state_n, output_)
                neigh_a = neigh_index(neigh[action_r])

                if neigh_a == []:
                    if action_r != target_:
                        print("XXXX No path:", path)
                        break
                    else:
                        neigh_a = [[1], [2]]

                if action_r in path:
                    path_s = 1.0
                else:
                    path_s = -1.0
                if state_n == start_:
                    start_statue = 1.0
                else:
                    start_statue = -1.0
                if action_r == target_:
                    target_statue = 1.0
                else:
                    target_statue = -1.0
                if neigh[action_r] == []:
                    neigh_status = 1.0
                else:
                    neigh_status = -1.0
                outs = sess.run([train_step, score, action_next, b, d, u, y_i, action_opinion, max_q2, input_2, output2, reward, y_i],
                                feed_dict={feature: f, opinion_all: opinion, cur_opinion: current_opinion, adj_opinion: adj, state: state_n, target_node: target_, start_node: start_, W11: weight[0], neigh_action: neigh_a, neigh_state:neigh_s,
                                           W22: weight[1], W33: weight[2], B11: weight[3], B22: weight[4], B33: weight[5], action: action_r, target_s: target_statue, n_statue: neigh_status, path_statue: path_s, start_s: start_statue})
                current_opinion = [outs[3], outs[4], outs[5]]
                state_n = action_r
                path.append(state_n)
                print("episode = ", episode, "loss: ", outs[1], "action = ", state_n)
                if state_n == target_:
                    print("find path:", path)
                    break
                if epoch == 20:
                    pass
        # test
        path_test = []
        for i in range(len(E_X)):
            start_n = E_X[i][0]
            target_ = E_X[i][1]
            state_n = start_n
            path1 = []
            path1.append(state_n)
            while state_n != target_:
                output_ = sess.run(output, feed_dict={feature: f, opinion_all: opinion, adj_opinion: adj, state: state_n, target_node: target_})
                action_r = greed2(0.0, neigh, state_n, output_)
                # out = output_[neigh[state_n]]
                # index = np.argsort(out)
                # for p in range(len(index)):
                #
                # m = np.argmax(out)
                # action = neigh[state][m]
                # if action_r != target_:
                #     if neigh[state_n] == []:

                state_n = action_r
                path1.append(state_n)
            print path1
            path_test.append(path1)
