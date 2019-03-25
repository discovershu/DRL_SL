from dqn_sl.initializations import *
from dqn_sl.metrics import *
from dqn_sl.input_data import load_synthetic_data
import tensorflow as tf
import time as time

# Parameters
learning_rate = 0.001
gamma = 0.5
decay_steps = 100
training_epochs = 100
batch_size = 200
C = 10
epsion = 0.1

H1_num = 32
H2_num = 16
input_dim = 3
categories = 1
edge = 25
node = 10
# Define placeholders

feature = tf.placeholder(tf.float32, [node, input_dim])
opinion_all = tf.placeholder(tf.float32, [node, node, 3])
cur_opinion = tf.placeholder(tf.float32, [3])
adj_opinion = tf.placeholder(tf.int32, [node, node])
action = tf.placeholder_with_default(0, shape=())
random_s = tf.placeholder_with_default(0.0, shape=())
target_s = tf.placeholder_with_default(0.0, shape=())
start_s = tf.placeholder_with_default(0.0, shape=())
state = tf.placeholder_with_default(0, shape=())
target_node = tf.placeholder_with_default(0, shape=())
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

mask_f = adj_opinion[state] + adj_opinion[target_node]
mask_f = tf.cast(mask_f, dtype=tf.bool)
mask_f = tf.cast(mask_f, dtype=tf.float32)
mask_f = tf.reshape(mask_f, [-1, 1])
mask_output = adj_opinion[state]
mask_output = tf.reshape(mask_output, [-1, 1])
mask_output = tf.cast(mask_output, dtype=tf.float32)

input_1 = feature * mask_f
output = tf.matmul(tf.nn.relu(tf.matmul(tf.nn.relu(tf.matmul(input_1, W1) + B1), W2) + B2), W3) + B3
max_q = tf.reduce_max(output * mask_output)
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
reward = 1.0 - u

mask_f2 = adj_opinion[action_next] + adj_opinion[target_node]
mask_f2 = tf.cast(mask_f2, dtype=tf.bool)
mask_f2 = tf.cast(mask_f2, dtype=tf.float32)
mask_f2 = tf.reshape(mask_f2, [-1, 1])
input_2 = feature * mask_f2

output2 = tf.matmul(tf.nn.relu(tf.matmul(tf.nn.relu(tf.matmul(input_2, W11) + B11), W22) + B22), W33) + B33
max_q2 = tf.reduce_max(output2 * mask_output)
bbb = tf.constant(2.0)
# if action_next == target_node:
#     y_i = reward
#     bbb = tf.square(bbb)
# else:
#     y_i = reward + gamma * max_q2
y_i = tf.cond(target_s > tf.constant(0.0), lambda: reward, lambda: tf.add(reward, gamma * max_q2))
score = tf.square(y_i - max_q)

train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=score)

# input data

saver = tf.train.Saver()

letter = []
word = []
# initialization variables
for o in range(1):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    init = tf.global_variables_initializer()
    with tf.Session(config=config) as sess:
        sess.run(init)
        f, opinion, adj, neigh = load_synthetic_data()
        Start_nodes = [0, 5, 3]
        Target_nodes = [9, 7, 9]
        for episode in range(3000):
            p = np.mod(episode, 3)
            state_ = Start_nodes[p]
            target_ = Target_nodes[p]
            state_n = state_
            path = []
            path.append(state_n)
            # if episode < 12:
            #     action_r = neigh[state_n][np.mod(episode, len(neigh[state_n]))]
            # else:
            #     output_ = sess.run(output, feed_dict={feature: f, opinion_all: opinion, adj_opinion: adj, state: state_n, target_node: target_})
            #     action_r = greed2(epsion, neigh, state_n, output_)
            # output_ = sess.run(output, feed_dict={feature: f, opinion_all: opinion, adj_opinion: adj, state: state_n, target_node: target_})
            # action_r = greed2(epsion, neigh, state_n, output_)
            # current_opinion = opinion[state_n][action_r]
            # state_n = action_r
            # print("episode = ", episode, "action = ", state_n)
            current_opinion = np.asarray([0, 0, 0])
            for epoch in range(training_epochs):

                if np.mod(episode, C) == 0:
                    weight = sess.run([W1, W2, W3, B1, B2, B3])
                # random_state, action_r = greed(epsion, neigh, state_n)
                output_, input_1_ = sess.run([output, input_1], feed_dict={feature: f, opinion_all: opinion, adj_opinion: adj, state: state_n, target_node: target_})
                action_r = greed2(epsion, neigh, state_n, output_)
                if state_n == state_:
                    statue_s = 1.0
                else:
                    statue_s = -1.0
                if action_r == target_:
                    statue = 1.0
                else:
                    statue = -1.0
                outs = sess.run([train_step, score, action_next, b, d, u, bbb],
                                feed_dict={feature: f, opinion_all: opinion, cur_opinion: current_opinion, adj_opinion: adj, state: state_n, target_node: target_, W11: weight[0],
                                           W22: weight[1], W33: weight[2], B11: weight[3], B22: weight[4], B33: weight[5], action: action_r, target_s: statue})
                current_opinion = [outs[3], outs[4], outs[5]]
                state_n = outs[2]
                path.append(state_n)
                print("episode = ", episode, "loss: ", outs[1], "action = ", state_n)
                if state_n == target_:
                    print path
                    break
                if neigh[state_n] == []:
                    print path
                    break
        # test
        for i in range(len(Start_nodes)):
            state_n = Start_nodes[i]
            target_ = Target_nodes[i]
            path1 = []
            path1.append(state_n)
            while state_n != target_:
                output_ = sess.run(output, feed_dict={feature: f, opinion_all: opinion, adj_opinion: adj, state: state_n, target_node: target_})
                action_r = greed2(0.0, neigh, state_n, output_)
                state_n = action_r
                path1.append(state_n)
            print path1
