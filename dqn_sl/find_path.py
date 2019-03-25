import numpy as np
import pickle
import networkx as nx
import matplotlib.pyplot as plt


def Convert(tup, di):
    for a, b in tup:
        di.setdefault(a, []).append(b)
    return di


def find_all_paths(graph, start, end, path=[]):
    path = path + [start]
    if start == end:
        return [path]
    if start not in graph:
        return []
    paths = []
    for node in graph[start]:
        if node not in path:
            newpaths = find_all_paths(graph, node, end, path)
            for newpath in newpaths:
                paths.append(newpath)
    return paths


#
# # test = dict()
# # for i in range(len(data[1])):
# #     paths = find_all_paths(graph, data[1][i][0], data[1][i][1])
# #     test[i]=len(paths)
# #
# #     number = []
# #     for j in range(len(paths)):
# #         for m in range(len(paths[j])):
# #             if paths[j][m] not in number:
# #                 number.append(paths[j][m])
# #     print(i, " ", test[i]," ",len(number))
#
#
# b_test = find_all_paths(graph, 29, 124)
# number = []
# for j in range(len(b_test)):
#     for m in range(len(b_test[j])):
#         if b_test[j][m] not in number:
#             number.append(b_test[j][m])
# startnode = number.index(29)
# endnode = number.index(124)
# number[0], number[startnode] = number[startnode], number[0]
# number[len(number)-1], number[endnode] = number[endnode], number[len(number)-1]
#
# nodedict = dict()
# for index,item in enumerate(number):
#     nodedict[index] = item
# final = []
# temp = []
# for i in range(len(b_test)):
#     for j in range(len(b_test[i])):
#         temp.append(list(nodedict.values()).index(b_test[i][j]))
#     final.append(temp)
#     print(temp)
#     temp = []
#
# #get all opinion
# r=0
# s=0
# w=2
# opinionall = dict()
# for item in data[2]:
#     for j in range(len(data[2][item])):
#         if data[2][item][j]==1:
#             r+=1
#         if data[2][item][j]==0:
#             s+=1
#     b = r/(r+s+w)+0.0001
#     d = s/(r+s+w)+0.0001
#     u = w/(r+s+w)+0.0001
#     opinionall[item]=[b,d,u,0.5]
#     r=0
#     s=0
#
# #get adj and opinion
# adj = np.zeros((len(number),len(number)), dtype=int)
# opinion = np.zeros([len(number), len(number), 4])
#
# for i in range(len(number)):
#     for j in range(len(number)):
#         if (nodedict[i],nodedict[j]) in data[1]:
#             adj[i,j] = 1
#             opinion[i,j]=opinionall[nodedict[i],nodedict[j]]
# adj[0,35]=0
# opinion[0,35] = [0.,0.,0.,0.]
#
# np.save('C:\\Users\\Shu\\Desktop\\totalnode_36_pathall_8_node29_to_124_adj.npy',adj)
# np.save('C:\\Users\\Shu\\Desktop\\totalnode_36_pathall_8_node29_to_124_opinion.npy',opinion)
#
# # G=nx.from_numpy_matrix(adj)
# #
# # G=nx.DiGraph()
# # mylist = list(range(36))
# # G.add_nodes_from(mylist)
# # for i in range(len(adj)):
# #     for j in range(len(adj[0])):
# #         if adj[i,j]==1:
# #             G.add_edge(i,j)
# #
# # nx.draw_networkx(G,arrows=True,with_labels = True, pos=nx.random_layout(G),node_size = 200, font_size = 10)
# # plt.show()
#
#
# # G1 = nx.generators.directed.random_k_out_graph(20, 3, 0.8, self_loops=False)
# # G2 = nx.generators.directed.scale_free_graph(20)
# # G3 = nx.generators.directed.gn_graph(20)
# # G4 = nx.generators.directed.gnr_graph(20,0.5)
# # G5 = nx.generators.directed.gnc_graph(20)
# # nx.draw(G1,with_labels = True)
# # plt.show()
# # nx.draw(G2,with_labels = True)
# # plt.show()
# # nx.draw(G3,with_labels = True)
# # plt.show()
# # nx.draw(G4,with_labels = True)
# # plt.show()
# # nx.draw(G5,with_labels = True)
# # plt.show()
#
#
#
# print("shu")

if __name__ == '__main__':
    with open("/network/rit/lab/ceashpc/xujiang/eopinion_data/tmp/nodes-1000-T-38-rate-0.1-testratio-0.2-swaprate-0.1-realization-0-data-X.pkl", 'rb') as f:
        [V, E, Obs, E_X] = pickle.load(f)
        # E_X = E_X[:100]
        # E_X = np.load("./data/node/new_EX_1000.npy")
        for e in E_X:
            E.remove(e)
        graph = {}
        Convert(E, graph)
        new_EX = []
        for e in E_X:
            start = e[0]
            end = e[1]
            paths = find_all_paths(graph, start, end, path=[])
            if len(paths) > 1:
                new_EX.append(e)
                print paths
            # if paths == []:
            #     pass
            # else:
            #     new_EX.append(e)
        np.save("./data/node/new_EX_1000.npy", new_EX)
        print len(new_EX)
