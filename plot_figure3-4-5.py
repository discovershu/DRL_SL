import numpy as np
import matplotlib.pyplot as plt





def epinion_path_1000():
    plt.style.use('seaborn-whitegrid') # seaborn-whitegrid, ggplot
    x = [1, 2, 3]

    SL = [0.311, 0.291, 0.282]
    # CSL = [0.150, 0.15, 0.15]
    RL_V = [0.271, 0.263, 0.262]
    RL_M = [0.271, 0.265, 0.263]
    RL_D = [0.31, 0.294, 0.293]
    plt.plot(x, SL, linewidth=2, color='tomato', linestyle='-', marker='o', ms=7, label='SL')
    # plt.plot(x, CSL, linewidth=2, color='royalblue', linestyle='-', marker='s', label='CSL', ms=7)
    plt.plot(x, RL_M, linewidth=2, color='darkorange', linestyle='-', marker='p', label='SL-DRL-M', ms=7) # lightseagreen
    plt.plot(x, RL_V, linewidth=2, color='lightseagreen', linestyle='-', marker='*', label='SL-DRL-V', ms=7) #darkorange
    plt.plot(x, RL_D, linewidth=2, color='slateblue', linestyle='-', marker='X', label='SL-DRL-D', ms=7)

    plt.legend(loc=0, fontsize=16, ncol=1)

    plt.xlabel('Number of Paths', fontsize=20, fontweight = "bold")
    plt.ylabel('EB-MSE', fontsize=20, fontweight = "bold")

    plt.yticks(fontsize=15, fontweight = "bold")
    plt.xticks(fontsize=15, fontweight = "bold")
    plt.ylim((0.14, 0.32))
    my_y_ticks = np.arange(0.15, 0.32, 0.05)
    plt.yticks(my_y_ticks)
    plt.xticks(x)
    plt.savefig("C:/Users/Shu/Desktop/Fusion19/epinion_path_1000.png", dpi=800)
    # plt.show()

    return


def epinion_path_1000_p():
    plt.style.use('seaborn-whitegrid') # seaborn-whitegrid, ggplot
    x = [1, 2, 3]

    SL = [0.923, 0.962, 0.962]
    # CSL = [0.960, 0.960, 0.960]
    RL_V = [0.962, 0.962, 0.962]
    RL_M = [0.962, 0.962, 0.962]
    RL_D = [0.884, 0.923, 0.923]
    plt.plot(x, SL, linewidth=2, color='tomato', linestyle='-', marker='o', ms=7, label='SL')
    # plt.plot(x, CSL, linewidth=2, color='royalblue', linestyle='-', marker='s', label='CSL', ms=7)
    plt.plot(x, RL_M, linewidth=2, color='darkorange', linestyle='-', marker='p', label='SL-DRL-M', ms=7) # lightseagreen
    plt.plot(x, RL_V, linewidth=2, color='lightseagreen', linestyle='-', marker='*', label='SL-DRL-V', ms=7) #darkorange
    plt.plot(x, RL_D, linewidth=2, color='slateblue', linestyle='-', marker='X', label='SL-DRL-D', ms=7)

    plt.legend(loc=0, fontsize=15, ncol=1)

    plt.xlabel('Number of Paths', fontsize=20, fontweight = "bold")
    plt.ylabel('Precision Accuracy', fontsize=20, fontweight = "bold")

    plt.yticks(fontsize=15, fontweight = "bold")
    plt.xticks(fontsize=15, fontweight = "bold")
    plt.ylim((0.88, 1.01))
    my_y_ticks = np.arange(0.88, 1.01, 0.02)
    plt.yticks(my_y_ticks)
    plt.xticks(x)
    plt.savefig("C:/Users/Shu/Desktop/Fusion19/epinion_path_1000_p.png", dpi=800)
    # plt.show()

    return

def epinion_data_size():
    plt.style.use('seaborn-whitegrid') # seaborn-whitegrid, ggplot
    x = [1, 2, 3]

    SL = [0.384, 0.291, 0.102]
    # CSL = [0.207, 0.150, 0.380]
    RL_V = [0.350, 0.263, 0.097]
    RL_M = [0.350, 0.265, 0.101]
    RL_D = [0.387, 0.294, 0.100]
    plt.plot(x, SL, linewidth=2, color='tomato', linestyle='-', marker='o', ms=7, label='SL')
    # plt.plot(x, CSL, linewidth=2, color='royalblue', linestyle='-', marker='s', label='CSL', ms=7)
    plt.plot(x, RL_M, linewidth=2, color='darkorange', linestyle='-', marker='p', label='SL-DRL-M', ms=7) # lightseagreen
    plt.plot(x, RL_V, linewidth=2, color='lightseagreen', linestyle='-', marker='*', label='SL-DRL-V', ms=7) #darkorange
    plt.plot(x, RL_D, linewidth=2, color='slateblue', linestyle='-', marker='X', label='SL-DRL-D', ms=7)

    plt.legend(loc=0, fontsize=16, ncol=1)

    plt.xlabel('Graph Size', fontsize=20, fontweight = "bold")
    plt.ylabel('EB-MSE', fontsize=20, fontweight = "bold")
    x_tick = ['500', '1000', '5000']

    plt.yticks(fontsize=15, fontweight = "bold")
    plt.xticks(fontsize=15, fontweight = "bold")
    plt.ylim((0.08, 0.41))
    my_y_ticks = np.arange(0.05, 0.41, 0.1)
    plt.yticks(my_y_ticks)
    plt.xticks(x, x_tick, rotation=0, fontsize=12)
    plt.savefig("C:/Users/Shu/Desktop/Fusion19/epinion_size.png", dpi=800)
    # plt.show()

    return


def epinion_size_p():
    plt.style.use('seaborn-whitegrid') # seaborn-whitegrid, ggplot
    x = [1, 2, 3]

    SL = [1.0, 0.923, 0.75]
    # CSL = [0.81, 0.96, 0.598]
    RL_V = [1.0, 0.962, 0.745]
    RL_M = [1.0, 0.962, 0.746]
    RL_D = [1.0, 0.923, 0.783]
    plt.plot(x, SL, linewidth=2, color='tomato', linestyle='-', marker='o', ms=7, label='SL')
    # plt.plot(x, CSL, linewidth=2, color='royalblue', linestyle='-', marker='s', label='CSL', ms=7)
    plt.plot(x, RL_M, linewidth=2, color='darkorange', linestyle='-', marker='p', label='SL-DRL-M', ms=7) # lightseagreen
    plt.plot(x, RL_V, linewidth=2, color='lightseagreen', linestyle='-', marker='*', label='SL-DRL-V', ms=7) #darkorange
    plt.plot(x, RL_D, linewidth=2, color='slateblue', linestyle='-', marker='X', label='SL-DRL-D', ms=7)

    plt.legend(loc=0, fontsize=16, ncol=1)

    plt.xlabel('Graph Size', fontsize=20, fontweight = "bold")
    plt.ylabel('Precision Accuracy', fontsize=20, fontweight = "bold")
    x_tick = ['500', '1000', '5000']

    plt.yticks(fontsize=15, fontweight = "bold")
    plt.xticks(fontsize=15, fontweight = "bold")
    plt.ylim((0.55, 1.01))
    my_y_ticks = np.arange(0.6, 1.01, 0.1)
    plt.yticks(my_y_ticks)
    plt.xticks(x, x_tick, rotation=0, fontsize=12)
    plt.savefig("C:/Users/Shu/Desktop/Fusion19/epinion_size_p.png", dpi=800)
    # plt.show()

    return

def epinion_T():
    plt.style.use('seaborn-whitegrid') # seaborn-whitegrid, ggplot
    x = [1, 2, 3, 4, 5]

    SL = [0.192,0.226,0.238,0.270,0.291]
    # CSL = [0.271,0.180,0.229,0.172,0.150]
    RL_V = [0.190,0.214,0.223,0.251,0.263]
    RL_M = [0.191,0.215,0.225,0.254,0.263]
    RL_D = [0.193,0.213,0.235,0.269,0.294]
    plt.plot(x, SL, linewidth=2, color='tomato', linestyle='-', marker='o', ms=7, label='SL')
    # plt.plot(x, CSL, linewidth=2, color='royalblue', linestyle='-', marker='s', label='CSL', ms=7)
    plt.plot(x, RL_M, linewidth=2, color='darkorange', linestyle='-', marker='p', label='SL-DRL-M', ms=7) # lightseagreen
    plt.plot(x, RL_V, linewidth=2, color='lightseagreen', linestyle='-', marker='*', label='SL-DRL-V', ms=7) #darkorange
    plt.plot(x, RL_D, linewidth=2, color='slateblue', linestyle='-', marker='X', label='SL-DRL-D', ms=7)

    plt.legend(loc=0, fontsize=16, ncol=1)

    plt.xlabel('Degree of Vacuity', fontsize=20, fontweight = "bold")
    plt.ylabel('EB-MSE', fontsize=20, fontweight = "bold")
    x_tick = ['25%-100%', '17%-100%', '13%-100%', '8%-100%', '5%-100%']

    plt.yticks(fontsize=15, fontweight = "bold")
    plt.xticks(fontsize=15, fontweight = "bold")
    plt.ylim((0.14, 0.31))
    my_y_ticks = np.arange(0.15, 0.31, 0.05)
    plt.yticks(my_y_ticks)
    plt.xticks(x, x_tick, rotation=0, fontsize=12)
    plt.savefig("C:/Users/Shu/Desktop/Fusion19/epinion_T.png", dpi=800)
    # plt.show()

    return


def epinion_T_p():
    plt.style.use('seaborn-whitegrid') # seaborn-whitegrid, ggplot
    x = [1, 2, 3, 4, 5]

    SL = [0.846,0.962,0.846,0.884,0.923]
    # CSL = [0.691,0.845,0.730,0.883,0.960]
    RL_V = [0.923,0.962,0.923,0.962,0.962]
    RL_M = [0.923,0.962,0.923,0.923,0.962]
    RL_D = [0.846,0.962,0.807,0.923,0.923]
    plt.plot(x, SL, linewidth=2, color='tomato', linestyle='-', marker='o', ms=7, label='SL')
    # plt.plot(x, CSL, linewidth=2, color='royalblue', linestyle='-', marker='s', label='CSL', ms=7)
    plt.plot(x, RL_M, linewidth=2, color='darkorange', linestyle='-', marker='p', label='SL-DRL-M', ms=7) # lightseagreen
    plt.plot(x, RL_V, linewidth=2, color='lightseagreen', linestyle='-', marker='*', label='SL-DRL-V', ms=7) #darkorange
    plt.plot(x, RL_D, linewidth=2, color='slateblue', linestyle='-', marker='X', label='SL-DRL-D', ms=7)

    plt.legend(loc=0, fontsize=16, ncol=1)

    plt.xlabel('Degree of Vacuity', fontsize=20, fontweight = "bold")
    plt.ylabel('Precision Accuracy', fontsize=20, fontweight = "bold")
    x_tick = ['25%-100%', '17%-100%', '13%-100%', '8%-100%', '5%-100%']

    plt.yticks(fontsize=15, fontweight = "bold")
    plt.xticks(fontsize=15, fontweight = "bold")
    plt.ylim((0.6, 1.0))
    my_y_ticks = np.arange(0.65, 1.0, 0.1)
    plt.yticks(my_y_ticks)
    plt.xticks(x, x_tick, rotation=0, fontsize=12)
    plt.savefig("C:/Users/Shu/Desktop/Fusion19/epinion_T_accuracy.png", dpi=800)
    # plt.show()

    return


def time_rpinion_plt():
    RL_V = [56, 56, 56, 56]
    RL_M = [2.3, 2.3, 2.3, 2.3]
    CSL = [107, 221, 1345]
    SL = [2.3, 4.41, 6.47, 8.6]

    # GCN = np.log([28, 32, 113, 125])
    # CSL = np.log([54, 144, 162, 358])
    # SL = np.log([29, 79, 578, 1445])

    a = range(3)

    N = 3

    ind = np.arange(N)  # the x locations for the groups
    width = 0.1  # the width of the bars

    plt.bar(ind, RL_V, width, color='c', label='SL-DRL-V', edgecolor='white', facecolor='orangered')
    plt.bar(ind + width * 1, RL_M, width, color='C0', label='SL-DRL-V', edgecolor='white', facecolor='orange')
    plt.bar(ind + width * 2, CSL, width, color='k', label='CSL', edgecolor='white', facecolor='yellowgreen')
    plt.bar(ind + width * 3, SL, width, color='g', label='SL', edgecolor='white', facecolor='dodgerblue')

    #
    # plt.plot(a, SL, 'bo-', ms=9, label='SL')
    # plt.plot(a, CSL, 'kX-', ms=10, label='CSL')
    # plt.plot(a, GCN_Semi, 'cs-', ms=8, label='GCN-Semi')
    # plt.plot(a, GCN_opinion, 'gH-', ms=10, label='GCN-opinion')
    # plt.plot(a, GCN_AVE_opinion, 'r*-', ms=12, label='GCN-AVE-opinion')
    plt.legend(loc='upper left', fontsize=10, ncol=2)

    plt.xlabel('Graph Size', fontsize=15)
    plt.ylabel('Log Computation Time (Seconds)', fontsize=15)

    names = ['500', '1000', '5000']
    plt.xticks(a, names, rotation=0, fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim((0.0, 73))
    my_y_ticks = np.arange(0.0, 73, 10)
    # plt.yticks(my_y_ticks)
    # plt.figure(figsize=(60, 10))
    # plt.savefig("C:/Users/Xujiang/Desktop/fusion19/epinion_time.png", dpi=800)
    # fig = matplotlib.pyplot.gcf()
    # fig.set_size_inches(10, 8)
    plt.show()

if __name__ == '__main__':
    # time_rpinion_plt()
    # epinion_path_1000()
    # epinion_path_1000_p()
    # epinion_data_size()
    epinion_size_p()
    # epinion_T()
    # epinion_T_p()