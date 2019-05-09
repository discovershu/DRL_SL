import numpy as np
import matplotlib.pyplot as plt

f = 20
l = 18
def PA_path_1000():
    plt.style.use('seaborn-whitegrid') # seaborn-whitegrid, ggplot
    x = [1, 2, 3]

    SL = [0.335, 0.332, 0.329]
    CSL = [0.180, 0.180, 0.180]
    RL_V = [0.329, 0.328, 0.328]
    RL_M = [0.329, 0.328, 0.328]
    RL_D = [0.338,0.333,0.333]
    plt.plot(x, SL, linewidth=2, color='tomato', linestyle='-', marker='o', ms=7, label='SL')
    # plt.plot(x, CSL, linewidth=2, color='royalblue', linestyle='-', marker='s', label='CSL', ms=7)
    plt.plot(x, RL_M, linewidth=2, color='darkorange', linestyle='-', marker='p', label='SL-DRL-M', ms=7)  # lightseagreen
    plt.plot(x, RL_V, linewidth=2, color='lightseagreen', linestyle='-', marker='*', label='SL-DRL-V', ms=7)  # darkorange
    plt.plot(x, RL_D, linewidth=2, color='slateblue', linestyle='-', marker='X', label='SL-DRL-D', ms=7)  # darkorange



    plt.legend(loc=0, fontsize=l, ncol=2)

    plt.xlabel('Number of Paths', fontsize=20, fontweight = "bold")
    plt.ylabel('EB-MSE', fontsize=20, fontweight = "bold")

    plt.yticks(fontsize=15, fontweight = "bold")
    plt.xticks(fontsize=15, fontweight = "bold")
    plt.ylim((0.32, 0.341))
    my_y_ticks = np.arange(0.32, 0.341, 0.01)
    plt.yticks(my_y_ticks)
    plt.xticks(x)
    plt.savefig("C:/Users/Shu/Desktop/Fusion19/PA_path_1000.png", dpi=800)
    # plt.show()

    return


def PA_path_1000_p():
    plt.style.use('seaborn-whitegrid') # seaborn-whitegrid, ggplot
    x = [1, 2, 3]

    SL = [0.743, 0.734, 0.727]
    CSL = [0.811, 0.811, 0.811]
    RL_V = [0.755, 0.755, 0.755]
    RL_M = [0.739, 0.739, 0.739]
    RL_D = [0.747, 0.747, 0.747]
    plt.plot(x, SL, linewidth=2, color='tomato', linestyle='-', marker='o', ms=7, label='SL')
    # plt.plot(x, CSL, linewidth=2, color='royalblue', linestyle='-', marker='s', label='CSL', ms=7)
    plt.plot(x, RL_M, linewidth=2, color='darkorange', linestyle='-', marker='p', label='SL-DRL-M', ms=7)  # lightseagreen
    plt.plot(x, RL_V, linewidth=2, color='lightseagreen', linestyle='-', marker='*', label='SL-DRL-V', ms=7)  # darkorange
    plt.plot(x, RL_D, linewidth=2, color='slateblue', linestyle='-', marker='X', label='SL-DRL-D', ms=7)  # darkorange


    plt.legend(loc=0, fontsize=l, ncol=2)

    plt.xlabel('Number of Paths', fontsize=20, fontweight = "bold")
    plt.ylabel('Precision Accuracy', fontsize=20, fontweight = "bold")

    plt.yticks(fontsize=15, fontweight = "bold")
    plt.xticks(fontsize=15, fontweight = "bold")
    plt.ylim((0.70, 0.78))
    my_y_ticks = np.arange(0.70, 0.78, 0.02)
    plt.yticks(my_y_ticks)
    plt.xticks(x)
    plt.savefig("C:/Users/Shu/Desktop/Fusion19/PA_path_1000_p.png", dpi=800)
    # plt.show()

    return



def PA_T():
    plt.style.use('seaborn-whitegrid') # seaborn-whitegrid, ggplot
    x = [1, 2, 3, 4, 5]

    SL = [0.287,0.317,0.332,0.345,0.332]
    CSL = [0.271,0.180,0.330,0.172,0.180]
    RL_V = [0.285,0.315,0.330,0.343,0.328]
    RL_M = [0.285,0.316,0.330,0.342,0.328]
    RL_D = [0.288, 0.318, 0.333, 0.347, 0.333]
    plt.plot(x, SL, linewidth=2, color='tomato', linestyle='-', marker='o', ms=7, label='SL')
    # plt.plot(x, CSL, linewidth=2, color='royalblue', linestyle='-', marker='s', label='CSL', ms=7)
    plt.plot(x, RL_M, linewidth=2, color='darkorange', linestyle='-', marker='p', label='SL-DRL-M', ms=7)  # lightseagreen
    plt.plot(x, RL_V, linewidth=2, color='lightseagreen', linestyle='-', marker='*', label='SL-DRL-V', ms=7)  # darkorange
    plt.plot(x, RL_D, linewidth=2, color='slateblue', linestyle='-', marker='X', label='SL-DRL-D', ms=7)  # darkorange


    plt.legend(loc=0, fontsize=l, ncol=1)

    plt.xlabel('Degree of Vacuity',fontsize=20, fontweight = "bold")
    plt.ylabel('EB-MSE', fontsize=20, fontweight = "bold")
    x_tick = ['25%-100%', '17%-100%', '13%-100%', '8%-100%', '5%-100%']

    plt.yticks(fontsize=15, fontweight = "bold")
    plt.xticks(fontsize=15, fontweight = "bold")
    plt.ylim((0.28, 0.351))
    my_y_ticks = np.arange(0.30, 0.34, 0.02)
    plt.yticks(my_y_ticks)
    plt.xticks(x, x_tick, rotation=0, fontsize=12)
    plt.savefig("C:/Users/Shu/Desktop/Fusion19/PA_T.png", dpi=800)
    # plt.show()

    return


def PA_T_p():
    plt.style.use('seaborn-whitegrid') # seaborn-whitegrid, ggplot
    x = [1, 2, 3, 4, 5]

    SL = [0.801,0.784,0.769,0.746,0.734]
    CSL = [0.691,0.845,0.730,0.883,0.960]
    RL_V = [0.800,0.792,0.758,0.754,0.755]
    RL_M = [0.802,0.793,0.771,0.746,0.735]
    RL_D = [0.802, 0.795, 0.753, 0.753, 0.749]
    plt.plot(x, SL, linewidth=2, color='tomato', linestyle='-', marker='o', ms=7, label='SL')
    # plt.plot(x, CSL, linewidth=2, color='royalblue', linestyle='-', marker='s', label='CSL', ms=7)
    plt.plot(x, RL_M, linewidth=2, color='darkorange', linestyle='-', marker='p', label='SL-DRL-M', ms=7)  # lightseagreen
    plt.plot(x, RL_V, linewidth=2, color='lightseagreen', linestyle='-', marker='*', label='SL-DRL-V', ms=7)  # darkorange
    plt.plot(x, RL_D, linewidth=2, color='slateblue', linestyle='-', marker='X', label='SL-DRL-D', ms=7)  # darkorange


    plt.legend(loc=0, fontsize=l, ncol=1)

    plt.xlabel('Degree of Vacuity', fontsize=20, fontweight = "bold")
    plt.ylabel('Precision Accuracy', fontsize=20, fontweight = "bold")
    x_tick = ['25%-100%', '17%-100%', '13%-100%', '8%-100%', '5%-100%']

    plt.yticks(fontsize=15, fontweight = "bold")
    plt.xticks(fontsize=15, fontweight = "bold")
    plt.ylim((0.72, 0.81))
    my_y_ticks = np.arange(0.73, 0.79, 0.02)
    plt.yticks(my_y_ticks)
    plt.xticks(x, x_tick, rotation=0, fontsize=12)
    plt.savefig("C:/Users/Shu/Desktop/Fusion19/PA_T_accuracy.png", dpi=800)
    # plt.show()

    return


if __name__ == '__main__':
    # PA_path_1000()
    # PA_path_1000_p()
    # PA_T()
    PA_T_p()