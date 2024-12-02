import numpy as np
np.random.seed(42)
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from GraphCDE import GraphCDE
import plotting_params
plotting_params.set_plotting_params(9, 10, 12)


def birth_death_epidemic(n, y0, num_steps, N, dyadic_order, M, t_ep = 0.5, epidemic = True):
    flows = {}

    # Birth flows
    for i in range(n - 1):
        flows[(i, i + 1)] = i

    # Death flows
    if epidemic:
        for i in range(n):
            flows[(i,i)] = n-2 + i

    G = GraphCDE(n, flows, num_steps, N, dyadic_order, M)

    d = 2*n-1 if epidemic else n-1

    x = torch.empty((1000, d))

    def sigmoid(xx, a=1., b=1., c=1.):
        return a / (1 + np.exp(-b * (xx - c)))

    for i in range(n - 1):
        arr = np.linspace(0, 1, total_steps)
        x[:, i] = torch.Tensor(sigmoid(arr, a=1, b=10, c=i / n))

    if epidemic:
        for i in range(n):
            arr = np.linspace(0, 1, total_steps)
            x[:, n-2 + i] = - torch.Tensor(sigmoid(arr, a = (n-i) / (n), b = 10, c = t_ep))

    G.set_x(x)
    G.run(y0)

    print("Sparsity: ", G.sparsity())

    return G.y


if __name__ == '__main__':
    n = 14
    num_steps = 100
    total_steps = 1000
    N = 2
    dyadic_order = 2
    M = 2
    y0 = torch.ones((n))

    cmap = cm.get_cmap("rainbow", n)
    t = np.linspace(0,1, num_steps + 1)

    #############################################
    # No Epidemic
    #############################################
    fig, ax = plt.subplots(2, 2, figsize = (7.5,6))

    y = birth_death_epidemic(n, y0, num_steps, N, dyadic_order, M, epidemic = False)
    for i in range(n):
        ax[0][0].plot(t, y[:, i], color=cmap(i), label = "Gen " + (str(i+1) if i < n - 1 else ">" + str(n - 1)))

    #############################################
    # t_ep = 0.8
    #############################################
    y = birth_death_epidemic(n, y0, num_steps, N, dyadic_order, M, t_ep=0.8, epidemic=True)
    for i in range(n):
        ax[0][1].plot(t, y[:, i], color=cmap(i), label="Gen " + (str(i + 1) if i < n - 1 else ">" + str(n - 1)))

    ax[0][1].axvline(x=0.8, linestyle = '--', label = "Epidemic Peak", color = "black")

    #############################################
    # t_ep = 0.5
    #############################################
    y = birth_death_epidemic(n, y0, num_steps, N, dyadic_order, M, t_ep=0.5, epidemic=True)
    for i in range(n):
        ax[1][0].plot(t, y[:, i], color=cmap(i), label="Gen " + (str(i + 1) if i < n - 1 else ">" + str(n - 1)))

    ax[1][0].axvline(x=0.5, linestyle='--', label="Epidemic Peak", color="black")

    #############################################
    # t_ep = 0.2
    #############################################
    y = birth_death_epidemic(n, y0, num_steps, N, dyadic_order, M, t_ep=0.2, epidemic=True)
    for i in range(n):
        ax[1][1].plot(t, y[:, i], color=cmap(i), label="Gen " + (str(i + 1) if i < n - 1 else ">" + str(n - 1)))

    ax[1][1].axvline(x=0.2, linestyle='--', label="Epidemic Peak", color="black")


    fig.legend(*ax[1][1].get_legend_handles_labels(), loc=(0.19,0.05), fancybox=True, shadow=True, ncol=5)

    fig.subplots_adjust(bottom=0.21)

    #plt.savefig("epidemic.png", dpi=300)
    #plt.savefig("epidemic.pdf", dpi=300)

    plt.show()
