import numpy as np
from numba import cuda
np.random.seed(42)
import torch
import time
from GraphCDE import GraphCDE, animate_2D
import plotting_params
plotting_params.set_plotting_params(9, 10, 12)

if __name__ == '__main__':
    n = 5
    num_steps = 100
    len_x = 1000
    N = 5
    dyadic_order = 2
    M = 2

    flows = {}

    # Birth flows
    for i in range(n):
        for j in range(n-1):
            flows[(i * n + j, i * n + j + 1)] = i * n + j

    for i in range(n - 1):
        for j in range(n):
            flows[(i * n + j, (i+1) * n + j)] = -j*n + i

    G = GraphCDE(n * n, flows, num_steps, N, dyadic_order, M)

    x = torch.rand((len_x, 2*(n-1)*n)) * 0.06

    def sigmoid(xx, scale, center):
        return 1 / (1 + np.exp(-scale * xx + center))


    for i in range(n):
        for j in range(n - 1):
            arr = np.linspace(0,1, len_x)
            x[:,i * n + j] += torch.Tensor(sigmoid(arr, 10, i + j))

    for i in range(n-1):
        for j in range(n):
            arr = np.linspace(0,1, len_x)
            x[:,- j * n + i] += torch.Tensor(sigmoid(arr, 10, i + j))

    x *= 0.01

    G.set_x(x)

    print("Sparsity: ", G.sparsity())

    start = time.time()
    G.run(torch.ones((n*n)))
    cuda.synchronize()
    end = time.time()
    print("Time: ", end - start)

    print(G.y)
    print("Num Coeffs:", len(G.walks))

    res = np.array(G.y).T
    res = np.reshape(res, (n,n,-1))

    animate_2D(res, save_as="2D_example.gif")