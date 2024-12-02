import numpy as np
from numba import cuda
import time
import torch
import matplotlib.pyplot as plt
import sigcoeff
import plotting_params
plotting_params.set_plotting_params(9, 10, 12)

if __name__ == '__main__':
    np.random.seed(42)

    # Set path
    X = np.random.uniform(size=(10000, 10))
    X = torch.Tensor(X, device = "cpu")

    # Algorithm parameters
    M = 1
    dyadic_order = 1

    k_vals = [k for k in range(1, 7)]
    k_time_chen = []
    k_time_kernels = []

    numRuns = 10

    #############################################
    # Chen (GPU)
    #############################################
    X = X.to(device = "cuda")
    for k in k_vals:
        multi_index = np.array([i for i in range(k)])
        start = time.time()
        for i in range(numRuns):
            res = sigcoeff.chen_cuda(X, multi_index)
            cuda.synchronize()  # Wait for cuda to finish. For timing purposes only
        end = time.time()
        print("Result: ", np.array(res))
        time_avg = (end - start) / numRuns
        print("Time: ", time_avg, "\n")
        k_time_chen.append(time_avg)

    print("#"*30)

    #############################################
    # Kernels (GPU)
    #############################################
    X = X.to(device='cuda')

    # First call to warm up the cache
    res = sigcoeff.compute_coeff(X, np.array([i for i in range(7)]), M=M, dyadic_order=dyadic_order)
    cuda.synchronize() # Wait for cuda to finish. For timing purposes only

    for k in k_vals:
        multi_index = np.array([i for i in range(k)])
        start = time.time()
        for i in range(numRuns):
            res = sigcoeff.compute_coeff(X, multi_index, M = M, dyadic_order = dyadic_order)
            cuda.synchronize() # Wait for cuda to finish. For timing purposes only
        end = time.time()
        print("Result: ", np.array(res.cpu()))
        time_avg = (end - start) / numRuns
        print("Time: ", time_avg, "\n")
        k_time_kernels.append(time_avg)

    #############################################
    # Plot
    #############################################
    plt.figure(figsize=(4, 3))
    plt.plot(k_vals, np.array(k_time_chen) , color = 'firebrick', marker = 'o')
    plt.plot(k_vals, np.array(k_time_kernels) , color='darkgreen', marker = 'o')
    plt.title(r"Dependence of time complexity on coefficient depth")
    plt.grid(True, linestyle='--')
    plt.legend([r'Chen (CUDA)', r'Kernels (CUDA)', r'CUDA'])
    plt.xlabel(r'Coefficient depth $n$')
    plt.ylabel(r'Elapsed Time (s)')
    plt.tight_layout()
    plt.subplots_adjust(right = 0.95)
    #plt.savefig("..\\plots\\depth_dependence_chen.png", dpi=300)
    #plt.savefig("..\\plots\\depth_dependence_chen.pdf", dpi=300)
    plt.show()