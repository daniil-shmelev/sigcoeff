import torch
import sigcoeff
import time
from numba import cuda
import numpy as np

if __name__ == '__main__':
    torch.manual_seed(42)

    #Underlying path
    len_x = 1000
    dim = 10
    X = torch.rand((len_x, dim), device = "cpu")

    #Target multi-index
    multi_index = [1,5,2,6,3]

    #Algorithm parameters
    M = 2
    dyadic_order = 3

    #############################################
    # Serial CPU computation
    #############################################
    start = time.time()
    coeff = sigcoeff.compute_coeff(X, multi_index, M = M, dyadic_order = dyadic_order, parallel = False)
    end = time.time()
    print(f"{'Coefficient:':<15} {float(coeff):<10.4f} {'Time:':<10} {end - start:.4f}")

    #############################################
    # Parallel CPU computation
    #############################################
    start = time.time()
    coeff = sigcoeff.compute_coeff(X, multi_index, M=M, dyadic_order=dyadic_order, parallel=True)
    end = time.time()
    print(f"{'Coefficient:':<15} {float(coeff):<10.4f} {'Time:':<10} {end - start:.4f}")

    #############################################
    # Parallel GPU computation
    #############################################
    start = time.time()
    coeff = sigcoeff.compute_coeff(X.cuda(), multi_index, M=M, dyadic_order=dyadic_order)
    cuda.synchronize() # Wait for cuda to finish. For timing purposes only
    end = time.time()
    print(f"{'Coefficient:':<15} {float(coeff):<10.4f} {'Time:':<10} {end - start:.4f}")

    #############################################
    # Varying dyadic orders for the two dimensions of the PDE grid
    #############################################
    start = time.time()
    coeff = sigcoeff.compute_coeff(X.cuda(), multi_index, M=M, dyadic_order=(3,2))
    cuda.synchronize() # Wait for cuda to finish. For timing purposes only
    end = time.time()
    print(f"{'Coefficient:':<15} {float(coeff):<10.4f} {'Time:':<10} {end - start:.4f}")

    #############################################
    # Extraction of the entire grid of coefficients. I.e. all coefficients given by multi_index[:i],
    # evaluated at all time points up to len_x. (CUDA only)
    #############################################
    start = time.time()
    coeff_grid = sigcoeff.compute_coeff(X.cuda(), multi_index, M=M, dyadic_order=dyadic_order, full = True)
    cuda.synchronize() # Wait for cuda to finish. For timing purposes only
    end = time.time()

    np.set_printoptions(suppress=True)
    print("\nCoefficient array of shape (len(x), len(multi_index)): \n", np.round(np.array(coeff_grid.cpu()), 4))
    print("\nTime: ", end - start)