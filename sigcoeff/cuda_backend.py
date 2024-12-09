import torch.cuda
from numba import cuda
import numba

@cuda.jit('int64(int64,int64)', fastmath = True, device = True)
def lambda_sign(lambda_, dim):
    # Returns 1 if lambda_ has an even number of zeros in its binary expansion and -1 otherwise
    # dim is the length of the binary representation of lambda_
    ones_count = 0
    while lambda_:
        ones_count += lambda_ & 1
        lambda_ >>= 1
    return 1 if (dim - ones_count) % 2 == 0 else -1

def get_alpha(beta, scaling_depth, dim, device):
    # Returns Vandermonde coefficients alpha given beta
    alpha_array = torch.empty(scaling_depth + 1, dtype = torch.float64, device = device)
    sign = -1 if scaling_depth & 1 else 1 #(-1) ** scaling_depth

    for i in range(scaling_depth + 1):
        alph = sign
        alph /= beta[i] ** dim
        for j in range(scaling_depth + 1):
            if j == i:
                continue
            alph *= beta[j] / (beta[i] - beta[j])
        alpha_array[i] = alph
    return alpha_array

#####################################################################
#get_coeff_cuda, get_kernel_cuda allocate the entire PDE grid
#####################################################################
def get_kernel_grid_cuda(pde_grid, path, scaling_depth, dyadic_order_1, dyadic_order_2, dim, max_lambda, dyadic_len, dyadic_dim, full = False):
    # Populates PDE grid K of shape (M + 1, lam_lim, dyadic_len, dyadic_dim + 1)

    # Assign initial values
    pde_grid[:, :, 0, :] = 1.
    pde_grid[:, :, :, 0] = 1.

    # Compute beta and alpha for Vandermonde
    beta = torch.linspace(0.1, 1, scaling_depth + 1, dtype=torch.float64, device='cuda') if scaling_depth > 0 else torch.ones(1, dtype=torch.float64, device='cuda')
    alpha = get_alpha(beta, scaling_depth, dim, 'cuda')

    # Total number of antidiagonals
    num_anti_diag = dyadic_len + dyadic_dim

    # Populate grid
    get_kernel_cuda[(scaling_depth + 1, max_lambda), dyadic_dim,](pde_grid, path, alpha, beta, dim, dyadic_dim, dyadic_len, num_anti_diag, dyadic_order_1, dyadic_order_2, full)

def get_coeff_cuda(path, scaling_depth, dyadic_order_1, dyadic_order_2, full = False):

    path_length = path.shape[0]
    dim = path.shape[1]

    # In the method, lambda represents a component-wise scaling. In the case of a central difference,
    # lambda is a vector in {-1,1}^dim. We choose to represent this vector as an integer, whose binary
    # representation denotes the entries of the vector. For example, 5 = (1,0,1)_2 corresponds to the
    # vector (1, -1, 1). max_lambda here denotes the upper bound for these integers.
    max_lambda = 1 << dim

    # Dyadically refined grid dimensions
    dyadic_len = ((path_length - 1) << dyadic_order_1) + 1
    dyadic_dim = (dim << dyadic_order_2)

    # Allocate PDE grids
    pde_grid = torch.empty((scaling_depth + 1, max_lambda, dyadic_len, dyadic_dim + 1), dtype=torch.float64, device='cuda')

    # Populate PDE grids
    get_kernel_grid_cuda(pde_grid, path, scaling_depth, dyadic_order_1, dyadic_order_2, dim, max_lambda, dyadic_len, dyadic_dim, full)

    # Sum as necessary
    if full:
        result = torch.empty((path_length, dim), dtype=torch.float64, device='cuda')
        for i in range(dim, 0, -1):
            result[:, i - 1] = torch.sum(pde_grid[:, :(1 << i), ::(1 << dyadic_order_1), i << dyadic_order_2], dim=(0, 1)) * (0.5 ** i)
        return result
    else:
        result = torch.sum(pde_grid[:, :, -1, -1])
        return result * (0.5 ** dim)

@cuda.jit('void(float64[:,:,:,:],float64[:,:],float64[:],float64[:],int64,int64,int64,int64,int64,int64,boolean)', fastmath = True)
def get_kernel_cuda(pde_grid, path, alpha_arr, beta_arr, dim, dyadic_dim, dyadic_len, num_anti_diag, dyadic_order_1, dyadic_order_2, full):
    # Each block corresponds to a single (beta, lambda_) pair.
    M_idx = int(cuda.blockIdx.x)
    lambda_ = int(cuda.blockIdx.y)
    # Each thread works on a node of a diagonal.
    thread_id = int(cuda.threadIdx.x)

    beta_frac = beta_arr[M_idx] / (1 << (dyadic_order_1 + dyadic_order_2))
    twelth = 1. / 12

    for p in range(2, num_anti_diag):  # First two antidiagonals are initialised to 1
        start_j = max(1, p - dyadic_len + 1)
        end_j = min(p, dyadic_dim + 1)

        j = start_j + thread_id

        if j < end_j:
            d = (j - 1) >> dyadic_order_2

            i = p - j  # Calculate corresponding i (since i + j = p)
            ii = ((i - 1) >> dyadic_order_1) + 1

            if lambda_ & (1 << d):
                deriv = beta_frac * (path[ii, d] - path[ii - 1, d])
            else:
                deriv = beta_frac * (path[ii - 1, d] - path[ii, d])

            deriv_2 = deriv * deriv * twelth
            pde_grid[M_idx, lambda_, i, j] = (pde_grid[M_idx, lambda_, i, j - 1] + pde_grid[M_idx, lambda_, i - 1, j]) * (
                    1. + 0.5 * deriv + deriv_2) - pde_grid[M_idx, lambda_, i - 1, j - 1] * (1. - deriv_2)

        # Wait for other threads in this block
        cuda.syncthreads()

    #scale as necessary
    if full:
        if thread_id < dim:
            j = thread_id + 1
            dim_idx = j << dyadic_order_2
            fact = lambda_sign(lambda_, j) * alpha_arr[M_idx] * (beta_arr[M_idx] ** (dim - j))
            for i in range(0, dyadic_len, 1 << dyadic_order_1):
                pde_grid[M_idx, lambda_, i, dim_idx] -= 1
                pde_grid[M_idx, lambda_, i, dim_idx] *= fact
    else:
        if thread_id == 0:
            pde_grid[M_idx, lambda_, -1, -1] -= 1
            pde_grid[M_idx, lambda_, -1, -1] *= lambda_sign(lambda_, dim) * alpha_arr[M_idx]


#####################################################################
#get_coeff_cuda_2, get_kernel_cuda_2 allocate only the required 3 anti-diagonals
#These are preferred in the single coefficient case for memory efficiency
#####################################################################
def get_coeff_cuda_2(path, scaling_depth, dyadic_order_1, dyadic_order_2):
    path_length = path.shape[0]
    dim = path.shape[1]

    # In the method, lambda represents a component-wise scaling. In the case of a central difference,
    # lambda is a vector in {-1,1}^dim. We choose to represent this vector as an integer, whose binary
    # representation denotes the entries of the vector. For example, 5 = (1,0,1)_2 corresponds to the
    # vector (1, -1, 1). max_lambda here denotes the upper bound for these integers.
    max_lambda = 1 << dim

    # Dyadically refined grid dimensions
    dyadic_len = ((path_length - 1) << dyadic_order_1) + 1
    dyadic_dim = (dim << dyadic_order_2)

    # Allocate array to store results
    results = torch.empty((scaling_depth + 1, max_lambda), dtype=torch.float64, device ='cuda') #Container for results of kernel evaluations

    # Get beta and alpha for Vandermonde
    beta = torch.linspace(0.1, 1, scaling_depth + 1, dtype=torch.float64, device ='cuda') if scaling_depth > 0 else torch.ones(1, dtype=torch.float64, device ='cuda')
    alpha = get_alpha(beta, scaling_depth, dim, 'cuda')

    #Total number of antidiagonals
    num_anti_diag = dyadic_len + dyadic_dim

    # Populate results
    sharedmem = 24 * (dyadic_dim + 1)
    get_kernel_cuda_2[(scaling_depth + 1, max_lambda), dyadic_dim, 0, sharedmem](results, path, alpha, beta, dim, dyadic_len, dyadic_dim, num_anti_diag, dyadic_order_1, dyadic_order_2)

    # Sum and scale
    result = torch.sum(results)
    return result * (0.5 ** dim)

@cuda.jit('void(float64[:,:],float64[:,:],float64[:],float64[:],int64,int64,int64,int64,int64,int64)', fastmath = True)
def get_kernel_cuda_2(results, path, alpha_arr, beta_arr, dim, dyadic_len, dyadic_dim, n_anti_diag, dyadic_order_1, dyadic_order_2):
    # Each block corresponds to a single (beta, lambda_) pair.
    M_idx = int(cuda.blockIdx.x)
    lambda_ = int(cuda.blockIdx.y)
    # Each thread works on a node of a diagonal.
    thread_id = int(cuda.threadIdx.x)

    beta_frac = beta_arr[M_idx] / (1 << (dyadic_order_1 + dyadic_order_2))
    twelth = 1. / 12

    #Shared memory for the 3 antidiagonals
    shared_memory = cuda.shared.array(shape=0, dtype=numba.float64)

    #Initialise to 1
    for i in range(3):
        shared_memory[i * (dyadic_dim + 1) + thread_id] = 1.

    #Only dyadic_dim many threads passed, so deal with last index using thread 0
    if thread_id == 0:
        for i in range(3):
            shared_memory[i * (dyadic_dim + 1) + dyadic_dim] = 1.

    #Wait for initialisation of shared memory
    cuda.syncthreads()

    # Indices determine the start points of the antidiagonals in memory
    # Instead of swaping memory, we swap indices to avoid memory copy
    prev_prev_diag_idx = 0
    prev_diag_idx = (dyadic_dim + 1)
    next_diag_idx = 2 * (dyadic_dim + 1)

    for p in range(2, n_anti_diag):  # First two antidiagonals are initialised to 1
        start = max(1, p - dyadic_len + 1)
        end = min(p, dyadic_dim + 1)

        j = start + thread_id

        if j < end:
            d = (j - 1) >> dyadic_order_2

            i = p - j  # Calculate corresponding i (since i + j = p)
            ii = ((i - 1) >> dyadic_order_1) + 1

            if lambda_ & (1 << d):
                deriv = beta_frac * (path[ii, d] - path[ii - 1, d])
            else:
                deriv = beta_frac * (path[ii - 1, d] - path[ii, d])

            deriv_2 = deriv * deriv * twelth

            # Update the next diagonal entry
            shared_memory[next_diag_idx + j] = (
                          shared_memory[prev_diag_idx + j] +
                          shared_memory[prev_diag_idx + j - 1]) * (
                              1. + 0.5 * deriv + deriv_2) - shared_memory[prev_prev_diag_idx + j - 1] * (1. - deriv_2)

        # Wait for all threads in this block to finish
        cuda.syncthreads()

        # Rotate the diagonals (swap indices, no data copying)
        prev_prev_diag_idx, prev_diag_idx, next_diag_idx = prev_diag_idx, next_diag_idx, prev_prev_diag_idx

        #Make sure all threads wait for the rotation of diagonals
        cuda.syncthreads()

    #Add results
    if thread_id == 0:
        results[M_idx, lambda_] = lambda_sign(lambda_, dim) * alpha_arr[M_idx] * (shared_memory[prev_diag_idx + dyadic_dim] - 1)

#####################################################################
# A serial implementation on CUDA for an apples-to-apples comparison
# For timing purposes only
#####################################################################
def get_coeff_cuda_serial(path, scaling_depth, dyadic_order_1, dyadic_order_2):
    path_length = path.shape[0]
    dim = path.shape[1]

    # In the method, lambda represents a component-wise scaling. In the case of a central difference,
    # lambda is a vector in {-1,1}^dim. We choose to represent this vector as an integer, whose binary
    # representation denotes the entries of the vector. For example, 5 = (1,0,1)_2 corresponds to the
    # vector (1, -1, 1). max_lambda here denotes the upper bound for these integers.
    max_lambda = 1 << dim
    result = 0

    # Dyadically refined grid dimensions
    dyadic_len = ((path_length - 1) << dyadic_order_1) + 1
    dyadic_dim = (dim << dyadic_order_2) + 1

    beta = torch.linspace(0.1, 1, scaling_depth + 1, dtype=torch.float64, device='cuda') if scaling_depth > 0 else torch.ones(1, dtype=torch.float64, device='cuda')
    alpha = get_alpha(beta, scaling_depth, dim, 'cuda')

    pde_grid = torch.empty((dyadic_len, dyadic_dim), dtype = torch.float64, device = "cuda")

    for lambda_ in range(max_lambda):
        for i in range(scaling_depth + 1):
            get_kernel_cuda_serial[1,1](pde_grid, path, lambda_, beta[i], alpha[i], dyadic_order_1, dyadic_order_2)
            result += pde_grid[-1,-1]

    result *= 0.5 ** dim
    return result

@cuda.jit('void(float64[:,:], float64[:,:],int64,float64,float64,int64,int64)', fastmath = True)
def get_kernel_cuda_serial(pde_grid, path, lambda_, beta, alpha, dyadic_order_1, dyadic_order_2):
    path_length = path.shape[0]
    dim = path.shape[1]
    beta_frac = beta / (1 << (dyadic_order_1 + dyadic_order_2))
    twelth = 1. / 12

    # Dyadically refined grid dimensions
    dyadic_len = ((path_length - 1) << dyadic_order_1) + 1
    dyadic_dim = (dim << dyadic_order_2) + 1

    # Initialization of K array
    for i in range(dyadic_len):
        pde_grid[i, 0] = 1.0

    for j in range(dyadic_dim):
        pde_grid[0, j] = 1.0

    for j in range(dyadic_dim - 1):
        d = j >> dyadic_order_2
        for i in range(dyadic_len - 1):
            ii = i >> dyadic_order_1

            if lambda_ & (1 << d):
                deriv = beta_frac * (path[ii + 1, d] - path[ii, d])
            else:
                deriv = beta_frac * (path[ii, d] - path[ii + 1, d])

            deriv_2 = deriv * deriv * twelth
            pde_grid[i + 1, j + 1] = (
                    (pde_grid[i + 1, j] + pde_grid[i, j + 1])
                    * (1.0 + 0.5 * deriv + deriv_2)
                    - pde_grid[i, j] * (1.0 - deriv_2)
            )
    pde_grid[dyadic_len - 1, dyadic_dim - 1] -= 1
    pde_grid[dyadic_len - 1, dyadic_dim - 1] *= lambda_sign(lambda_, dim) * alpha

#####################################################################
# An implementation of Chen's relation for an apples-to-apples comparison
#####################################################################
def chen_cuda_(path):
    dim = path.shape[1]
    sharedmem = 16 * (dim + 1)
    result = torch.empty(1, dtype = torch.float64, device = "cuda")
    run_chen_cuda[1, dim, 0, sharedmem](path, result)
    return result

@cuda.jit('void(double[:,:], double[:])', fastmath = True)
def run_chen_cuda(path, result):
    thread_id = int(cuda.threadIdx.x)
    dim = path.shape[1]
    path_length = path.shape[0]

    shared_memory = cuda.shared.array(shape=0, dtype=numba.float64)

    last_coeffs_idx = 0
    new_coeffs_idx = dim + 1

    # Set the 0^th coefficient to 1
    if thread_id == 0:
        shared_memory[last_coeffs_idx] = 1
        shared_memory[new_coeffs_idx] = 1

    # Compute coefficients for the first linear segment
    fact_prod = 1.
    for i in range(1, dim + 1):
        fact_prod *= (path[1, i - 1] - path[0, i - 1]) / i
        shared_memory[last_coeffs_idx + i] = fact_prod  # prod / factorial(i)

    for LL in range(1, path_length - 1):
        for ii in range(dim):
            i = ii+1
            shared_memory[new_coeffs_idx + i] = shared_memory[last_coeffs_idx + i]

            fact_prod = 1.
            for k in range(i-1, -1, -1):
                fact_prod *= (path[LL + 1, k] - path[LL, k]) / (i - k)
                shared_memory[new_coeffs_idx + i] += shared_memory[last_coeffs_idx + k] * fact_prod

        new_coeffs_idx, last_coeffs_idx = last_coeffs_idx, new_coeffs_idx

    #Store result
    result[0] = shared_memory[last_coeffs_idx + dim]