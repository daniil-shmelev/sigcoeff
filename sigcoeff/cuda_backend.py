import torch.cuda
from mpmath import matrix
from numba import cuda
import numba

@cuda.jit('int64(int64,int64)', fastmath = True, device = True)
def lam_sign(n, dim):
    # Returns 1 if n has an even number of zeros in its binary expansion and -1 otherwise
    # dim is the length of the binary representation of n
    count = 0
    while n:
        count += n & 1
        n >>= 1
    return 1 if (dim - count) % 2 == 0 else -1

def get_alpha(beta, M, dim, device):
    # Returns Vandermonde coefficients alpha given beta
    out = torch.empty(M + 1, dtype = torch.float64, device = device)
    sign = -1 if M & 1 else 1 #(-1) ** M

    for i in range(M + 1):
        alph = sign
        alph /= beta[i] ** dim
        for j in range(M + 1):
            if j == i:
                continue
            alph *= beta[j] / (beta[i] - beta[j])
        out[i] = alph
    return out

#####################################################################
#get_coeff_cuda, get_kernel_cuda allocate the entire PDE grid
#####################################################################
def get_kernel_grid_cuda(K, X, M, dyadic_order_1, dyadic_order_2, len_x, dim, lam_lim, dyadic_len, dyadic_dim, full = False):
    # Populates PDE grid K of shape (M + 1, lam_lim, dyadic_len, dyadic_dim + 1)

    # Assign initial values
    K[:, :, 0, :] = 1.
    K[:, :, :, 0] = 1.

    # Compute beta and alpha for Vandermonde
    beta = torch.linspace(0.1, 1, M + 1, dtype=torch.float64, device='cuda') if M > 0 else torch.ones(1, dtype=torch.float64, device='cuda')
    alpha = get_alpha(beta, M, dim, 'cuda')

    # Total number of antidiagonals
    n_anti_diag = dyadic_len + dyadic_dim

    # Populate grid
    get_kernel_cuda[(M + 1, lam_lim), dyadic_dim,](K, X, alpha, beta, dim, dyadic_dim, dyadic_len, n_anti_diag, dyadic_order_1, dyadic_order_2, full)

def get_coeff_cuda(X, M, dyadic_order_1, dyadic_order_2, full = False):
    len_x = X.shape[0]
    dim = X.shape[1]
    lam_lim = 1 << dim

    # Dyadically refined grid dimensions
    dyadic_len = ((len_x - 1) << dyadic_order_1) + 1
    dyadic_dim = (dim << dyadic_order_2)

    # Allocate PDE grids
    K = torch.empty((M + 1, lam_lim, dyadic_len, dyadic_dim + 1), dtype=torch.float64, device='cuda')

    # Populate PDE grids
    get_kernel_grid_cuda(K, X, M, dyadic_order_1, dyadic_order_2, len_x, dim, lam_lim, dyadic_len, dyadic_dim, full)

    # Sum as necessary
    if full:
        res = torch.empty((len_x, dim), dtype=torch.float64, device='cuda')
        for i in range(dim, 0, -1):
            res[:, i - 1] = torch.sum(K[:, :(1 << i), ::(1 << dyadic_order_1), i << dyadic_order_2], dim=(0, 1)) * (0.5 ** i)
        return res
    else:
        res = torch.sum(K[:, :, -1, -1])
        return res * (0.5 ** dim)

@cuda.jit('void(float64[:,:,:,:],float64[:,:],float64[:],float64[:],int64,int64,int64,int64,int64,int64,boolean)', fastmath = True)
def get_kernel_cuda(K, X, alpha_arr, beta_arr, dim, dyadic_dim, dyadic_len, n_anti_diag, dyadic_order_1, dyadic_order_2, full):
    # Each block corresponds to a single (beta, lam) pair.
    M_idx = int(cuda.blockIdx.x)
    lam = int(cuda.blockIdx.y)
    # Each thread works on a node of a diagonal.
    thread_id = int(cuda.threadIdx.x)

    beta_frac = beta_arr[M_idx] / (1 << (dyadic_order_1 + dyadic_order_2))
    twelth = 1. / 12

    for p in range(2, n_anti_diag):  # First two antidiagonals are initialised to 1
        start_j = max(1, p - dyadic_len + 1)
        end_j = min(p, dyadic_dim + 1)

        j = start_j + thread_id

        if j < end_j:
            d = (j - 1) >> dyadic_order_2

            i = p - j  # Calculate corresponding i (since i + j = p)
            ii = ((i - 1) >> dyadic_order_1) + 1

            if lam & (1 << d):
                deriv = beta_frac * (X[ii, d] - X[ii - 1, d])
            else:
                deriv = beta_frac * (X[ii - 1, d] - X[ii, d])

            deriv_2 = deriv * deriv * twelth
            K[M_idx, lam, i, j] = (K[M_idx, lam, i, j - 1] + K[M_idx, lam, i - 1, j]) * (
                    1. + 0.5 * deriv + deriv_2) - K[M_idx, lam, i - 1, j - 1] * (1. - deriv_2)

        # Wait for other threads in this block
        cuda.syncthreads()

    #scale as necessary
    if full:
        if thread_id < dim:
            j = thread_id + 1
            dim_idx = j << dyadic_order_2
            fact = lam_sign(lam, j) * alpha_arr[M_idx] * (beta_arr[M_idx] ** (dim - j))
            for i in range(0, dyadic_len, 1 << dyadic_order_1):
                K[M_idx, lam, i, dim_idx] -= 1
                K[M_idx, lam, i, dim_idx] *= fact
    else:
        if thread_id == 0:
            K[M_idx, lam, -1, -1] -= 1
            K[M_idx, lam, -1, -1] *= lam_sign(lam, dim) * alpha_arr[M_idx]


#####################################################################
#get_coeff_cuda_2, get_kernel_cuda_2 allocate only the required 3 anti-diagonals
#These are preferred in the single coefficient case for memory efficiency
#####################################################################
def get_coeff_cuda_2(X, M, dyadic_order_1, dyadic_order_2):
    len_x = X.shape[0]
    dim = X.shape[1]
    lam_lim = 1 << dim

    # Dyadically refined grid dimensions
    dyadic_len = ((len_x - 1) << dyadic_order_1) + 1
    dyadic_dim = (dim << dyadic_order_2)

    # Allocate array to store results
    results = torch.empty((M+1, lam_lim), dtype=torch.float64, device = 'cuda') #Container for results of kernel evaluations

    # Get beta and alpha for Vandermonde
    beta = torch.linspace(0.1, 1, M + 1, dtype=torch.float64, device = 'cuda') if M > 0 else torch.ones(1, dtype=torch.float64, device = 'cuda')
    alpha = get_alpha(beta, M, dim, 'cuda')

    #Total number of antidiagonals
    n_anti_diag = dyadic_len + dyadic_dim

    # Populate results
    sharedmem = 24 * (dyadic_dim + 1)
    get_kernel_cuda_2[(M + 1, lam_lim), dyadic_dim, 0, sharedmem](results, X, alpha, beta, dim, dyadic_len, dyadic_dim, n_anti_diag, dyadic_order_1, dyadic_order_2)

    # Sum and scale
    res = torch.sum(results)
    return res * (0.5 ** dim)

@cuda.jit('void(float64[:,:],float64[:,:],float64[:],float64[:],int64,int64,int64,int64,int64,int64)', fastmath = True)
def get_kernel_cuda_2(results, X, alpha_arr, beta_arr, dim, dyadic_len, dyadic_dim, n_anti_diag, dyadic_order_1, dyadic_order_2):
    # Each block corresponds to a single (beta, lam) pair.
    M_idx = int(cuda.blockIdx.x)
    lam = int(cuda.blockIdx.y)
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

            if lam & (1 << d):
                deriv = beta_frac * (X[ii, d] - X[ii - 1, d])
            else:
                deriv = beta_frac * (X[ii - 1, d] - X[ii, d])

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
        results[M_idx, lam] = lam_sign(lam, dim) * alpha_arr[M_idx] * (shared_memory[prev_diag_idx + dyadic_dim] - 1)

#####################################################################
# A serial implementation on CUDA for an apples-to-apples comparison
# For timing purposes only
#####################################################################
def get_coeff_cuda_serial(X, M, dyadic_order_1, dyadic_order_2):
    len_x = X.shape[0]
    dim = X.shape[1]
    lam_lim = 1 << dim
    res = 0

    # Dyadically refined grid dimensions
    dyadic_len = ((len_x - 1) << dyadic_order_1) + 1
    dyadic_dim = (dim << dyadic_order_2) + 1

    beta = torch.linspace(0.1, 1, M + 1, dtype=torch.float64, device='cuda') if M > 0 else torch.ones(1, dtype=torch.float64, device='cuda')
    alpha = get_alpha(beta, M, dim, 'cuda')

    K = torch.empty((dyadic_len, dyadic_dim), dtype = torch.float64, device = "cuda")

    for lam in range(lam_lim):
        for i in range(M + 1):
            get_kernel_cuda_serial[1,1](K, X, lam, beta[i], alpha[i], dyadic_order_1, dyadic_order_2)
            res += K[-1,-1]

    res *= 0.5 ** dim
    return res

@cuda.jit('void(float64[:,:], float64[:,:],int64,float64,float64,int64,int64)', fastmath = True)
def get_kernel_cuda_serial(K, X, lam, beta, alpha, dyadic_order_1, dyadic_order_2):
    len_x = X.shape[0]
    dim = X.shape[1]
    beta_frac = beta / (1 << (dyadic_order_1 + dyadic_order_2))
    twelth = 1. / 12

    # Dyadically refined grid dimensions
    dyadic_len = ((len_x - 1) << dyadic_order_1) + 1
    dyadic_dim = (dim << dyadic_order_2) + 1

    # Initialization of K array
    for i in range(dyadic_len):
        K[i, 0] = 1.0

    for j in range(dyadic_dim):
        K[0, j] = 1.0

    for j in range(dyadic_dim - 1):
        d = j >> dyadic_order_2
        for i in range(dyadic_len - 1):
            ii = i >> dyadic_order_1

            if lam & (1 << d):
                deriv = beta_frac * (X[ii + 1, d] - X[ii, d])
            else:
                deriv = beta_frac * (X[ii, d] - X[ii + 1, d])

            deriv_2 = deriv * deriv * twelth
            K[i + 1, j + 1] = (
                    (K[i + 1, j] + K[i, j + 1])
                    * (1.0 + 0.5 * deriv + deriv_2)
                    - K[i, j] * (1.0 - deriv_2)
            )
    K[dyadic_len - 1, dyadic_dim - 1] -= 1
    K[dyadic_len - 1, dyadic_dim - 1] *= lam_sign(lam, dim) * alpha

#####################################################################
# An implementation of Chen's relation for an apples-to-apples comparison
#####################################################################
def chen_cuda_(X):
    dim = X.shape[1]
    sharedmem = 16 * (dim + 1)
    result = torch.empty(1, dtype = torch.float64, device = "cuda")
    run_chen_cuda[1, dim, 0, sharedmem](X, result)
    return result

@cuda.jit('void(double[:,:], double[:])', fastmath = True)
def run_chen_cuda(X, result):
    thread_id = int(cuda.threadIdx.x)
    dim = X.shape[1]
    L = X.shape[0]

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
        fact_prod *= (X[1, i - 1] - X[0, i - 1]) / i
        shared_memory[last_coeffs_idx + i] = fact_prod  # prod / factorial(i)

    for LL in range(1, L - 1):
        for ii in range(dim):
            i = ii+1
            shared_memory[new_coeffs_idx + i] = shared_memory[last_coeffs_idx + i]

            fact_prod = 1.
            for k in range(i-1, -1, -1):
                fact_prod *= (X[LL + 1, k] - X[LL, k]) / (i - k)
                shared_memory[new_coeffs_idx + i] += shared_memory[last_coeffs_idx + k] * fact_prod

        new_coeffs_idx, last_coeffs_idx = last_coeffs_idx, new_coeffs_idx

    #Store result
    result[0] = shared_memory[last_coeffs_idx + dim]