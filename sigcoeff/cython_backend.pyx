from cython.parallel cimport prange, parallel
from libc.stdlib cimport malloc, free
cimport openmp
import cython
import numpy as np

cpdef int lamda_sign(unsigned int lambda_, unsigned int dim) nogil:
    # Returns 1 if lambda_ has an even number of zeros in its binary expansion and -1 otherwise
    # dim is the length of the binary representation of lambda_
    # This sign is necessary for computation of the central difference
    cdef int ones_count = 0
    while lambda_:
        ones_count += lambda_ & 1
        lambda_ >>= 1
    return 1 if (dim - ones_count) % 2 == 0 else -1

cpdef double[:] get_alpha(double[:] beta, unsigned int scaling_depth, unsigned int dim):
    # Returns Vandermonde coefficients alpha given beta
    cdef double[:] alpha_array = np.empty(scaling_depth + 1, dtype = np.double)
    cdef int sign = -1 if scaling_depth & 1 else 1 #(-1) ** scaling_depth
    cdef double alpha
    cdef int i

    for i in range(scaling_depth + 1):
        alpha = sign
        alpha /= beta[i] ** dim
        for j in range(scaling_depth + 1):
            if j == i:
                continue
            alpha *= beta[j] / (beta[i] - beta[j])
        alpha_array[i] = alpha
    return alpha_array

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double get_coeff_cython(double[:,:] path, unsigned int scaling_depth, unsigned int dyadic_order_1, unsigned int dyadic_order_2, bint use_parallel):

    cdef unsigned int dim = path.shape[1]

    # In the method, lambda represents a component-wise scaling. In the case of a central difference,
    # lambda is a vector in {-1,1}^dim. We choose to represent this vector as an integer, whose binary
    # representation denotes the entries of the vector. For example, 5 = (1,0,1)_2 corresponds to the
    # vector (1, -1, 1). max_lambda here denotes the upper bound for these integers.
    cdef int max_lambda = 1 << dim

    cdef double result = 0
    cdef int lambda_, sgn, i, max_threads, chunk

    cdef double[:] beta = np.linspace(0.1, 1, scaling_depth + 1, dtype = np.double) if scaling_depth > 0 else np.array([1.], dtype = np.double)
    cdef double[:] alpha = get_alpha(beta, scaling_depth, dim)

    if use_parallel:
        max_threads = openmp.omp_get_max_threads()
        openmp.omp_set_num_threads(max_threads)
        chunk = <int>(max_lambda / max_threads)

        for lambda_ in prange(max_lambda, num_threads=max_threads, chunksize = chunk, schedule ="static", nogil=True):
            sgn = lamda_sign(lambda_, dim)
            for i in range(scaling_depth + 1):
                result += sgn * alpha[i] * (get_kernel_cython(path, lambda_, beta[i], dyadic_order_1, dyadic_order_2) - 1)

    else:
        for lambda_ in range(max_lambda):
            sgn = lamda_sign(lambda_, dim)
            for i in range(scaling_depth+1):
                result += sgn * alpha[i] * (get_kernel_cython(path, lambda_, beta[i], dyadic_order_1, dyadic_order_2) - 1)

    result *= 0.5 ** dim
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double get_kernel_cython(double[:,:] path, int lambda_, double beta, unsigned int dyadic_order_1, unsigned int dyadic_order_2) nogil:
    cdef int path_length = path.shape[0]
    cdef int dim = path.shape[1]
    cdef int i, j, sgn, d, ii
    cdef double beta_frac = beta / (1 << (dyadic_order_1 + dyadic_order_2))
    cdef double twelth = 1. / 12
    cdef double result

    # Dyadically refined grid dimensions
    cdef int dyadic_len = ((path_length - 1) << dyadic_order_1) + 1
    cdef int dyadic_dim = (dim << dyadic_order_2) + 1

    # Allocate (flattened) PDE grid
    cdef double* pde_grid = <double*>malloc(dyadic_len * dyadic_dim * sizeof(double))
    cdef double deriv, deriv_2

    # Initialization of K array
    for i in range(dyadic_len):
        pde_grid[i * dyadic_dim] = 1.0  # Set K[i, 0] = 1.0

    for j in range(dyadic_dim):
        pde_grid[j] = 1.0  # Set K[0, j] = 1.0

    for j in range(dyadic_dim - 1):
        d = j >> dyadic_order_2
        for i in range(dyadic_len - 1):
            ii = i >> dyadic_order_1

            if lambda_ & (1 << d):
                deriv = beta_frac * (path[ii + 1, d] - path[ii, d])
            else:
                deriv = beta_frac * (path[ii, d] - path[ii + 1, d])

            deriv_2 = deriv * deriv * twelth
            pde_grid[(i + 1) * dyadic_dim + (j + 1)] = (
                    (pde_grid[(i + 1) * dyadic_dim + j] + pde_grid[i * dyadic_dim + (j + 1)])
                    * (1.0 + 0.5 * deriv + deriv_2)
                    - pde_grid[i * dyadic_dim + j] * (1.0 - deriv_2)
            )
    result = <double> pde_grid[dyadic_len * dyadic_dim - 1]

    # Free PDE grid
    free(pde_grid)
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double chen_(double[:,:] path) nogil:
    cdef int dim = path.shape[1]
    cdef int path_length = path.shape[0]
    cdef double prod, result
    cdef int i, j, k, m, ii

    cdef double* last_coeffs = <double*>malloc((dim+1) * sizeof(double))
    cdef double* new_coeffs = <double*>malloc((dim+1) * sizeof(double))

    last_coeffs[0] = 1
    new_coeffs[0] = 1

    fact_prod = 1.
    for i in range(1, dim + 1):
        fact_prod *= (path[1, i-1] - path[0, i-1]) / i
        last_coeffs[i] = fact_prod # prod / factorial(i)

    for m in range(1, path_length - 1):
        for ii in range(dim):
            i = ii+1
            new_coeffs[i] = last_coeffs[i]

            fact_prod = 1.
            for k in range(i-1, -1, -1):
                fact_prod *= (path[m + 1, k] - path[m, k]) / (i - k)
                new_coeffs[i] += last_coeffs[k] * fact_prod

        new_coeffs, last_coeffs = last_coeffs, new_coeffs

    result = last_coeffs[dim]

    free(last_coeffs)
    free(new_coeffs)

    return result