from cython.parallel cimport prange, parallel
from libc.stdlib cimport malloc, free
cimport openmp
import cython
import numpy as np

cpdef int lam_sign(unsigned int n, unsigned int dim) nogil:
    # Returns 1 if n has an even number of zeros in its binary expansion and -1 otherwise
    # dim is the length of the binary representation of n
    cdef int count = 0
    while n:
        count += n & 1
        n >>= 1
    return 1 if (dim - count) % 2 == 0 else -1

cpdef double[:] get_alpha(double[:] beta, unsigned int M, unsigned int dim):
    # Returns Vandermonde coefficients alpha given beta
    cdef double[:] out = np.empty(M + 1, dtype = np.double)
    cdef int sign = -1 if M & 1 else 1 #(-1) ** M
    cdef int i
    cdef double alph

    for i in range(M + 1):
        alph = sign
        alph /= beta[i] ** dim
        for j in range(M + 1):
            if j == i:
                continue
            alph *= beta[j] / (beta[i] - beta[j])
        out[i] = alph
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double get_coeff_cython(double[:,:] X, unsigned int M, unsigned int dyadic_order_1, unsigned int dyadic_order_2, bint use_parallel):
    cdef unsigned int dim = X.shape[1]
    cdef int lam_lim = 1 << dim
    cdef double res = 0
    cdef int lam, sgn, i, max_threads, chunk

    cdef double[:] beta = np.linspace(0.1, 1, M + 1, dtype = np.double) if M > 0 else np.array([1.], dtype = np.double)
    cdef double[:] alpha = get_alpha(beta, M, dim)

    if use_parallel:
        max_threads = openmp.omp_get_max_threads()
        openmp.omp_set_num_threads(max_threads)
        chunk = <int>(lam_lim / max_threads)

        for lam in prange(lam_lim, num_threads=max_threads, chunksize = chunk, schedule = "static", nogil=True):
            sgn = lam_sign(lam, dim)
            for i in range(M + 1):
                res += sgn * alpha[i] * (get_kernel_cython(X, lam, beta[i], dyadic_order_1, dyadic_order_2) - 1)

    else:
        for lam in range(lam_lim):
            sgn = lam_sign(lam, dim)
            for i in range(M+1):
                res += sgn * alpha[i] * (get_kernel_cython(X, lam, beta[i], dyadic_order_1, dyadic_order_2) - 1)

    res *= 0.5 ** dim
    return res

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double get_kernel_cython(double[:,:] X, int lam, double beta, unsigned int dyadic_order_1, unsigned int dyadic_order_2) nogil:
    cdef int len_x = X.shape[0]
    cdef int dim = X.shape[1]
    cdef int i, j, sgn, d, ii
    cdef double beta_frac = beta / (1 << (dyadic_order_1 + dyadic_order_2))
    cdef double twelth = 1. / 12
    cdef double out

    # Dyadically refined grid dimensions
    cdef int dyadic_len = ((len_x - 1) << dyadic_order_1) + 1
    cdef int dyadic_dim = (dim << dyadic_order_2) + 1

    # Allocate (flattened) PDE grid
    cdef double* K = <double*>malloc(dyadic_len * dyadic_dim * sizeof(double))
    cdef double deriv, deriv_2

    # Initialization of K array
    for i in range(dyadic_len):
        K[i * dyadic_dim] = 1.0  # Set K[i, 0] = 1.0

    for j in range(dyadic_dim):
        K[j] = 1.0  # Set K[0, j] = 1.0

    for j in range(dyadic_dim - 1):
        d = j >> dyadic_order_2
        for i in range(dyadic_len - 1):
            ii = i >> dyadic_order_1

            if lam & (1 << d):
                deriv = beta_frac * (X[ii + 1, d] - X[ii, d])
            else:
                deriv = beta_frac * (X[ii, d] - X[ii + 1, d])

            deriv_2 = deriv * deriv * twelth
            K[(i + 1) * dyadic_dim + (j + 1)] = (
                    (K[(i + 1) * dyadic_dim + j] + K[i * dyadic_dim + (j + 1)])
                    * (1.0 + 0.5 * deriv + deriv_2)
                    - K[i * dyadic_dim + j] * (1.0 - deriv_2)
            )
    out = <double> K[dyadic_len * dyadic_dim - 1]

    # Free PDE grid
    free(K)

    return out

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double chen_(double[:,:] X) nogil:
    cdef int dim = X.shape[1]
    cdef int L = X.shape[0]
    cdef double prod, res
    cdef int i, j, k, LL, ii

    cdef double* last_coeffs = <double*>malloc((dim+1) * sizeof(double))
    cdef double* new_coeffs = <double*>malloc((dim+1) * sizeof(double))

    last_coeffs[0] = 1
    new_coeffs[0] = 1

    fact_prod = 1.
    for i in range(1, dim + 1):
        fact_prod *= (X[1, i-1] - X[0, i-1]) / i
        last_coeffs[i] = fact_prod # prod / factorial(i)

    for LL in range(1, L - 1):
        for ii in range(dim):
            i = ii+1
            new_coeffs[i] = last_coeffs[i]

            fact_prod = 1.
            for k in range(i-1, -1, -1):
                fact_prod *= (X[LL + 1, k] - X[LL, k]) / (i - k)
                new_coeffs[i] += last_coeffs[k] * fact_prod

        new_coeffs, last_coeffs = last_coeffs, new_coeffs

    res = last_coeffs[dim]

    free(last_coeffs)
    free(new_coeffs)

    return res