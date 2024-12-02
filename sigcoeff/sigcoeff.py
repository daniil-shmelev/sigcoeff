from cython_backend import *
from .cuda_backend import *
import torch
import warnings
from numba import NumbaPerformanceWarning

def reorder_path(X, multi_index):
    # Retrieves the path x^multi_index
    len_x = X.shape[0]
    new_dim = len(multi_index)
    new_X = torch.empty((len_x, new_dim), dtype=torch.double, device = X.device)
    for i in range(new_dim):
        new_X[:, i] = X[:, multi_index[i]]
    return new_X

def compute_coeff(X, multi_index, dyadic_order = 2, M = 2, parallel = True, scale = True, full = False):
    """
    Computes the signature coefficient S(X)^{multi_index} using the kernel approach.

    :param X: Underlying path. Should be a torch.Tensor of shape (path length, path dimension)
    :param multi_index: Array-like multi-index of signature coefficient to compute. If full = True this is the terminal multi-index in the grid.
    :param dyadic_order: int or 2-tuple. If int, dyadic order is taken to be the same over both dimensions of the PDE grid. Otherwise, dyadic orders taken from tuple (dyadic_order_len, dyadic_order_dim).
    :param M: Vandermonde scaling depth.
    :param parallel: If true, computes in parallel when using cpu. This parameter is ignored if X.device = "cuda".
    :param scale: If true, pre-scales X by max(abs(X)) before computing coefficient. Can improve performance for paths taking large values.
    :param full: If true, returns the full grid of signature coefficients of shape (path length, multi-index length).
    :return: torch.Tensor
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
        return compute_coeff_(X, multi_index, dyadic_order, M, parallel, scale, full)


def compute_coeff_(X, multi_index, dyadic_order = 2, M = 2, parallel = True, scale = True, full = False):
    multi_index_ = list(multi_index)
    dim = len(multi_index_)

    if type(dyadic_order) == int:
        dyadic_order_1, dyadic_order_2 = dyadic_order, dyadic_order
    elif type(dyadic_order) == tuple and len(dyadic_order) == 2:
        dyadic_order_1, dyadic_order_2 = dyadic_order
    else:
        raise ValueError("dyadic_order must by int or tuple of length 2")

    # if len(multi_index_) == 1:
    #     return X[-1, multi_index_[0]] - X[0, multi_index_[0]]

    if X.shape[0] < 2 or X.shape[1] < 1:
        return torch.tensor(0.)

    new_X = reorder_path(X, multi_index_)

    if scale:
        scaling = torch.max(torch.abs(X))
        new_X /= scaling

    if X.device.type == "cpu":
        if not full:
            res = get_coeff_cython(new_X.numpy(), M, dyadic_order_1, dyadic_order_2, parallel)
        else:
            raise ValueError("full = True not supported with cpu")
    else:
        if full:
            res = get_coeff_cuda(new_X.cuda(), M, dyadic_order_1, dyadic_order_2, full = True)
        else:
            #get_coeff_cuda_2 does not store the grid so is more memory efficient if we're only interested in one coefficient
            res = get_coeff_cuda_2(new_X.cuda(), M, dyadic_order_1, dyadic_order_2)

    if scale:
        if full:
            scaling_pow = float(scaling)
            for i in range(1, dim + 1):
                res[:, i - 1] *= scaling_pow
                scaling_pow *= scaling
        else:
            res *= scaling ** dim

    return res

def cuda_serial(X, multi_index, dyadic_order = 2, M = 2, parallel = True, scale = True, full = False):
    """
    For timing purposes only. A serial implementation on CUDA for an apples-to-apples comparison.

    :param X: Underlying path. Should be a torch.Tensor of shape (path length, path dimension)
    :param multi_index: Array-like multi-index of signature coefficient to compute. If full = True this is the terminal multi-index in the grid.
    :param dyadic_order: int or 2-tuple. If int, dyadic order is taken to be the same over both dimensions of the PDE grid. Otherwise, dyadic orders taken from tuple (dyadic_order_len, dyadic_order_dim).
    :param M: Vandermonde scaling depth.
    :param parallel: If true, computes in parallel when using cpu. This parameter is ignored if X.device = "cuda".
    :param scale: If true, pre-scales X by max(abs(X)) before computing coefficient. Can improve performance for paths taking large values.
    :param full: If true, returns the full grid of signature coefficients of shape (path length, multi-index length).
    :return: torch.Tensor
    """
    if X.device == "cpu":
        raise ValueError("X not on CUDA in cuda_serial")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
        return cuda_serial_(X, multi_index, dyadic_order, M, parallel, scale, full)

def cuda_serial_(X, multi_index, dyadic_order = 2, M = 2, parallel = True, scale = True, full = False):
    multi_index_ = list(multi_index)
    dim = len(multi_index_)

    if type(dyadic_order) == int:
        dyadic_order_1, dyadic_order_2 = dyadic_order, dyadic_order
    elif type(dyadic_order) == tuple and len(dyadic_order) == 2:
        dyadic_order_1, dyadic_order_2 = dyadic_order
    else:
        raise ValueError("dyadic_order must by int or tuple of length 2")

    # if len(multi_index_) == 1:
    #     return X[-1, multi_index_[0]] - X[0, multi_index_[0]]

    if X.shape[0] < 2 or X.shape[1] < 1:
        return torch.tensor(0.)

    new_X = reorder_path(X, multi_index_)

    if scale:
        scaling = torch.max(torch.abs(X))
        new_X /= scaling

    res = get_coeff_cuda_serial(new_X.cuda(), M, dyadic_order_1, dyadic_order_2)

    if scale:
        res *= scaling ** dim

    return res

def chen_cython(X, multi_index):
    """
    Computes the signature coefficient S(X)^{multi_index} using Chen's relation, computed with Cython.

    :param X: Underlying path. Should be a torch.Tensor of shape (path length, path dimension)
    :param multi_index: Array-like multi-index of signature coefficient to compute
    :return: double
    """
    multi_index_ = list(multi_index)
    new_X = reorder_path(X, multi_index_)
    return chen_(new_X.numpy())

def chen_cuda(X, multi_index):
    """
    Computes the signature coefficient S(X)^{multi_index} using Chen's relation, computed with CUDA.

    :param X: Underlying path. Should be a torch.Tensor of shape (path length, path dimension)
    :param multi_index: Array-like multi-index of signature coefficient to compute
    :return: double
    """

    multi_index_ = list(multi_index)
    new_X = reorder_path(X, multi_index_)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
        res = chen_cuda_(new_X).cpu()

    return float(res)