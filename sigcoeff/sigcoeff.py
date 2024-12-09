from cython_backend import *
from .cuda_backend import *
import torch
import warnings
from numba import NumbaPerformanceWarning

def reorder_path(path, multi_index):
    # Retrieves the path x^{multi_index}
    path_length = path.shape[0]
    new_dim = len(multi_index)
    new_path = torch.empty((path_length, new_dim), dtype=torch.double, device = path.device)
    for i in range(new_dim):
        new_path[:, i] = path[:, multi_index[i]]
    return new_path

def coeff(path, multi_index, dyadic_order = 2, scaling_depth = 2, parallel = True, normalise = True, full = False):
    """
    Computes the signature coefficient S(X)^{multi_index} using the kernel approach.

    :param path: Underlying path. Should be a torch.Tensor of shape (path length, path dimension)
    :param multi_index: Array-like multi-index of signature coefficient to compute. If full = True this is the terminal multi-index in the grid.
    :param dyadic_order: int or 2-tuple. If int, dyadic order is taken to be the same over both dimensions of the PDE grid. Otherwise, dyadic orders taken from tuple (dyadic_order_len, dyadic_order_dim).
    :param scaling_depth: Vandermonde scaling depth.
    :param parallel: If true, computes in parallel when using cpu. This parameter is ignored if X.device = "cuda".
    :param normalise: If true, pre-scales X by max(abs(X)) before computing coefficient. Can improve performance for paths taking large values.
    :param full: If true, returns the full grid of signature coefficients of shape (path length, multi-index length).
    :return: torch.Tensor
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
        return coeff_(path, multi_index, dyadic_order, scaling_depth, parallel, normalise, full)


def coeff_(path, multi_index, dyadic_order = 2, scaling_depth = 2, parallel = True, normalise = True, full = False):
    _multi_index = list(multi_index)
    dim = len(_multi_index)
    path_length = path.shape[0]

    if type(dyadic_order) == int:
        dyadic_order_1, dyadic_order_2 = dyadic_order, dyadic_order
    elif type(dyadic_order) == tuple and len(dyadic_order) == 2:
        dyadic_order_1, dyadic_order_2 = dyadic_order
    else:
        raise ValueError("dyadic_order must by int or tuple of length 2")

    # It is obviously better to return the increment in the dim = 1 case as below,
    # but we ignore this here for consistency of results.
    # if len(_multi_index) == 1:
    #     return X[-1, _multi_index[0]] - X[0, _multi_index[0]]

    if path_length < 2 or dim < 1:
        return torch.tensor(0.)

    path_at_multi_index = reorder_path(path, _multi_index)

    if normalise:
        scaling = torch.max(torch.abs(path))
        path_at_multi_index /= scaling

    if path.device.type == "cpu":
        if not full:
            result = get_coeff_cython(path_at_multi_index.numpy(), scaling_depth, dyadic_order_1, dyadic_order_2, parallel)
        else:
            raise ValueError("full = True not supported with cpu")
    else:
        if full:
            result = get_coeff_cuda(path_at_multi_index.cuda(), scaling_depth, dyadic_order_1, dyadic_order_2, full = True)
        else:
            #get_coeff_cuda_2 does not store the grid so is more memory efficient if we're only interested in one coefficient
            result = get_coeff_cuda_2(path_at_multi_index.cuda(), scaling_depth, dyadic_order_1, dyadic_order_2)

    if normalise:
        if full:
            scaling_pow = float(scaling)
            for i in range(1, dim + 1):
                result[:, i - 1] *= scaling_pow
                scaling_pow *= scaling
        else:
            result *= scaling ** dim

    return result

def coeff_cuda_serial(path, multi_index, dyadic_order=2, scaling_depth=2, normalise=True):
    """
    For timing purposes only. A serial implementation on CUDA for an apples-to-apples comparison.

    :param path: Underlying path. Should be a torch.Tensor of shape (path length, path dimension)
    :param multi_index: Array-like multi-index of signature coefficient to compute. If full = True this is the terminal multi-index in the grid.
    :param dyadic_order: int or 2-tuple. If int, dyadic order is taken to be the same over both dimensions of the PDE grid. Otherwise, dyadic orders taken from tuple (dyadic_order_len, dyadic_order_dim).
    :param scaling_depth: Vandermonde scaling depth.
    :param normalise: If true, pre-scales X by max(abs(X)) before computing coefficient. Can improve performance for paths taking large values.
    :return: torch.Tensor
    """
    if path.device == "cpu":
        raise ValueError("X not on CUDA in cuda_serial")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
        return _coeff_cuda_serial(path, multi_index, dyadic_order, scaling_depth, normalise)

def _coeff_cuda_serial(path, multi_index, dyadic_order=2, scaling_depth=2, normalise=True):
    _multi_index = list(multi_index)
    dim = len(_multi_index)
    path_length = path.shape[0]

    if type(dyadic_order) == int:
        dyadic_order_1, dyadic_order_2 = dyadic_order, dyadic_order
    elif type(dyadic_order) == tuple and len(dyadic_order) == 2:
        dyadic_order_1, dyadic_order_2 = dyadic_order
    else:
        raise ValueError("dyadic_order must by int or tuple of length 2")

    # It is obviously better to return the increment in the dim = 1 case as below,
    # but we ignore this here for consistency of results.
    # if len(_multi_index) == 1:
    #     return X[-1, _multi_index[0]] - X[0, _multi_index[0]]

    if path_length < 2 or path.shape[1] < 1:
        return torch.tensor(0.)

    path_at_multi_index = reorder_path(path, _multi_index)

    if normalise:
        scaling = torch.max(torch.abs(path))
        path_at_multi_index /= scaling

    result = get_coeff_cuda_serial(path_at_multi_index.cuda(), scaling_depth, dyadic_order_1, dyadic_order_2)

    if normalise:
        result *= scaling ** dim

    return result

def coeff_chen_cython(path, multi_index):
    """
    Computes the signature coefficient S(X)^{multi_index} using Chen's relation, computed with Cython.

    :param path: Underlying path. Should be a torch.Tensor of shape (path length, path dimension)
    :param multi_index: Array-like multi-index of signature coefficient to compute
    :return: double
    """
    _multi_index = list(multi_index)
    path_at_multi_index = reorder_path(path, _multi_index)
    return chen_(path_at_multi_index.numpy())

def coeff_chen_cuda(path, multi_index):
    """
    Computes the signature coefficient S(X)^{multi_index} using Chen's relation, computed with CUDA.
    CUDA is used purely for comparability with coeff(x.cuda(), multi_index) and Chen's relation is run serially on one thread.

    :param path: Underlying path. Should be a torch.Tensor of shape (path length, path dimension)
    :param multi_index: Array-like multi-index of signature coefficient to compute
    :return: double
    """

    _multi_index = list(multi_index)
    path_at_multi_index = reorder_path(path, _multi_index)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
        result = chen_cuda_(path_at_multi_index).cpu()

    return float(result)