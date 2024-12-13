<h1 align='center'>sigcoeff</h1>
<h2 align='center'>Sparse Signature Coefficient Recovery via Kernels</h2>

This package provides an example implementation for computing deep coefficients of the path signature transform using
signature kernels on both CPU and GPU, to accompany the paper
_[Sparse Signature Coefficient Recovery via Kernels](https://arxiv.org/abs/2412.08579)_. This method allows for 
parallelised computation of coefficients, with a time complexity independent of their depth.

---

## Installation

```bash
pip install git+https://github.com/daniil-shmelev/sigcoeff.git
```

## How to use the package

To run the below, see `./examples/example_usage.py`

```python
import torch
import sigcoeff

torch.manual_seed(42)

# Underlying path
len_x = 1000
dim = 10
X = torch.rand((len_x, dim), device="cpu")

# Target multi-index
multi_index = [1, 5, 2, 6, 3]

# Algorithm parameters
M = 2
dyadic_order = 3

#############################################
# Serial CPU computation
#############################################
coeff = sigcoeff.coeff(X, multi_index, scaling_depth=M, dyadic_order=dyadic_order, parallel=False)

#############################################
# Parallel CPU computation
#############################################
coeff = sigcoeff.coeff(X, multi_index, scaling_depth=M, dyadic_order=dyadic_order, parallel=True)

#############################################
# Parallel GPU computation
#############################################
coeff = sigcoeff.coeff(X.cuda(), multi_index, scaling_depth=M, dyadic_order=dyadic_order)

#############################################
# Varying dyadic orders for the two dimensions of the PDE grid
#############################################
coeff = sigcoeff.coeff(X.cuda(), multi_index, scaling_depth=M, dyadic_order=(3, 2))

#############################################
# Extraction of the entire grid of coefficients. I.e. all coefficients given by multi_index[:i],
# evaluated at all time points up to len_x. (CUDA only)
#############################################
coeff_grid = sigcoeff.coeff(X.cuda(), multi_index, scaling_depth=M, dyadic_order=dyadic_order, full=True)
```

## Citation

```bibtex
@article{shmelev2024sparsesig,
  title={Sparse Signature Coefficient Recovery via Kernels},
  author={Shmelev, Daniil and Cristopher, Salvi},
  journal={arXiv preprint arXiv:2412.08579},
  year={2024}
}
```

<!-- 
-->

