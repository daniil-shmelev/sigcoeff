import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import plotting_params
plotting_params.set_plotting_params(8, 10, 12)
import iisignature
from sigcoeff import compute_coeff
from tqdm import tqdm

def M_error_imshow(len_x, ax, num):
    torch.manual_seed(42)

    # Parameters
    device = 'cuda'
    dim = 13
    maxdepth = 3
    sample_size = 100

    errmat = np.empty((dim - 1, maxdepth, sample_size))

    for ss in tqdm(range(sample_size)):
        X = torch.rand((len_x, 2), dtype=torch.float64, device=device)
        idx = [0, 1] * int(dim / 2)

        #############################################
        # Get true values from iisignature
        #############################################
        sig = iisignature.sig(np.array(X.cpu()), dim - 1)

        for i in range(1, dim):
            multi_index = idx[:i]
            level = len(multi_index)

            start = iisignature.siglength(2, level - 1) if level > 1 else 0

            res = sig[start: iisignature.siglength(2, level)]
            true_ = res.reshape(tuple([2] * level))[*multi_index]

            #############################################
            # Compute values for different choices of M
            #############################################
            for depth in range(maxdepth):
                estimate = float(compute_coeff(X, np.array(multi_index), dyadic_order=3, M=depth))
                errmat[i - 1, depth, ss] = abs(true_ - estimate)

    errmat = np.mean(errmat, axis=2)

    #############################################
    # Plot
    #############################################
    im = ax.imshow(errmat.T)
    ax.set_yticks(ticks=np.arange(maxdepth), labels=np.arange(0, maxdepth))
    ax.xaxis.set_ticks_position('bottom')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax)

    if num == 1:
        ax.set_xticks(ticks=[], labels=[])
        ax.tick_params(axis='x', which='both', size=0, labelsize=0)
    elif num == 2:
        ax.set_ylabel(r'Scaling Depth $(M)$')
        ax.set_xticks(ticks=[], labels=[])
        ax.tick_params(axis='x', which='both', size=0, labelsize=0)
    else:
        ax.set_xlabel(r'Level $(n)$')
        ax.set_xticks(ticks=np.arange(dim - 1), labels=np.arange(1, dim))

if __name__ == '__main__':
    fig, ax = plt.subplots(3, 1, figsize=(5, 3.5))

    M_error_imshow(100, ax[0], 1)
    M_error_imshow(500, ax[1], 2)
    M_error_imshow(1000, ax[2], 3)

    plt.tight_layout()
    #plt.savefig("..\\plots\\M_error.png", dpi=300)
    #plt.savefig("..\\plots\\M_error.pdf", dpi=300)
    plt.show()