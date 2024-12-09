import numpy as np
import torch
import matplotlib.pyplot as plt
import plotting_params
import sigcoeff

plotting_params.set_plotting_params(8, 10, 12)
from sigcoeff import coeff
from tqdm import tqdm

if __name__ == '__main__':
    torch.manual_seed(42)

    # Parameters
    len_x = 150
    M = 3
    device = 'cuda'
    dim = 9
    sample_size = 100

    est_2 = np.zeros((7, sample_size))
    est_3 = np.zeros((7, sample_size))
    est_4 = np.zeros((7, sample_size))
    true = np.zeros((7, sample_size))

    depth_error_2 = np.zeros((7, sample_size))
    depth_error_3 = np.zeros((7, sample_size))
    depth_error_4 = np.zeros((7, sample_size))

    n = [i for i in range(1, 8)]

    for ss in tqdm(range(sample_size)):
        X = torch.rand((len_x, dim), dtype=torch.float64, device=device)

        for i in n:
            multi_index = [j for j in range(i)]
            dim_ = np.array(multi_index).max() + 1
            level = len(multi_index)
            X_ = X[:, :dim_]

            #############################################
            # dyadic_order = 2
            #############################################
            estimate_2 = float(coeff(X_, np.array(multi_index), dyadic_order=2, scaling_depth= M))
            est_2[i - 1, ss] = estimate_2

            #############################################
            # dyadic_order = 3
            #############################################
            estimate_3 = float(coeff(X_, np.array(multi_index), dyadic_order=3, scaling_depth=M))
            est_3[i - 1, ss] = estimate_3

            #############################################
            # dyadic_order = 4
            #############################################
            estimate_4 = float(coeff(X_, np.array(multi_index), dyadic_order=4, scaling_depth=M))
            est_4[i - 1, ss] = estimate_4

            #############################################
            # Get true value from chen
            #############################################
            true_ = sigcoeff.coeff_chen_cython(X.cpu(), multi_index)
            true[i - 1, ss] = true_

            depth_error_2[i - 1, ss] = abs(true_ - estimate_2)
            depth_error_3[i - 1, ss] = abs(true_ - estimate_3)
            depth_error_4[i - 1, ss] = abs(true_ - estimate_4)

    avg_depth_error_2 = depth_error_2 / sample_size
    avg_depth_error_2 = np.sum(avg_depth_error_2, axis=1)

    avg_depth_error_3 = depth_error_3 / sample_size
    avg_depth_error_3 = np.sum(avg_depth_error_3, axis=1)

    avg_depth_error_4 = depth_error_4 / sample_size
    avg_depth_error_4 = np.sum(avg_depth_error_4, axis=1)

    true_avg = np.mean(np.abs(true), axis = 1)
    x_2 = np.mean(depth_error_2, axis=1)
    x_3 = np.mean(depth_error_3, axis=1)
    x_4 = np.mean(depth_error_4, axis=1)

    #############################################
    # Plot Absolute Error
    #############################################
    plt.figure(figsize=(4, 3))
    plt.plot(n, x_2, color="mediumblue", marker = 'o')
    plt.plot(n, x_3, color="darkgreen", marker = 'o')
    plt.plot(n, x_4, color="firebrick", marker = 'o')
    plt.legend(["Dyadic order = 2", "Dyadic order = 3", "Dyadic order = 4"])
    plt.title(r"$\text{Dependence of Error on Coefficient Depth}$", )
    plt.xlabel("Coefficient depth $n$")
    plt.ylabel("Absolute error")
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    #plt.savefig("..\\plots\\abs_depth_error.png", dpi=300)
    #plt.savefig("..\\plots\\abs_depth_error.pdf", dpi=300)
    plt.show()

    #############################################
    # Plot Scaled Error
    #############################################
    x_2 /= true_avg
    x_3 /= true_avg
    x_4 /= true_avg

    plt.figure(figsize=(4, 3))
    plt.plot(n, x_2, color="mediumblue", marker = 'o')
    plt.plot(n, x_3, color="darkgreen", marker = 'o')
    plt.plot(n, x_4, color="firebrick", marker = 'o')
    plt.legend(["Dyadic order = 2", "Dyadic order = 3", "Dyadic order = 4"])
    plt.title(r"$\text{Dependence of Error on Coefficient Depth}$", )
    plt.xlabel("Coefficient depth $n$")
    plt.ylabel("Scaled error")
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    #plt.savefig("..\\plots\\scaled_depth_error.png", dpi=300)
    #plt.savefig("..\\plots\\scaled_depth_error.pdf", dpi=300)
    plt.show()