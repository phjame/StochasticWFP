# CREATED BY: Hamza Patwa
# This file calculates the histogram of a generated numerical solution, and uses that as the Wigner function w.
# It then calculates the norm between that and the steady-state solution as in the Gamba et. al. paper


from numpy import array
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from numpy import pi, ravel, shape, reshape, sqrt
import argparse
import sys

# Wigner function representing the steady-state analytical solution
def W0_SS(x,k):
    A = (1/5)*(x**2) + (1/5)*(k*x) + (3/10)*(k**2)
    return (1/(2*np.sqrt(5)*pi)) * np.exp(-1*A)


def gaussian_fnct(x, k, covmatrix):
    u = np.array([x,k])
    return (1/(2*pi*np.sqrt(np.linalg.det(covmatrix)))) * np.exp(-(1.0/2.0) * u.T@np.linalg.inv(covmatrix)@u)


def norm_wrt_steady_state(covmatrix):
    a = covmatrix[0][0]
    b = covmatrix[0][1]
    d = covmatrix[1][1]

    expr = (b**2 - a*d) * (-20.0 + (4*a) + (4*b) + (b**2) + (6*d) - (a*d))
    print(-1.0 + (5.0 / np.sqrt(expr)))
    return np.sqrt(-1.0 + (5.0 / np.sqrt(expr)))

def wehrl_entropy(covmatrix, s):
    a = covmatrix[0][0]
    b = covmatrix[0][1]
    d = covmatrix[1][1]

    determinant = -(b**2) + (a*d)

    a_inv = d / determinant
    b_inv = -b / determinant
    d_inv = a / determinant

    # Just renaming for convenience
    a = a_inv
    b = b_inv
    d = d_inv

    N = d + (d*a*(s**2)) + (4*(s**2)) + (4*a*(s**4)) - ((b**2)*(s**2))
    new_a = a*d + (4*a*(s**2)) - (b**2)
    new_b = 4*b*(s**2)
    new_d = (4*d*(s**2)) + (4*d*a*(s**4)) - (4*(b**2)*(s**4))

    new_covmatrix_inverse = (1.0/N) * np.array([[new_a, new_b], [new_b, new_d]])

    return -1 * np.log(np.linalg.det(new_covmatrix_inverse)) / 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", help="Input file")
    parser.add_argument("-T", type=float, help="Time to plot until.")
    args = parser.parse_args()

    T = args.T
    dT = 0.01

    ind_stop = int(T/dT)
    ss_cov = np.array([[2/5, 2/10], [2/10, 6/10]])

    u_all = np.load(args.f)
    print("Loaded array of all points.")

    u_partial = u_all[:ind_stop]

    sigma_val = (1/10.0) * (5-np.sqrt(5))
    sigma_val_largest = (1/10.0) * (5+np.sqrt(5))
    ic_cov = np.cov(u_partial[0], rowvar=False)
    print(ic_cov)
    print(wehrl_entropy(ic_cov, 1/np.sqrt(2)))

    initial_norm = norm_wrt_steady_state(np.cov(u_partial[0], rowvar=False))

    def exp_plot_points(sig):
        expfunction_bound = lambda t: np.exp(-t*sig) * initial_norm
        expfunction_tvals = np.arange(0, T, 0.05)
        expfunction_vals = np.array([expfunction_bound(expfunction_tvals[i]) for i in range(np.shape(expfunction_tvals)[0])])
        return (expfunction_tvals, expfunction_vals)

    tvalues = np.arange(0, T, 0.01)
    normvalues = np.zeros((np.shape(tvalues)[0]))
    entropyvalues = np.zeros((np.shape(tvalues)[0]))

    for i in range(np.shape(tvalues)[0]):
        cmatr = np.cov(u_partial[i], rowvar=False)
        entropyvalues[i] = wehrl_entropy(cmatr, 1/np.sqrt(2))
        normvalues[i] = norm_wrt_steady_state(cmatr)

    entropy_inf = wehrl_entropy(np.array([[3,-1],[-1,2]]), 1/sqrt(2))

    # Entropy evolution without error bars
    plt.figure(figsize=(12,12))
    plt.xlim(0, T)
    plt.scatter(tvalues, entropyvalues, s=5, label="Entropy of Numerical Solution")
    plt.axhline(y=entropy_inf, color='gray', linestyle='-', label="Entropy of Steady-state solution")
    plt.title("Wehrl Entropy vs. time")
    plt.xlabel("Time")
    plt.ylabel("Entropy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{args.f[:-4]}_entropy_evolution_no_error_bars.pdf")

    # Entropy evolution with error bars
    plt.figure(figsize=(12,12))
    plt.xlim(0, T)
    plt.errorbar(tvalues, entropyvalues, yerr=np.array([0.03]*tvalues.shape[0]), color='lightblue', alpha=0.3)
    plt.scatter(tvalues, entropyvalues, s=5, label="Entropy of Numerical Solution", c='lightblue')
    plt.axhline(y=entropy_inf, color='gray', linestyle='-', label="Entropy of Steady-state solution")
    plt.title("Wehrl Entropy vs. time")
    plt.xlabel("Time")
    plt.ylabel("Entropy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{args.f[:-4]}_entropy_evolution_with_error_bars.pdf")

    # Norm evolution
    plt.figure(figsize=(12,12))
    plt.xlim(0, T)

    print(np.cov(u_partial[0], rowvar=False))
    initial_norm = norm_wrt_steady_state(np.cov(u_partial[0], rowvar=False))
    print(f"Initial norm: {initial_norm}")
    print(f"Sigma_largest: {sigma_val_largest}")

    plt.plot(*exp_plot_points(sigma_val), c='red', label="Analytical Upper Bound")
    plt.scatter(tvalues, normvalues, s=5)
    plt.title("L2 norm vs. time")
    plt.xlabel("Time")
    plt.ylabel("L2 norm")
    plt.legend()

    #plt.savefig(f"{args.f[:-4]}_L2_norm_evolution_T_{T}.pdf")
    plt.show()

    #pltpts = np.array([tvalues, normvalues])
    #np.save("expdecay_of_norm_T_10.npy", pltpts)


if __name__ == "__main__":
    main()
