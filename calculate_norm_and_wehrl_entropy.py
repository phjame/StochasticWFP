# CREATED BY: Hamza Patwa (2023)
# This program calculates the L2 norm given in the Gamba, Gualdani, and Sharp (2009) paper for a set of coherent states that obey the WFP equation.
# It also calculates the time-evolution of the Wehrl entropy of these states.


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


# L2 norm
def norm_wrt_steady_state(covmatrix):
        a = covmatrix[0][0]
        b = covmatrix[0][1]
        d = covmatrix[1][1]

        expr = (b**2 - a*d) * (-20.0 + (4*a) + (4*b) + (b**2) + (6*d) - (a*d))
        print(-1.0 + (5.0 / np.sqrt(expr)))
        return np.sqrt(-1.0 + (5.0 / np.sqrt(expr)))


# Wehrl entropy
def wehrl_entropy(covmatrix, s):

        # a, b, and d are the elements of the inverse covariance matrix of the Wigner function
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

        # Calculation of the inverse covariance matrix elements of the Husimi function based on analytical calculations done separately
        N = d + (d*a*(s**2)) + (4*(s**2)) + (4*a*(s**4)) - ((b**2)*(s**2))
        new_a = a*d + (4*a*(s**2)) - (b**2)
        new_b = 4*b*(s**2)
        new_d = (4*d*(s**2)) + (4*d*a*(s**4)) - (4*(b**2)*(s**4))

        # The covariance matrix of the Husimi function calculated from the Wigner function
        new_covmatrix_inverse = (1.0/N) * np.array([[new_a, new_b], [new_b, new_d]])

        # Returning the Werhl entropy of the Husimi function's covariance matrix
        return -1 * np.log(np.linalg.det(new_covmatrix_inverse)) / 2


def main():

        parser = argparse.ArgumentParser()
        parser.add_argument("-f", help="Input .npy (NumPy) file containing an array of shape (N_t, N_p, 2), where N_t is the number of time steps, N_p is the number of points for Monte Carlo, and 2 is the phase space dimension. This file can be generated using the 'mc_solver_wfp_eqn.py' program.")
        parser.add_argument("-T", type=float, help="Time to plot until (cannot exceed N_t * dT).")
        args = parser.parse_args()

        # Assumes dT = 0.01
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

        initial_norm = norm_wrt_steady_state(np.cov(u_partial[0], rowvar=False))

        # Generates x and y data points for plotting an exponential. For L2 norm only
        def exp_plot_points(sig):
                expfunction_bound = lambda t: np.exp(-t*sig) * initial_norm
                expfunction_tvals = np.arange(0, T, 0.05)
                expfunction_vals = np.array([expfunction_bound(expfunction_tvals[i]) for i in range(np.shape(expfunction_tvals)[0])])
                return (expfunction_tvals, expfunction_vals)

        tvalues = np.arange(0, T, 0.01)
        normvalues = np.zeros((np.shape(tvalues)[0]))
        entropyvalues = np.zeros((np.shape(tvalues)[0]))

        # Loop through all time steps and calculate L2 norm and Wehrl entropy of each covariance matrix
        for i in range(np.shape(tvalues)[0]):

                # Covariance matrix from points (assumes 0 mean)
                cmatr = np.cov(u_partial[i], rowvar=False)

                # Entropy and L2 norm calculation
                entropyvalues[i] = wehrl_entropy(cmatr, 1/np.sqrt(2))
                normvalues[i] = norm_wrt_steady_state(cmatr)


        # Entropy of steady-state
        entropy_inf = wehrl_entropy(np.array([[3,-1],[-1,2]]), 1/sqrt(2))


        # WEHRL ENTROPY ---------------------------------------------------------------------------------
        plt.figure(figsize=(12,12))
        plt.xlim(0, T)

        plt.scatter(tvalues, entropyvalues, s=5, label="Entropy of Numerical Solution")
        plt.axhline(y=entropy_inf, color='gray', linestyle='-', label="Entropy of Steady-state solution")
        plt.title("Wehrl Entropy vs. time")
        plt.xlabel("Time")
        plt.ylabel("Entropy")
        plt.legend()

        plt.savefig(f"{args.f[:-4]}_entropy_evolution_T_{T}.pdf")

        # L2 NORM ---------------------------------------------------------------------------------------
        plt.figure(figsize=(12,12))
        plt.xlim(0, T)

        plt.plot(*exp_plot_points(sigma_val), c='red', label="Analytical Upper Bound")
        plt.scatter(tvalues, normvalues, s=5)
        plt.title("L2 norm vs. time")
        plt.xlabel("Time")
        plt.ylabel("L2 norm")
        plt.legend()

        plt.savefig(f"{args.f[:-4]}_L2_norm_evolution_T_{T}.pdf")


if __name__ == "__main__":
        main()
