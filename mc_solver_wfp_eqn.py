# Monte Carlo Solver of Wigner-Fokker-Planck Equation
# Created by: Hamza Patwa
# Spring 2023
# Research project under Dr. Jose Morales

from numpy import array
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import sys
import argparse

# Constants
hbar = 1; m = 1; w = 1; pi = math.pi
a = math.sqrt(hbar/(m*w)); h = 2*pi

# Standard deviations of Wigner function of Harmonic ground state
sigma_x_W0 = a/math.sqrt(2)
sigma_k_W0 = 1/(math.sqrt(2)*a)

# Total time is t_tot, time step is dt, and number of points is N
#t_tot = 100; dt = 0.1; N = int(1e5)

# Wigner function of harmonic oscillator ground state
def W0(x,k):
    return (2/h) * np.exp((-1*(a**2)*(k**2)) - (x**2)/(a**2))


# Wigner function representing the steady-state analytical solution
def W0_SS(x,k):
    A = (1/5)*(x**2) + (1/5)*(k*x) + (3/10)*(k**2)
    return (1/(2*np.sqrt(5)*pi)) * np.exp(-1*A)


# Evolves all points (x,k) in the array u by time dt.
def evolve_points(u, D, dt):
    # Define the gaussian random variable defined by the diffusion matrix
    mu = np.array([0,0])

    # TODO: do this operation without the for loop by acting directly on numpy array

    for i in range(np.shape(u)[0]):
        # Diffusion
        E = np.random.multivariate_normal(mu, D*dt, 1)

        # Forward Euler evolution
        u[i] = u[i] + np.array([u[i][1], -1*u[i][0] - u[i][1]])*dt + E


def main():
    print("Starting main function")
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", nargs="?", type=float, default=1e4, help="Number of points to sample")
    parser.add_argument("-T", nargs="?", type=float, default=50, help="Total time")
    parser.add_argument("-dT", nargs="?", type=float, default=0.01, help="Time step")
    parser.add_argument("-IC", nargs=3, type=float, default=[0.5,0,0.5], help="Covariance matrix elements of initial condition. Enter the 00, then the 01, then the 11 element space-separated.")
    parser.add_argument("--not_store_final", dest="store_final", default=True, action='store_false', help="Pass this flag if you don't want to store the final plot points")
    parser.add_argument("--not_store_all", dest="store_all", default=True, action='store_false', help="Pass this flag if you don't want to store all the plot points.")
    args = parser.parse_args()

    # MONTE CARLO PARAMETERS
    N = int(args.N)
    t_tot = args.T
    dt = args.dT
    ai, bi, di = args.IC
    
    start_time = time.perf_counter()
    
    print("--------------------------------")
    print("MONTE CARLO PARAMETERS USED IN THIS RUN ARE:")
    print(f"N={N}, T={t_tot}, dT={dt}")
    print(f"Covariance matrix: [{ai}   {bi}")
    print(f"                    {bi}   {di}]")
    print()

    if args.store_all:
            print("Storing all points.")

    if args.store_final:
            print("Storing final points.")

    print("--------------------------------")
    print()
    
    # Wigner function of harmonic oscillator ground state, W0
    #W0 = (2/h) * math.exp(((-1*(a**2)*(p**2))/(hbar**2)) - (x**2)/(a**2))


    # Sample N points (x,k) from W0, which is a 2D gaussian, and put them in the variable u.
    # mu0 contains the mean values of the x and k gaussians of W0, respectively.
    # cov0 contains the standard deviations of the x and k gaussians of W0 along the diagonal elements. Since x and k are indep., off-diagonal elements are 0

    mu0 = np.array([0,0])
    cov0 = np.array([[ai, bi], [bi, di]])
    u = np.random.multivariate_normal(mu0, cov0, N)

    # u_all is the array with the state at all times. The first element of it is the initial distribution. Only define is args.store_all is True.
    if args.store_all:
            u_all = np.zeros((int(t_tot/dt)+1, N, 2))
            u_all[0] = u
    
    # Plot initial distribution (W0)
    fig = plt.figure()
    ax = plt.axes()

    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)

    ax.scatter(u[:,0], u[:,1], s=10, alpha=0.1)
    plt.title("Initial distribution: Harmonic ground state")
    #plt.savefig(f"Images/Initial_Distribution_N_{N}.pdf")
    
    # Conduct the forward Euler evolution for as many steps of dt as it takes to get to t_tot
    count = 0
    start_t = time.time()

    for i in range(int(t_tot/dt)):
            if count % 10 == 0:
                    print(time.time() - start_t)
                    print(count)
            evolve_points(u, np.array([[2,0],[0,2]]), dt)

            # Only store the points if specified by the command line argument
            if args.store_all:
                    u_all[i+1] = u

            count = count + 1

    end_time = time.perf_counter()
    
    # Optionally save final results into NumPy array
    if args.store_final:
            np.save(f"Numerical_Solution_Final_N_{N}_T_{t_tot}_dT_{dt}_{ai}_{bi}_{di}", u)

    # Optionally save results of all plot points into NumPy array
    if args.store_all:
            np.save(f"Numerical_Solution_All_N_{N}_T_{t_tot}_dT_{dt}_IC_{ai}_{bi}_{di}", u_all)

    # Sampling from steady state solution
    mu0_ss = np.array([0,0])
    cov0_ss = np.array([[2/5, 2/10], [2/10, 6/10]])
    u_ss = np.random.multivariate_normal(mu0_ss, np.linalg.inv(cov0_ss), N)

    # Create contours of steady-state solution
    X,Y = np.meshgrid(np.linspace(-6,6,int(10e2)), np.linspace(-6,6,int(10e2)))
    Z = W0_SS(X,Y)

    # Plot numerical solution and steady-state contours
    fig2 = plt.figure(figsize=(12,8))
    ax2 = plt.axes()

    ax2.scatter(u[:,0], u[:,1], s=10, alpha=0.1)
    ax2.contour(X, Y, Z, levels=15, alpha=0.5)
    plt.title(f"Numerical solution with N={N}, T={t_tot}, dT={dt}")
    #plt.savefig(f"Images/Numerical_Solution_N_{N}_T_{t_tot}_dT_{dt}.pdf")

    # Plot (in a separate window) the steady state points and the steady-state contours
    fig3 = plt.figure(figsize=(12,8))
    ax3 = plt.axes()

    ax3.scatter(u_ss[:,0], u_ss[:,1], s=10, alpha=0.1)
    ax3.contour(X, Y, Z, levels=15, alpha=0.5)
    plt.title("Steady-state solution")
    #plt.savefig(f"Images/Steady_State_N_{N}.pdf")
    
    print(f"Program runtime: {end_time-start_time}")
    #plt.show()


if __name__ == "__main__":
    main()
