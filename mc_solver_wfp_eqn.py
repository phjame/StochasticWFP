# Monte Carlo Solver of Wigner-Fokker-Planck Equation
# Created by: Hamza Patwa
# Spring 2023
# Research project under Dr. Jose Morales

from numpy import array
import numpy as np
import math
import matplotlib.pyplot as plt
import time


# Constants
hbar = 1; m = 1; w = 1; pi = math.pi
a = math.sqrt(hbar/(m*w)); h = 2*pi	

# Standard deviations of Wigner ground state
sigma_x_W0 = a/math.sqrt(2)
sigma_k_W0 = 1/(math.sqrt(2)*a)

# Total time is t_tot, time step is dt, and number of points is N
t_tot = 100; dt = 0.1; N = int(1e3)


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
	
	# Wigner function of harmonic oscillator ground state, W0
	#W0 = (2/h) * math.exp(((-1*(a**2)*(p**2))/(hbar**2)) - (x**2)/(a**2))

	# Sample N points (x,k) from W0, which is a 2D gaussian, and put them in the variable u.
	# mu0 contains the mean values of the x and k gaussians of W0, respectively.
	# cov0 contains the standard deviations of the x and k gaussians of W0 along the diagonal elements. Since x and k are indep., off-diagonal elements are 0.
	
	mu0 = np.array([0,0])
	cov0 = np.array([[sigma_x_W0, 0], [0, sigma_k_W0]])
	u = np.random.multivariate_normal(mu0, cov0, N)
	
	# Plot initial distribution (W0)
	'''
	fig = plt.figure()
	ax = plt.axes()
	
	ax.set_xlim(-6, 6)
	ax.set_ylim(-6, 6)

	ax.scatter(u[:,0], u[:,1], s=10, alpha=0.1)
	plt.show()
	'''
	
	# Conduct the forward Euler evolution for as many steps of dt as it takes to get to t_tot
	
	count = 0	
	#start_t = time.time()
	
	for i in range(int(t_tot/dt)):
		if count % 10 == 0:
		#	print(time.time() - start_t)
			print(count)
		evolve_points(u, np.array([[1,0],[0,1]]), dt)
		count = count + 1
	
	
	"""
	# Sampling from steady state solution
	mu0 = np.array([0,0])
	cov0 = np.array([[2/5, 2/10], [2/10, 6/10]])
	u = np.random.multivariate_normal(mu0, np.linalg.inv(cov0), int(10e4))
	"""

	'''
	# Numerical solution (pre-stored)
	with open("wfp_n_10e4_pts.txt", 'r') as f:
		u_str = f.readlines()
		u = eval(" ".join(u_str).replace("\n", ""))
	'''
	
	# Plotting of contours of the analytical steady-state solution
	X,Y = np.meshgrid(np.linspace(-6,6,int(10e2)), np.linspace(-6,6,int(10e2)))
	Z = W0_SS(X,Y)
	
	# Axes
	fig = plt.figure(figsize=(12,8))
	ax = plt.axes()

	ax.scatter(u[:,0], u[:,1], s=10, alpha=0.2)
	ax.contour(X, Y, Z, levels=15, alpha=0.5, colors='k')
	plt.show()
	

if __name__ == "__main__":
	main()
