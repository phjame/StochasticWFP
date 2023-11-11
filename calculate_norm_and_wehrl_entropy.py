# CREATED BY: Hamza Patwa
# This file calculates the histogram of a generated numerical solution, and uses that as the Wigner function w.
# It then calculates the norm between that and the steady-state solution as in the Gamba et. al. paper


from numpy import array
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from numpy import pi, ravel, shape, reshape
import scipy as sp
import scipy.integrate as integrate


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
	return np.sqrt(-1.0 + (5.0 / np.sqrt(expr)))

def wehrl_entropy(covmatrix):
	return np.log(np.linalg.det(covmatrix)) / 2

	
def main():
	T = 50; dT = 0.01
	ind_stop = int(T/dT)
	ss_cov = np.array([[2/5, 2/10], [2/10, 6/10]])

	u_all = np.load("T_200_dt_0.01_all_history.npy")
	print("Loaded array of all points.")
		
	u_partial = u_all[:ind_stop]

	sigma_val = (1/10.0) * (5-np.sqrt(5))
	sigma_val_largest = (1/10.0) * (5+np.sqrt(5))
	print(np.cov(u_partial[0], rowvar=False))
	initial_norm = norm_wrt_steady_state(np.cov(u_partial[0], rowvar=False))
	print(f"Initial norm: {initial_norm}")
	print(f"Sigma_largest: {sigma_val_largest}")
	print(sigma_val)
		
	def exp_plot_points(sig):
		expfunction_bound = lambda t: np.exp(-t*sig) * initial_norm
		expfunction_tvals = np.arange(0, T, 0.05)
		expfunction_vals = np.array([expfunction_bound(expfunction_tvals[i]) for i in range(np.shape(expfunction_tvals)[0])])
		return (expfunction_tvals, expfunction_vals)
		
	tvalues = np.arange(0, T, 0.01)
	normvalues = np.zeros((np.shape(tvalues)[0]))
	entropyvalues = np.zeros((np.shape(tvalues)[0]))
	"""
	for i in range(np.shape(tvalues)[0]):
		cmatr = np.cov(u_partial[i], rowvar=False)
		print(cmatr)
		normvalues[i] = norm_wrt_steady_state(cmatr)
		print(i)
		print()
	"""
	for i in range(np.shape(tvalues)[0]):
		cmatr = np.cov(u_partial[i], rowvar=False)
		print(cmatr)
		entropyvalues[i] = wehrl_entropy(cmatr)
		print(i)
		print()
	
	entropy_inf = np.log(np.linalg.det(np.array([[3,-1],[-1,2]]))) / 2.0
			
	plt.figure(figsize=(12,12))
	plt.scatter(tvalues, entropyvalues, s=5, label="Entropy of Numerical Solution")
	plt.axhline(y=entropy_inf, color='gray', linestyle='-', label="Entropy of Steady-state solution")
	plt.title("Wehrl Entropy vs. time")
	plt.xlabel("Time")
	plt.ylabel("Entropy")
	plt.legend()
	"""
	plt.scatter(tvalues, normvalues, s=5)
	plt.scatter(*exp_plot_points(sigma_val), c='red', s=3, label="sigma")
	plt.scatter(*exp_plot_points(sigma_val*0.5), c='orange', s=3, label="0.5*sigma")
	plt.scatter(*exp_plot_points(sigma_val*1.5), c='purple', s=3, label="1.5*sigma")
	plt.scatter(*exp_plot_points(sigma_val*2.0), c='brown', s=3, label="2.0*sigma")
	plt.title("L2 norm vs. time")
	plt.xlabel("Time")
	plt.ylabel("L2 norm")
	plt.legend()
	"""
	plt.savefig("entropy_evolution.png")
	plt.show()
	#pltpts = np.array([tvalues, normvalues])
	#np.save("expdecay_of_norm_T_10.npy", pltpts)
	
	
if __name__ == "__main__":
	main()
