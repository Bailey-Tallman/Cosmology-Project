import corner
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM

# The data is copied into a .txt file
def inputData():
    file = open('lcparam_full_long.txt', 'r')
    lines = []
    z = []
    mb = []
    err = []
    lines = file.readlines()
    for line in lines:
        info = line.split(" ")
        if info[0] == "#name":
            continue
        z.append(float(info[1]))
        mb.append(float(info[4]))
        err.append(float(info[5]))
    return z, mb, err

# defining my intial variables
H_0_1 = 70
Omega_M_1 = 0.3
Omega_L_1 = 0.7

xdata, ydata, yerr = inputData()

xdata = np.array(xdata)
ydata = np.array(ydata)
yerr = np.array(yerr)

print(xdata)
print(ydata)
print(yerr)

def Distance_Modulus(H_0, Omega_M):
    model = []
    cosmo = FlatLambdaCDM(H_0, Omega_M)
    model = cosmo.distmod(xdata).value
    return model


# Using the maximum likelihood to take account my error to get a best fit for the supernovea data
def log_likelihood(theta, xdata, ydata, yerr):
    H_0, Omega_M = theta
    if (H_0 < 0 or Omega_M < 0):
        return -np.inf
    model = []
    cosmo = FlatLambdaCDM(H_0, Omega_M)
    model = cosmo.distmod(xdata).value
    sigma2 = (yerr) ** 2
    return -0.5 * np.sum(((ydata + 19.3 - model) ** 2) / sigma2)

# Fitting of my data to find the parameters: Hubble Constant (H_0) and Omega_Matter (Omega_M)
from scipy.optimize import minimize

nll = lambda *args: -log_likelihood(*args)
initial = np.array([H_0_1, Omega_M_1])
soln = minimize(nll, initial, args=(xdata, ydata, yerr), bounds=([60, 75], [0, 1]))
H_0, Omega_M, = soln.x

print("Maximum likelihood estimates:")
print("H_0 = {0:.3f}".format(H_0))
print("Omega_M = {0:.3f}".format(Omega_M))
print("Omega_L = {0:.3f}".format(1 - Omega_M))

plt.errorbar(xdata, ydata, yerr=yerr, fmt=".k", capsize=0)
plt.plot(xdata, ydata, "k", alpha=0.3, lw=3, label="truth")
plt.plot(xdata, Distance_Modulus(H_0, Omega_M), ":k", label="ML")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

def log_prior(theta):
    H_0, Omega_M = theta
    if 60 < H_0 < 75 and 0 < Omega_M < 1:
        return 0.0
    return -np.inf

def log_probability(theta, xdata, ydata, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, xdata, ydata, yerr)

# Using EMCEE to find the error in my data
import emcee

pos = soln.x + 1e-4 * np.random.randn(32, 2)
nwalkers, ndim = pos.shape

sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability, args=(xdata, ydata, yerr)
)
sampler.run_mcmc(pos, 5000, progress=True);


fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ["H_0", "Omega_M, Omega_L"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");
plt.show()

tau = sampler.get_autocorr_time()
print(tau)

flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
print(flat_samples.shape)

# Using EMCEE data for H_0 and EMCEE to produce the corener plots
fig = corner.corner(
    flat_samples, labels=labels, truths=[H_0, Omega_M]
);

plt.show()
