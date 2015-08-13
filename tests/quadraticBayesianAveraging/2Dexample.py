import numpy as np
import bayesianoracle as bo

from pprint import pprint as pp

ndim = 2
bmao = bo.optimizer.QuadraticBMAOptimizer(ndim=ndim)

H = np.array([[  1389.4217,   1151.5168],
           [  1151.5168,  36896.3534]])
g = np.array([  643.2191,  6206.7597])
f = 596.83446220293399
xk = np.array([ 0.,  0.])

bmao.add_observation(xk, f, g, H)

for i in xrange(3):
    H = np.array([[  1846.2641,    870.278 ],
                  [   870.278 ,  31874.2671]])
    g = np.array([  -71.6421, -1062.788 ])
    f = 18.327839153167439
    xk = np.array([-0.332, -0.158])

    bmao.add_observation(xk, f, g, H)

"""
H = np.array([[  1846.2641,    870.278 ],
              [   870.278 ,  31874.2671]])
g = np.array([  -71.6421, -1062.788 ])
f = 18.327839153167439
xk = np.array([-0.332, -0.158])

bmao.add_observation(xk, f, g, H)
"""

print("hyperprior")
print(bmao.bma.kernel_prior.a)
print(bmao.bma.kernel_prior.scale)

bmao.optimize_hyperparameters()
#print(bmao.predict_with_unc(np.array([[-3.0797e-01 , -1.2921e-01]]).T))
#print(bmao.bma.estimate_model_weights(np.array([[-3.0797e-01 , -1.2921e-01]]).T, return_likelihoods=True))

print(bmao.predict_with_unc(np.array([xk]).T))
print("lololol")
pp(bmao.bma.estimate_model_weights(np.array([xk]).T, return_likelihoods=True))
print('done')
pp(bmao.bma.calc_relevance_weights(np.array([xk]).T))
#pp(bmao.bma.quadratic_models)
pp(bmao.bma.estimate_model_priors(np.array([xk]).T))
pp(bmao.bma.model_predictions(np.array([xk]).T))

# Get the relevence weights (nModels x p)
relevance_weights = bmao.bma.calc_relevance_weights(np.array([xk]).T)
print("relevance weights")
pp(relevance_weights)

bma = bmao.bma

import matplotlib.pyplot as plt
### Likelihood plots
fig4, ax = plt.subplots(3, sharex=True)
kernel_grid = np.logspace(-2.0, 2.0, num=50)

# Get the likelihoods 
unreg_loglikelihood = np.array([bma.loglikelihood(kernel_range, regularization=False, skew=False) for kernel_range in kernel_grid])
skewness = np.array([bma.estimate_skewness(kernel_range) for kernel_range in kernel_grid])
reg_loglikelihood = np.array([bma.loglikelihood(kernel_range) for kernel_range in kernel_grid])

# Plot the two terms
ll1 = ax[0].plot(kernel_grid, unreg_loglikelihood)
ax[0].set_xscale('log')
ll2 = ax[1].plot(kernel_grid, skewness)
ax[1].set_xscale('log')
ll3 = ax[2].plot(kernel_grid, reg_loglikelihood)
ax[2].set_xscale('log')

pp(reg_loglikelihood)

ax[0].set_xlim([kernel_grid[0],kernel_grid[-1]])
ax[1].set_xlim([kernel_grid[0],kernel_grid[-1]])
ax[2].set_xlim([kernel_grid[0],kernel_grid[-1]])

plt.setp(ll1, color="red", linewidth=3.0, alpha=0.5, linestyle='-',
         dash_capstyle='round')
plt.setp(ll2, color="red", linewidth=3.0, alpha=0.5, linestyle='-',
         dash_capstyle='round')
plt.setp(ll3, color="red", linewidth=3.0, alpha=0.5, linestyle='-',
         dash_capstyle='round')


ax[2].set_xlabel("kernel range",fontsize=16)

plt.savefig("figures/2Dexample_bma_loglikelihood.png")


print("skew")
print(bma.estimate_skewness())
