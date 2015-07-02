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

H = np.array([[  1846.2641,    870.278 ],
              [   870.278 ,  31874.2671]])
g = np.array([  -71.6421, -1062.788 ])
f = 18.327839153167439
xk = np.array([-0.332, -0.158])

bmao.add_observation(xk, f, g, H)

bmao.optimize_hyperparameters()
print(bmao.bma.kernel_range)
#print(bmao.predict_with_unc(np.array([[-3.0797e-01 , -1.2921e-01]]).T))
#print(bmao.bma.estimate_model_weights(np.array([[-3.0797e-01 , -1.2921e-01]]).T, return_likelihoods=True))

print(bmao.predict_with_unc(np.array([xk]).T))
print(bmao.bma.estimate_model_weights(np.array([xk]).T, return_likelihoods=True))

#pp(bmao.bma.quadratic_models)
pp(bmao.bma.estimate_model_priors(np.array([xk]).T))
pp(bmao.bma.model_predictions(np.array([xk]).T))

# Get the relevence weights (nModels x p)
relevance_weights = bmao.bma.calc_relevance_weights(np.array([xk]).T)
print("relevance weights")
pp(relevance_weights)


# Get the quadratic predictions at each position
def Q(x):
    return np.vstack([(0.5*(A.dot(x)).T.dot(x) + b.T.dot(x) + d)
                      for (y, A, b, d, a, Hchol) in bmao.bma.quadratic_models])
# gets a n_models x n_models matrix with quadratic of model i on observation x_j
model_means_at_obs = np.hstack([Q(a) for (y, A, b, d, a, Hchol) in bmao.bma.quadratic_models])

pp(model_means_at_obs)

# Get the observations made at each model
y_obs = np.hstack([y for (y, A, b, d, a, Hchol) in bmao.bma.quadratic_models])

pp(y_obs)

# Get the J matrix
J_mat = model_means_at_obs - y_obs[None,:]
# Get the J Matrix elementwise square
J_mat_sq = np.square(J_mat)

pp(J_mat_sq)




