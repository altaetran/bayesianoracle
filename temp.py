import numpy as np

# Now, each point gets one column, rather than one row
mu = np.array([[1, 10],[2, 0], [4, 0]])
Sigma_diag = np.array([[1, 1],[20, 20],[0.001, 0.001]])
# Get the sum of squares along each of the residuals
mu_sum_sq = np.apply_along_axis(np.linalg.norm, 0, mu)**2
Sigma_sum_sq = np.apply_along_axis(np.linalg.norm, 0, Sigma_diag)**2
# Add the terms in place to get the mean.
mean = mu_sum_sq
mean += Sigma_sum_sq
# Get higher order matrix multiplications for the quadratic std
Sigma_Sigma = np.einsum('ij,ji->i', Sigma_diag.T, Sigma_diag)
mu_Sigma_mu = np.einsum('ij,ji->i', mu.T, Sigma_diag*mu)
print Sigma_Sigma
print mu_Sigma_mu
# Add up the results in place
sigma = Sigma_Sigma
sigma *= 2
mu_Sigma_mu *= 4
sigma += mu_Sigma_mu
# Get the square root
sigma **= 0.5

print mean
print sigma
