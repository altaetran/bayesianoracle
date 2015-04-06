import numpy as np
from bayesianoracle import process_objects

""" Simple test for basic functionality of the GaussianProcess object """

ndim = 2

X = np.array([[-.9, .1], [.9, .2], [-.8, .2]])
y = [1.,200., 20.]
grad = [np.array([10., 10.]), np.array([-10., -10.]), np.array([11., 9.0])]

gp = process_objects.EnrichedGaussianProcess(ndim, kernel_type='Matern52', noisy=True)
gp.add_data(X, y, der_order=0)
gp.add_data(X, grad, der_order=1)

gp.fit(10)

gp.clear_data()
gp.add_data(X, y, der_order=0)
gp.add_data(X, grad, der_order=1)
gp.fit(10)

print(gp.predict(X))
