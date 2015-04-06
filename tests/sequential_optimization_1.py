import bayesianoracle as bo
import numpy as np
import matplotlib.pyplot as plt
import scipy

# Nonlinear least squares tests

# Number of dimensions
ndim = 2
offset = 0

def r(x):
    """ Custom residuals function """
    return np.array([x[0]-.5, x[1]-.5])
    #return np.array([4*x[0], x[1]])

def f(x):
    """ Objective only """
    return np.sum(r(x)**2) - offset

def J(x):
    """ Gradient's of residuals """
    # step size
    h = .00001
    rx = r(x);
    r_grad = []
    I = np.eye(ndim)
    for i in range(ndim):
        # Use finite differencing to get residual graidents
        r_grad.append((r(x+h*I[i,:])-rx)/h)
    return np.array(r_grad).T

def f_full(x):
    """ Compute function val, Gradient, and Gauss-Newton Hessian approximation """
    r_val = r(x)
    f_val = np.sum(r_val**2) - offset
    J_val = J(x)
    G_val = (2*r_val[np.newaxis,:].dot(J_val)).reshape((ndim,))
    H_val = 2*J_val.T.dot(J_val)
    return f_val, G_val, H_val

def plot(gp, X, mode, k_fig):
    """ Auxillary plotting function
    
    Parameters
    ----------
    gp : GaussianProcessEI() object to be plotted
    X  : The values that have been previously traversed
    mode : mode = "predict" if the GaussianProcessEI prediction is disired
         or mode = "EI" if the expected improvement is desired
    k_fig  : The suffix seed for saving the figure
    
    """
    import matplotlib.pyplot as plt

    # Plot for 2d
    num_points = 50
    phi_m = np.linspace(-1, 1, num_points)
    phi_p = np.linspace(-1, 1, num_points)
    X_plot,Y_plot = np.meshgrid(phi_p, phi_m)

    points = np.array([X_plot.flatten().tolist(), Y_plot.flatten().tolist()]).T

    # Get mean values at each of the grid points
    Z = np.zeros(X_plot.shape)
    if mode == "predict":
        file_prefix = "pred"
        Z_rolled = gp.predict(points)[0]
        k = 0
        for i in range(num_points):
            for j in range(num_points):
                Z[i,j] = Z_rolled[k]
                k += 1
    elif mode == "EI":
        file_prefix = "EI"
        Z_rolled = gp.calculate_EI(points, .01, 1)
        k = 0
        for i in range(num_points):
            for j in range(num_points):
                Z[i,j] = Z_rolled[k]
                k += 1
    elif mode == "std":
        file_prefix = "std"
        Z_rolled = gp.predict(points)[1]
        k = 0
        for i in range(num_points):
            for j in range(num_points):
                Z[i,j] = Z_rolled[k]
                k += 1

    fig, ax = plt.subplots()

    p = ax.pcolor(X_plot, Y_plot, Z, cmap='RdBu', vmin=Z.min(), vmax=Z.max())
    cb = fig.colorbar(p, ax=ax)
    plt.scatter(X[:,0], X[:,1])
    plt.savefig("figures/gp_fig_"+file_prefix+"_"+str(k_fig)+".png")

# Number of trials
X = np.array([[-0.7, 0.95]])
n_trials = len(X)

# Initialize storage lists
y_test = []
grad_test = []
hess_test = []
y_err = []
grad_err = []
hess_err = []

# Initialize Gaussian Process
gp1 = bo.optimizer.SequentialOptimizer(ndim, constraints=[(-1,1), (-1,1)], noisy=True)

# Initialize x_next
x_next = X[0, :]
for k in range(10):
    # Update x, and X array
    x = x_next
    if k == 0:
        X = X
    else:
        X = np.vstack((X, x_next))
    # Get y, grad, hess, and update corresponding lists
    f_val, G_val, H_val = f_full(x)
    print([k, x, f_val, G_val, H_val])
    #f_val = f_val +  1.0*np.random.randn()
    #G_val = G_val + 1.0*np.random.randn(ndim)
    #H_val = H_val + 2.0*np.random.randn(ndim, ndim)
    y_test.append(f_val)
    grad_test.append(G_val)
    hess_test.append(H_val)

    # Update errors
    y_err.append(.5)
    grad_err.append([.5] * ndim)
    hess_err.append([[.5] * ndim] * ndim)

    print([k, x, f_val, G_val, H_val])

    # Add data
    gp1.clear_data()
    gp1.add_data(X, y_test, y_err=y_err, der_order=0)
    gp1.add_data(X, grad_test, y_err=grad_err, der_order=1)
    gp1.add_data(X, hess_test, y_err=hess_err, der_order=2)

    # Fit gaussian process
    gp1.fit(50)

    # Plot result
    plot(gp1, X, "predict", k)
    plot(gp1, X, "EI", k)
    plot(gp1, X, "std", k)
    plt.show()

    # Get next point
    x_next = gp1.locate_acquisition_point(x, 0.5)


