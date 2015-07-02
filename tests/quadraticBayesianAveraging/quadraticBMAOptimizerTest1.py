import numpy as np
import bayesianoracle as bo

offset = 0

def r(x):
    """ Custom residuals function """
    return np.array([x[0]-.5, x[1]-.5])
    #return np.array([3*(np.tanh(10*x[0]-10*x[1]-8)-.1), x[0], (x[1]-.5)])
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

def plot(bma, X, mode, k_fig):
    """ Auxillary plotting function
    
    Parameters
    ----------
    bma : bayesianModelAveraging object to be plotted
    X  : The values that have been previously traversed
    mode : mode = "predict" if the GaussianProcessEI prediction is disired
         or mode = "EI" if the expected improvement is desired
    k_fig  : The suffix seed for saving the figure
    
    """
    import matplotlib.pyplot as plt

    # Plot for 2d
    num_points = 50
    phi_m = np.linspace(-5, 5, num_points)
    phi_p = np.linspace(-5, 5, num_points)
    X_plot,Y_plot = np.meshgrid(phi_p, phi_m)

    points = np.array([X_plot.flatten().tolist(), Y_plot.flatten().tolist()]).T

    # Get mean values at each of the grid points
    Z = np.zeros(X_plot.shape)
    S = np.zeros(X_plot.shape)
    U = np.zeros(X_plot.shape)
    if mode == "predict":
        Z_rolled = bma.predict_with_unc(points.T)
        k = 0
        for i in range(num_points):
            for j in range(num_points):
                # Mean
                Z[i,j] = Z_rolled[0, k]
                # Std
                S[i,j] = Z_rolled[1, k]
                # Uncertainty
                U[i,j] = Z_rolled[2, k]
                k += 1

        fig, ax = plt.subplots()
        
        p = ax.pcolor(X_plot, Y_plot, Z, cmap='RdBu', vmin=Z.min(), vmax=Z.max())
        cb = fig.colorbar(p, ax=ax)
        plt.scatter(X[:,0], X[:,1])
        plt.savefig("figures/gp_fig_mean_"+str(k_fig)+".png")

        fig, ax = plt.subplots()
        
        p = ax.pcolor(X_plot, Y_plot, S, cmap='RdBu', vmin=0, vmax=S.max())
        cb = fig.colorbar(p, ax=ax)
        plt.scatter(X[:,0], X[:,1])
        plt.savefig("figures/gp_fig_std_"+str(k_fig)+".png")

        fig, ax = plt.subplots()

        p = ax.pcolor(X_plot, Y_plot, U, cmap='RdBu', vmin=0, vmax=100)
        cb = fig.colorbar(p, ax=ax)
        plt.scatter(X[:,0], X[:,1])
        plt.savefig("figures/gp_fig_unc_"+str(k_fig)+".png")



ndim = 2

# Create optimization process
bmao = bo.optimizer.QuadraticBMAOptimizer(ndim = ndim)

# Number of trials
X = np.array([[-0.7, 0.95]])
n_trials = len(X)

# Initialize x_next
x_next = X[0, :]

for k in range(20):
    # Update x, and X array
    x = x_next
    if k == 0:
        X = X
    else:
        X = np.vstack((X, x_next))
    # Get y, grad, hess, and update corresponding lists
    f_val, G_val, H_val = f_full(x)
    print([k, x, f_val, G_val, H_val])
#    f_val = f_val +  2.0*np.random.randn()
#    G_val = G_val + 2.0*np.random.randn(ndim)
#    H_add = 1.0*np.random.randn(ndim, ndim)
#    H_add = H_add.T.dot(H_add)
#    H_val = H_val + H_add
    print([k, x, f_val, G_val, H_val])
    bmao.add_observation(x, f_val, G_val, H_val)
    
    # Plot result
    plot(bmao, X, "predict", k)
    # plt.show()

    # Get next point
    x_next = bmao.locate_next_point()




