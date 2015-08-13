import bayesianoracle as bo
import numpy as np
import matplotlib.pyplot as plt
import scipy

# Nonlinear least squares tests

# Number of dimensions
ndim = 2
offset = 0.0
neq = 3

def r(x):
    """ Custom residuals function """
    #return np.array([(np.cos(x[0])+.5), 2*(np.sin(x[1])**2-.1), .5*(x[1]-.5)])
    #return np.array([x[0]**2-np.sin(x[1]-x[0])-.2, 1.4*(x[0]-.1)])
    #return np.array([3*(np.tanh(30*x[0]-x[1]-10)-.1), x[0], x[1]-.5])
    return np.array([3*(np.tanh(10*x[0]-10*x[1]-8)-.1), x[0], x[1]-.5])
    #return np.array([x[0]-2*x[1]+1, np.sin(x[0])+np.cos(x[1]-x[0]**2)**2, .01*np.linalg.norm(x+0.5)])
    #return np.array([x[0]-0.5,x[1]-0.5])

def f(x):
    """ Objective only """
    return np.sum(r(x)**2) - offset

def J(x):
    """ Gradient's of resi.duals """
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
    return f_val, G_val, H_val, r_val, J_val

def plot_f():
    import matplotlib.pyplot as plt
    # Plot for 2d
    num_points = 50
    phi_m = np.linspace(-2, 2, num_points)
    phi_p = np.linspace(-2, 2, num_points)
    X_plot,Y_plot = np.meshgrid(phi_p, phi_m)

    points = np.array([X_plot.flatten().tolist(), Y_plot.flatten().tolist()]).T

    # Get mean values at each of the grid points
    Z = np.zeros(X_plot.shape)
    for i in range(num_points):
        for j in range(num_points):
            Z[i,j] = np.min([f(np.array([X_plot[i,j],Y_plot[i,j]])), 100000])

    fig, ax = plt.subplots()

    p = ax.pcolor(X_plot, Y_plot, Z, cmap='RdBu', vmin=np.array([0.0, Z.min()]).min(), vmax=Z.max())
    cb = fig.colorbar(p, ax=ax)
    plt.scatter(X[:,0], X[:,1])
    plt.savefig("figures/f"+".png")    

def plot(bmao, X, k_fig):
    """ 
    Auxillary plotting function

    args:
    -----
    bmao : bayesianModelAveragingOptimizer object to be plotted
    X  : The values that have been previously traversed
    k_fig  : The suffix seed for saving the figure

    """
    import matplotlib
    matplotlib.use('Agg') # Use to plot without X-forwarding
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    bma = bmao.bma

    # Plot for 2d
    num_points = 100
    phi_p = np.linspace(-3.0, 3.0, num_points)
    phi_m = np.linspace(-3.0, 3.0, num_points)
    X_plot,Y_plot = np.meshgrid(phi_p, phi_m)

    #points = np.array([X_plot.flatten().tolist(), Y_plot.flatten().tolist(), (Y_plot*0).flatten().tolist(), (Y_plot*0).flatten().tolist()]).T
    points = np.array([X_plot.flatten().tolist(), Y_plot.flatten().tolist()]).T

    # Get mean values at each of the grid points
    Z = np.zeros(X_plot.shape)
    S = np.zeros(X_plot.shape)
    K = np.zeros(X_plot.shape)
    N = np.zeros(X_plot.shape)

    Z_rolled = bma.predict_with_unc(points.T)
    k = 0

    for i in range(num_points):
        for j in range(num_points):
            # Mean
            Z[i,j] = Z_rolled[0, k]
            # Unexplained standard deviation
            S[i,j] = Z_rolled[1, k] + 0.000000001
            # Explained standard deviation
            K[i,j] = Z_rolled[2, k] + 0.000000001
            # Effective sample size
            N[i,j] = Z_rolled[3, k] + 0.000000001
            k += 1

    fig, ax = plt.subplots()

    #p = ax.pcolor(X_plot, Y_plot, Z, cmap='RdBu', vmin=Z.min(), vmax=Z.max(), norm=LogNorm(vmin=Z.min(), vmax=Z.max()))
    p = ax.pcolor(X_plot, Y_plot, Z, cmap='RdBu', vmin=0.0, vmax=20.0)
    cb = fig.colorbar(p, ax=ax)
    plt.scatter(X[:,0], X[:,1])
    plt.savefig("figures/bmao_fig_mean_"+str(k_fig)+".png")


    fig, ax = plt.subplots()

    p = ax.pcolor(X_plot, Y_plot, S, cmap='RdBu', vmin=1e-4, vmax=1e4, norm=LogNorm(vmin=1e-4, vmax=1e4))
    cb = fig.colorbar(p, ax=ax)
    plt.scatter(X[:,0], X[:,1])
    plt.savefig("figures/bmao_fig_unexpstd_"+str(k_fig)+".png")

    fig, ax = plt.subplots()

    p = ax.pcolor(X_plot, Y_plot, K, cmap='RdBu', vmin=1e-4, vmax=1e4, norm=LogNorm(vmin=1e-4, vmax=1e4))
    cb = fig.colorbar(p, ax=ax)
    plt.scatter(X[:,0], X[:,1])
    plt.savefig("figures/bmao_fig_expstd_"+str(k_fig)+".png")            

    fig, ax = plt.subplots()

    p = ax.pcolor(X_plot, Y_plot, N, cmap='RdBu', vmin=0, vmax=20)
    cb = fig.colorbar(p, ax=ax)
    plt.scatter(X[:,0], X[:,1])
    plt.savefig("figures/bmao_fig_neff_"+str(k_fig)+".png")            



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

residual = [[] for _ in range(neq)]
residual_grad = [[] for _ in range(neq)]

# Initialize Optimizer
bmao = bo.optimizer.QuadraticBMAOptimizer(ndim, precision_beta=10.0)

# Initialize x_next
x_next = X[0, :]

def func(x):
    f_val, G_val, H_val, r_val, J_val = f_full(x)
    #r_val = r_val +  1.0*np.random.randn(neq)
    f_val = np.sum(r_val**2)
    f_val = f_val + 1.0*np.random.randn()
    G_val = G_val + 1.0*np.random.randn(ndim)
    H_val = H_val + 1.0*np.random.randn(ndim, ndim)
    return f_val, G_val, H_val

import scipy

result = scipy.optimize.minimize(func, x_next, method='BFGS', jac=True)
print(result)

for k in range(20):
    # Update x, and X array
    x = x_next
    if k == 0:
        X = X
    else:
        X = np.vstack((X, x_next))

    print("iter begin")
    print(x)
    # Get y, grad, hess, and update corresponding lists
    f_val, G_val, H_val, r_val, J_val = f_full(x)
    #r_val = r_val +  1.0*np.random.randn(neq)
    f_val = np.sum(r_val**2)
    f_val = f_val + 1.0*np.random.randn()
    G_val = G_val + 1.0*np.random.randn(ndim)
    H_val = H_val + 1.0*np.random.randn(ndim, ndim)
    #H_val = H_val*0.0
    y_test.append(f_val)
    grad_test.append(G_val)
    hess_test.append(H_val)

    print([k, x, f_val, f(x), G_val, H_val, r_val, J_val])

    bmao.add_observation(x, f_val, G_val, H_val)
    print("prediction")
    print(bmao.predict(np.array([[0.0],[0.0]])))
    print(bmao.predict(np.array([[0.5],[-0.3]])))

    x_next = bmao.locate_next_point()
    plot(bmao, X, k)

