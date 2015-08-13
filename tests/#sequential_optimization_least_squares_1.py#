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

def plot_gpc(gpc, X, k_fig):
    """ Auxillary plotting function to plot a chi square decomposition
    
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
    phi_m = np.linspace(-2, 2, num_points)
    phi_p = np.linspace(-2, 2, num_points)
    X_plot,Y_plot = np.meshgrid(phi_p, phi_m)

    points = np.array([X_plot.flatten().tolist(), Y_plot.flatten().tolist()]).T

    def f_plot(x,y):
        #return gpc.discounted_mean(np.array([[x,y]]), 1.0)[0]
        return gpc.predict(np.array([[x,y]]))[0].tolist()[0]

    # Get mean values at each of the grid points
    Z = np.zeros(X_plot.shape)
    for i in range(num_points):
        for j in range(num_points):
            Z[i,j] = np.min([f_plot(X_plot[i,j],Y_plot[i,j]), 100000])

    fig, ax = plt.subplots()

    p = ax.pcolor(X_plot, Y_plot, Z, cmap='RdBu', vmin=np.array([0.0, Z.min()]).min(), vmax=Z.max())
    cb = fig.colorbar(p, ax=ax)
    plt.scatter(X[:,0], X[:,1])
    plt.savefig("figures/gp_sqsum_"+str(k_fig)+".png")
    plt.show()

def plot_gp(gp, X, mode, k_fig):
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

    def f_plot(x,y):
        return gp.predict([[x,y]])[0].tolist()[0]

    # Get mean values at each of the grid points
    Z = np.zeros(X_plot.shape)
    if mode == "predict":
        file_prefix = "pred"
        for i in range(num_points):
            for j in range(num_points):
                Z[i,j] = np.min([f_plot(X_plot[i,j],Y_plot[i,j]), 100000])
    elif mode == "EI":
        file_prefix = "EI"
        Z_rolled = gp._gen_expected_improvement(points, .01, 1)
        k = 0
        for i in range(num_points):
            for j in range(num_points):
                Z[i,j] = Z_rolled[k]
                k += 1
        #        Z[i,j] = gp._gen_expected_improvement(np.array([[X_plot[i,j],Y_plot[i,j]]]), .01, 1)
           
    #print(Z)

    fig, ax = plt.subplots()

    p = ax.pcolor(X_plot, Y_plot, Z, cmap='RdBu', vmin=abs(Z).min(), vmax=abs(Z).max())
    cb = fig.colorbar(p, ax=ax)
    plt.scatter(X[:,0], X[:,1])
    plt.savefig("figures/gp_fig_"+file_prefix+str(k_fig)+".png")

def plot_residual(gpc, X, res_idx, k_fig):
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

    def f_plot(x,y):
        return gpc.multi_gp[res_idx].predict(np.array([[x,y]]))[0].tolist()[0]

    # Get mean values at each of the grid points
    Z = np.zeros(X_plot.shape)
    for i in range(num_points):
        for j in range(num_points):
            Z[i,j] = np.min([f_plot(X_plot[i,j],Y_plot[i,j]), 100000])

    fig, ax = plt.subplots()

    p = ax.pcolor(X_plot, Y_plot, Z, cmap='RdBu', vmin=Z.min(), vmax=Z.max())
    cb = fig.colorbar(p, ax=ax)
    plt.scatter(X[:,0], X[:,1])
    plt.savefig("figures/res_"+str(res_idx)+"_"+str(k_fig)+".png")

def plot_residuals(gpc, X, k_fig):
    for res_idx in range(neq):
        plot_residual(gpc, X, res_idx, k_fig)

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

# Initialize Gaussian Process
#gp1 = pm.gpei.GaussianProcessEI(ndim, constraints=[(-1,1), (-1,1)], subtract_mode='None')
gpc = bo.optimizer.SequentialLeastSquaresOptimizer(ndim, neq, constraints=[(-1,1), (-1,1)])

plot_f()

# Initialize x_next
x_next = X[0, :]
for k in range(10):
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
    #G_val = G_val + 1.0*np.random.randn(ndim)
    #H_val = H_val + 2.0*np.random.randn(ndim, ndim)
    y_test.append(f_val)
    grad_test.append(G_val)
    hess_test.append(H_val)
    # Update errors
    y_err.append(.5)
    grad_err.append([.5] * ndim)
    hess_err.append([[.5] * ndim] * ndim)

    print([k, x, f_val, f(x), G_val, H_val, r_val, J_val])
    
    # Generate Data
    for res_idx in range(neq):
        residual[res_idx].append(r_val[res_idx])
        residual_grad[res_idx].append(J_val[res_idx,:])

    # Add data to gpc
    for res_idx in range(neq):
        gpc.clear_residual_data(res_idx)
        gpc.add_residual_data(res_idx, X, residual[res_idx])
        gpc.add_residual_data(res_idx, X, residual_grad[res_idx], order=1)
    if k == 0:
        n_starts = 1
    else:
        n_starts = 1
    # Fit data
    gpc.fit(n_starts)
    
    # Plot result
    #print("IMPORTANT")
    #print(gpc.predict(np.array([[0., .5]])))

    plot_residuals(gpc, X, k)
    plot_gpc(gpc, X, k)
    plt.show()
    
    x_next = np.random.rand(1, ndim)
    x_next = x_next[0]
#    print(X)
#    print(residual)
    x_next = gpc.locate_acquisition_point(x0=x, trust_radius=3.)

