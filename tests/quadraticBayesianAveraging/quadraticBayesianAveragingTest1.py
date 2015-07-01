import numpy as np
import bayesianoracle as bo

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
    phi_m = np.linspace(-1, 1, num_points)
    phi_p = np.linspace(-1, 1, num_points)
    X_plot,Y_plot = np.meshgrid(phi_p, phi_m)

    points = np.array([X_plot.flatten().tolist(), Y_plot.flatten().tolist()]).T

    # Get mean values at each of the grid points
    Z = np.zeros(X_plot.shape)
    S = np.zeros(X_plot.shape)
    if mode == "predict":
        Z_rolled = bma.predict(points)
        k = 0
        for i in range(num_points):
            for j in range(num_points):
                # Mean
                Z[i,j] = Z_rolled[k, 0]
                # Std
                S[i,j] = Z_rolled[k, 1]
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


# Create process object
bma = bo.process_objects.QuadraticBMAProcess(ndim = 2)

A = np.array([[20, 0],[0, 50]])
mean = np.array([[0],[0.5]])
fmean = 0
[A, b, d] = bo.process_objects.centered_quad_to_quad(A, mean, fmean)
a = np.array([[0.8],[0.5]])

bma.add_model(A, b, d, a)

X = a.T

A = np.array([[20, 0],[0, 20]])
mean = np.array([[0.01],[0.5]])
fmean = 5
[A, b, d] = bo.process_objects.centered_quad_to_quad(A, mean, fmean)
a = np.array([[0.0],[0.0]])

bma.add_model(A, b, d, a)

X = np.vstack([X, a.T])

"""
A = np.array([[-20, 0],[0, 50]])
mean = np.array([[0],[-0.5]])
fmean = 10
[A, b, d] = bo.process_objects.centered_quad_to_quad(A, mean, fmean)
a = np.array([[0.3],[-0.3]])

bma.add_model(A, b, d, a)

X = np.vstack([X, a.T])
"""

print(X)

#plot(bma, X, 'predict', 1)

print("LOL")
print(bma.predict_single(np.array([[0.0],[0.5]])))
print(bma.predict_single(np.array([[0.01],[0.43]])))
print(bma.predict_single(np.array([[0.0],[0.0]])))
print(bma.predict_single(np.array([[0.0],[0.25]])))

X_test = np.array(np.array([[0.0,0.5],[0.01,0.43], [0.0, 0.0], [0.0, 0.25]]))

print(bma.predict(X_test))
print(bma.predict_single_grad(np.array([[0.0], [0.5]])))
