import bayesianoracle as bo
import numpy as np


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
    u = np.zeros(X_plot.shape)
    if mode == "predict":
        Z_rolled = bma.predict(points.T, bool_weights=False)
        k = 0
        for i in range(num_points):
            for j in range(num_points):
                # Mean
                Z[i,j] = Z_rolled[0, k]
                # disagreement
                u[i,j] = Z_rolled[1, k]
                k += 1

    fig, ax = plt.subplots()

    p = ax.pcolor(X_plot, Y_plot, Z, cmap='RdBu', vmin=Z.min(), vmax=Z.max())
    cb = fig.colorbar(p, ax=ax)
    plt.scatter(X[:,0], X[:,1])
    plt.savefig("figures/gp_fig_mean_"+str(k_fig)+".png")

    fig, ax = plt.subplots()

    p = ax.pcolor(X_plot, Y_plot, u, cmap='RdBu', vmin=0, vmax=u.max())
    cb = fig.colorbar(p, ax=ax)
    plt.scatter(X[:,0], X[:,1])
    plt.savefig("figures/gp_fig_std_"+str(k_fig)+".png")


nDim = 2
bma = bo.process_objects.QuadraticBMAProcess(nDim)

A1 = np.array([[20, 0],[0, 50]])
mean1 = np.array([[0],[0.5]])
fmean1 = 0
[A1, b1, d1] = bo.process_objects.centered_quad_to_quad(A1, mean1, fmean1)
a1 = np.array([[0.8],[0.5]])

fa1 = (0.5*(A1.dot(a1)).T.dot(a1) + b1.T.dot(a1) + d1)[0,0]

bma.add_model(fa1, A1, b1, d1, a1, np.linalg.inv(np.linalg.cholesky(A1)))

X = a1.T

A2 = np.array([[20, 0],[0, 20]])
mean2 = np.array([[0.01],[0.5]])
fmean2 = 5
[A2, b2, d2] = bo.process_objects.centered_quad_to_quad(A2, mean2, fmean2)
a2 = np.array([[0.0],[0.0]])

fa2 = (0.5*(A2.dot(a2)).T.dot(a2) + b2.T.dot(a2) + d2)[0,0]
bma.add_model(fa2, A2, b2, d2, a2, np.linalg.inv(np.linalg.cholesky(A2)))

X = np.vstack([X, a2.T])

# Cross model predictions
fa1_2 = (0.5*(A1.dot(a2)).T.dot(a2) + b1.T.dot(a2) + d1)[0,0]
fa2_1 = (0.5*(A2.dot(a1)).T.dot(a1) + b2.T.dot(a1) + d2)[0,0]

print(np.array([[fa1, fa1_2],[fa2_1, fa2]]))

print("LOL")
#
#print(bma.calc_relevance_weights(np.array([[0.0,0.8, 0.4, 1.0],[0.0,0.5, 0.25, 0.0]])))
#print(bma.estimate_model_priors(np.array([[0.0,0.8, 0.4, 1.0],[0.0,0.5, 0.25, 0.0]])))
#print(bma.estimate_model_weights(np.array([[0.0,0.8, 0.4, 1.0],[0.0,0.5, 0.25, 0.0]])))
print(bma.predict(np.array([[0.0,0.8, 0.4, 1.0],[0.0,0.5, 0.25, 0.0]]), bool_weights=True))
#print(bma.predict_single(np.array([[0.0],[0.5]])))
#print(bma.predict_single(np.array([[0.01],[0.43]])))
#print(bma.predict_single(np.array([[0.0],[0.0]])))
#print(bma.predict_single(np.array([[0.0],[0.25]])))

plot(bma, X, 'predict', 1)
