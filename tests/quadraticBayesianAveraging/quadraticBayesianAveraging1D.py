import numpy as np
import bayesianoracle as bo

def plot(bma, X, k_fig):
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

    # Get number of models
    nModels = len(bma.quadratic_models)

    # Plot for 1d
    num_points = 100
    x_plot = np.linspace(-2.5, 2.5, num_points)
    x_grid = np.array([x_plot])

    # Get mean values at each of the grid points
    Z = np.zeros(x_plot.shape)
    S = np.zeros(x_plot.shape)
    
    Z_rolled = bma.predict(x_grid, bool_weights=False)
    Z = Z_rolled[0,:]
    S = Z_rolled[1,:]

    fig, ax = plt.subplots()

    plt.plot(x_plot,Z)
    plt.show()
    plt.savefig("figures/gp_fig_mean_"+str(k_fig)+".png")

    plt.clf()

    fig, ax = plt.subplots(4, sharex=True)

    # Create the figure of all the models
    # Get model predictions
    model_predictions = bma.model_predictions(x_grid)
    for i in range(nModels):
        ax[0].plot(x_plot, model_predictions[i,:])
        f = bma.quadratic_models[i][0]
        print(f)
        print(X)
        ax[0].scatter(X[0,i],f)

    # Show the model priors
    model_priors = bma.estimate_model_priors(x_grid)
    for i in range(nModels):
        ax[1].plot(x_plot, model_priors[i,:])

    # Show the model weights
    model_weights = bma.estimate_model_weights(x_grid)
    for i in range(nModels):
        ax[2].plot(x_plot, model_weights[i,:])

    # Show the kernel values 
    kernel_weights = bma.calc_relevance_weights(x_grid)
    for i in range(nModels):
        ax[3].plot(x_plot, kernel_weights[i,:])

    plt.show()
    plt.savefig("figures/model_.png")

# Create 1 dimensional bma process object
bma = bo.process_objects.QuadraticBMAProcess(ndim = 1)

A = np.array([[5.0]])
mean = np.array([[0.0]])
fmean = 0
[A, b, d] = bo.process_objects.centered_quad_to_quad(A, mean, fmean)
a = np.array([[0.8]])
f = (0.5*a.T.dot(A).dot(a)+b.T.dot(a)+d)[0,0]

bma.add_model(f, A, b, d, a, A)

# X is a n x p stack of points
X = a

A = np.array([[20.0]])
mean = np.array([[-1.0]])
fmean = 5
[A, b, d] = bo.process_objects.centered_quad_to_quad(A, mean, fmean)
a = np.array([[-2.0]])
f = (0.5*a.T.dot(A).dot(a)+b.T.dot(a)+d)[0,0]

bma.add_model(f, A, b, d, a, A)

X = np.hstack([X, a])

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
print(bma.quadratic_models)

plot(bma, X, 1)

print(bma.predict(np.array([[0.0]]), bool_weights=False))

X_test = np.array(np.array([[0.0],[0.01], [-0.1]]).T)

print(bma.predict(X_test, bool_weights=False))
