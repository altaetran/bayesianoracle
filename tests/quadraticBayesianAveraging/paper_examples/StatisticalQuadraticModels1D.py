import numpy as np
import bayesianoracle as bo
import bayesianoracle.plot as boplotter

# Import function information
from function_data import *
execfile("function_data.py")

def plot_prior(bmao, model_ind, beta):
    """ Auxillary plotting function
    
    Parameters
    ----------
    bmao : Bayesian model averaging optimization process
    X  : The values that have been previously traversed
    mode : mode = "predict" if the GaussianProcessEI prediction is disired
         or mode = "EI" if the expected improvement is desired
    k_fig  : The suffix seed for saving the figure
    
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib import colors as cl
    from matplotlib import gridspec

    boplt = boplotter.Plotter1D(x_range=x_range, y_range=y_range, num_points=num_points)
    boplt.set_bma(bmao.bma)

    ### Plot the data and the models
    fig, ax = plt.subplots()

    # Plot the heatmap of probabilties, THEN the function THEN mean line
    boplt.plot_model(ax, model_ind=model_ind, bool_dataless=True, color='k', linestyle='-')
    func_line = boplt.plot_fun(ax, fun)
    mean_line = boplt.plot_model_mean(ax, model_ind=model_ind, color=boplt.colorcycle[model_ind], linestyle='-')

    # Create legend
    legend = plt.legend([mean_line, func_line], 
                        ['Quadratic Approximation', 'True Mean Function'],
                        loc='upper center', bbox_to_anchor=(0.5, 1.075), ncol=1, fancybox=True, shadow=False, scatterpoints=1)

    plt.setp(legend.get_texts(), fontsize=12)

    plt.savefig("StatisticalQuadraticModels1D_figures/"+str(model_ind)+"_"+str(beta)+"_prior.png", dpi=dpi)
    plt.close(fig)

def plot_model(bmao, X, y_hist, model_ind, kernel_range, beta):
    """ Auxillary plotting function
    
    Parameters
    ----------
    bmao : Bayesian model averaging optimization process
    X  : The values that have been previously traversed
    mode : mode = "predict" if the GaussianProcessEI prediction is disired
         or mode = "EI" if the expected improvement is desired
    k_fig  : The suffix seed for saving the figure
    
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib import colors as cl
    from matplotlib import gridspec

    boplt = boplotter.Plotter1D(x_range=x_range, y_range=y_range, num_points=num_points)
    boplt.set_bma(bmao.bma)

    ### Plot the data and the models
    fig, ax = plt.subplots()

    # Plot the heatmap of probabilties, THEN the function THEN mean line THEN data
    boplt.plot_model(ax, model_ind=model_ind, kernel_range=kernel_range, bool_dataless=False, color='k', linestyle='-')
    func_line = boplt.plot_fun(ax, fun)
    mean_line = boplt.plot_model_mean(ax, model_ind=model_ind, color=boplt.colorcycle[model_ind], linestyle='-')
    scat = boplt.plot_data(ax, X, y_hist, bool_color_cycled=True)

    # Create legend
    legend = plt.legend([mean_line, func_line, scat], 
                        ['Quadratic Approximation', 'True Mean Function', 'Data'],
                        loc='upper center', bbox_to_anchor=(0.5, 1.075), ncol=2, fancybox=True, shadow=False, scatterpoints=1)
    legend.legendHandles[2]._sizes = [30]

    plt.setp(legend.get_texts(), fontsize=12)

    plt.savefig("StatisticalQuadraticModels1D_figures/"+str(model_ind)+"_predictive_"+str(kernel_range)+"_"+str(beta)+".png", dpi=dpi)
    plt.close(fig)

bmao = bo.optimizer.QuadraticBMAOptimizer(ndim = 1, 
                                          init_kernel_range=0.25, 
                                          n_int=50,
                                          precision_beta = 1000.0,
                                          constraints = [constr1, constr2],
                                          bounding_box = bounding_box,
                                          bool_compact = True,
                                          kernel_type='Gaussian')

# Simulated sampling of the function.
X = None
y_hist = np.array([])

# Populate bmao
for k in xrange(X_complete.shape[1]):
    # Get next in sequence
    x_next = X_complete[:,k]
    x = x_next
    if k == 0:
        X = np.array([x_next])
    else:
        X = np.hstack([X, np.array([x_next])])
    
    # Get y, grad, hess from precomputed lists
    f = f_complete[k]
    grad = grad_complete[k]
    Hess = Hess_complete[k]

    y_hist = np.append(y_hist, f)
    
    # Add observations to the bmao
    bmao.add_observation(x, f, grad, Hess)

betas = [1000, 10000]
model_inds = [1, 5]
kernel_ranges = [0.1, 0.25, 2.0]


# Try different betas
for beta in betas:
    bmao.set_precision_beta(beta)

    # Try different model indices
    for model_ind in model_inds:
        # Plot the prior
        plot_prior(bmao, model_ind, beta)

        # Try different kernel ranges
        for kernel_range in kernel_ranges:
            # Plot prior and statistical models
            plot_model(bmao, X, y_hist, model_ind, kernel_range, beta)
