import numpy as np
import bayesianoracle as bo
import bayesianoracle.plot as boplotter

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors as cl
from matplotlib import gridspec, ticker


# Import function information
from function_data import *
execfile("function_data.py")

def plot_prior(bmao, model_ind, precision_alpha, precision_beta, bias_lambda):
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

    plt.savefig("StatisticalQuadraticModels1D_figures/"+str(model_ind)+"_"+str(precision_alpha)+"_"+str(precision_beta)+"_"+str(bias_lambda)+"_prior.png", dpi=dpi)
    plt.close(fig)

def plot_model(bmao, X, y_hist, model_ind, kernel_range, precision_alpha, precision_beta, bias_lambda):
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
    mean_line = boplt.plot_biased_model_mean(ax, model_ind=model_ind, kernel_range=kernel_range, color=boplt.colorcycle[model_ind], linestyle='-')
    scat = boplt.plot_data(ax, X, y_hist, bool_color_cycled=True)

    # Create legend
    legend = plt.legend([mean_line, func_line, scat], 
                        ['Quadratic Approximation', 'True Mean Function', 'Data'],
                        loc='upper center', bbox_to_anchor=(0.5, 1.075), ncol=2, fancybox=True, shadow=False, scatterpoints=1)
    legend.legendHandles[2]._sizes = [30]

    plt.setp(legend.get_texts(), fontsize=12)

    plt.savefig("StatisticalQuadraticModels1D_figures/"+str(model_ind)+"_predictive_"+str(kernel_range)+"_"+str(precision_alpha)+"_"+str(precision_beta)+"_"+str(bias_lambda)+".png", dpi=dpi)
    plt.close(fig)

def plot_everything(bmao, X, y_hist, model_ind, sets, kernel_range):
    n_subplot = 10
    fig = plt.figure(figsize=(8, 12), dpi=dpi)
    gs = gridspec.GridSpec(n_subplot/2, 2, height_ratios=[10, 10, 10, 10, 1])

    # Make axes for plots
    ax = []
    for i in range(n_subplot-2):
        ax.append(plt.subplot(gs[i]))
    # Make axis for colorbar
    ax_cb = plt.subplot(gs[-1,:])

    row_i = 0
    for (precision_alpha, precision_beta, bias_lambda) in sets:
        # Set parameter
        bmao.set_precision_prior_params(precision_alpha, precision_beta)
        bmao.set_bias_prior_params(bias_lambda)

        boplt = boplotter.Plotter1D(x_range=x_range, y_range=y_range, num_points=num_points)
        boplt.set_bma(bmao.bma)

        # Plot Prior
        heatmap = boplt.plot_model(ax[row_i], model_ind=model_ind, bool_dataless=True, color='k', linestyle='-', bool_colorbar=False, xlabel=None, upper=0.4)
        func_line = boplt.plot_fun(ax[row_i], fun, xlabel=None, color='black')
        mean_line = boplt.plot_model_mean(ax[row_i], model_ind=model_ind, color=boplt.colorcycle[model_ind], linestyle='-', xlabel=None)

        # Plot posterior
        heatmap = boplt.plot_model(ax[row_i+1], model_ind=model_ind, kernel_range=kernel_range, bool_dataless=False, color='k', linestyle='-', bool_colorbar=False, xlabel=None, upper=0.4)
        func_line = boplt.plot_fun(ax[row_i+1], fun, xlabel=None, color='black')
        mean_line = boplt.plot_biased_model_mean(ax[row_i+1], model_ind=model_ind, kernel_range=kernel_range, color=boplt.colorcycle[model_ind], linestyle='-', xlabel=None)
        scat = boplt.plot_data(ax[row_i+1], X, y_hist, bool_color_cycled=False, xlabel=None, edgecolor='white')

        # Add right legend
        #h = plt.ylabel(r'$\alpha='+str(precision_alpha)+r'$'+"\n"+
        #               r'$\beta='+str(precision_beta)+r'$'+"\n"+
        #               r'$\lambda='+str(bias_lambda)+r'$',
        #               rotation=0,
        #               multialignment='left',
        #               horizontalalignment='left',
        #               verticalalignment='center')
        #ax[row_i+1].yaxis.set_label_position("right")
        h = plt.ylabel(r'$\alpha='+str(precision_alpha)+r'$ '+
                       r'$\beta='+str(precision_beta)+r'$ '+
                       r'$\lambda='+str(bias_lambda)+r'$',
                       rotation=270,
                       multialignment='center',
                       verticalalignment='center')

        #ax[row_i+1].yaxis.labelpad = 1.0
        ax[row_i+1].yaxis.set_label_coords(1.05, 0.55)

        # Custom colorbar on axis 2
        cbar = fig.colorbar(heatmap, cax=ax_cb, orientation='horizontal')
        cbar.set_label('probability')
        #tick_locator = ticker.MaxNLocator(nbins=11)
        #cbar.locator = tick_locator
        #cbar.update_ticks()

        row_i+=2

    # Figure legend
    legend = fig.legend([mean_line, func_line, scat], 
                        ['Model Mean', 'True Mean', 'Data'],
                        loc='upper center', bbox_to_anchor=(0.5, 0.97), ncol=2, fancybox=True, shadow=False, scatterpoints=1)
    legend.legendHandles[2]._sizes = [30]

    plt.setp(legend.get_texts(), fontsize=12)

    # Set titles
    ax[0].set_title(r'prior $y, p_{x}\left(y\mid \mathcal{M}, \gamma\right)$', y=1.09)
    ax[1].set_title(r'posterior $y, p_{x}\left(y\mid \mathcal{M},\mathcal{D}, \gamma\right)$', y=1.09)

    # Hide tick labels
    for k in range(len(ax)):
        ax[k].xaxis.set_ticklabels([])
        if k % 2 == 1:
            ax[k].yaxis.set_ticklabels([])

    # Set last x labels
    ax[6].set_xlabel(r'$x$')
    ax[7].set_xlabel(r'$x$')

    plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.075, hspace=0.2)

    plt.savefig("StatisticalQuadraticModels1D_figures/"+str(model_ind)+"_"+str(kernel_range)+"_predictive_all.png", dpi=dpi)

bmao = bo.optimizer.QuadraticBMAOptimizer(ndim = 1, 
                                          init_kernel_range=0.2, 
                                          n_int=1,
                                          precision_beta = 1000.0,
                                          bias_lambda = 1.0,
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

sets = [(2, 1000, 1),
        (1.1, 100, 1),
        (2, 20, 0.01),
        (51.5, 1000, 0.01)]
model_inds = [1]
kernel_range = 0.25

# Try different betas
for model_ind in model_inds:
    plot_everything(bmao, X, y_hist, model_ind, sets, kernel_range)
