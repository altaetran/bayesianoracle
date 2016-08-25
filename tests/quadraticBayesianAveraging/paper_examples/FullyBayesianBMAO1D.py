import numpy as np
import bayesianoracle as bo
import bayesianoracle.plot as boplotter

# Import function information
from function_data import *
execfile("function_data.py")

def plot(bmao, X, k_fig, x_next, y_next):
    """ Auxillary plotting function
    
    Parameters
    ----------
    bma : EnrichedBayesianModelAveraging object to be plotted
    X  : The values that have been previously traversed
    mode : mode = "predict" if the GaussianProcessEI prediction is disired
         or mode = "EI" if the expected improvement is desired
    k_fig  : The suffix seed for saving the figure
    
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib import colors as cl
    from matplotlib import gridspec

    # Plot ranges
    x_range = [-3.0, 3.0]
    y_range = [-150.0, 150.0]

    # Number of x points to plot
    num_points = 200

    # Get the bma
    bma = bmao.bma

    boplt = boplotter.Plotter1D(x_range=x_range, y_range=y_range, num_points=num_points)
    boplt.set_bma(bma)
    boplt.get_bayesian_bma_evals()

    ### Plot the data and the models
    fig, ax = plt.subplots()

    boplt.plot_fun(ax, fun)
    boplt.plot_models(ax)
    boplt.plot_data(ax, bool_color_cycled=True)
    plt.savefig("FullyBayesianBMAO1D_figures/"+str(k_fig)+"_a_obs_.png", dpi=dpi)
    plt.close(fig)

    ### Plot the kernel range likelihoods
    #n_subplot = 2
    #fig = plt.figure(figsize=(8, 8), dpi=dpi)
    #gs = gridspec.GridSpec(n_subplot, 1, height_ratios=[3, 1])
    
    # Create axis array
    #ax = []
    #for i in range(n_subplot):
    #    ax.append(plt.subplot(gs[i]))

    #boplt.plot_bayesian_likelihoods(ax[0])
    #boplt.plot_bayesian_avg_kernel_ranges(ax[1])
    #plt.savefig("FullyBayesianBMAO1D_figures/"+str(k_fig)+"_b_likelihoods_.png", dpi=dpi)
    #plt.close(fig)

    ### Plot the bayesian quantities 
    n_subplot = 3
    fig = plt.figure(figsize=(6, 8), dpi=dpi)
    gs = gridspec.GridSpec(n_subplot, 1, height_ratios=[3, 1, 1])
    
    # Create axis array
    ax = []
    for i in range(n_subplot):
        ax.append(plt.subplot(gs[i]))    

    func_line = boplt.plot_fun(ax[0], fun, xlabel=None)
    mean_line = boplt.plot_means(ax[0], xlabel=None)
    exp_fill, unexp_fill, std_fill = boplt.plot_confidence(ax[0], bool_bayesian=True, xlabel=None)
    #boplt.plot_kernel_weights(ax[1], bool_bayesian=True, ylabel=r'$N_{\gamma}(x)$')
    cbar = boplt.plot_bayesian_likelihoods(ax[1], xlabel=None, ylabel=r'$\gamma$', bool_colorbar=False)

    # upper value of KL to plot
    upper_KL = 0.5

    # Plot the acquisition function if prev phase was detail or explore
    if bmao.get_prev_phase() == 'detail':
        boplt.plot_KL(ax[2], bool_bayesian=True, upper=upper_KL)

        kappa = bmao.get_kappa_detail()
        acq_line = boplt.plot_discounted_means(ax[0], kappa, xlabel=None)
        # Plot points
        scat = boplt.plot_data(ax[0], bool_color_cycled=True, xlabel=None)
        next = boplt.plot_next(ax[0], x_next, y_next, markersize=800)
        legend = plt.legend([exp_fill, unexp_fill, std_fill, mean_line, acq_line, func_line, scat, next], 
                            ['Explained Deviation', 'Unexplained Deviation', 'Total Deviation', 'Estimated Mean Function', 'Acquisition Function', 'True Mean Function', 'Data', 'Next Test Location'], 
                            loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fancybox=True, shadow=False, scatterpoints=1)
        legend.legendHandles[6]._sizes = [30]
        legend.legendHandles[7]._sizes = [30]
        plt.setp(legend.get_texts(), fontsize=8)

    elif bmao.get_prev_phase() == 'exploration':
        boplt.plot_KL(ax[2], bool_bayesian=True, upper=upper_KL)

        kappa = bmao.get_kappa_explore()
        acq_line = boplt.plot_discounted_means(ax[0], kappa, xlabel=None)
        # Plot points
        scat = boplt.plot_data(ax[0], bool_color_cycled=True)
        next = boplt.plot_next(ax[0], x_next, y_next, markersize=800)
        legend = plt.legend([exp_fill, unexp_fill, std_fill, mean_line, acq_line, func_line, scat, next], 
                            ['Explained Deviation', 'Unexplained Deviation', 'Total Deviation', 'Estimated Mean Function', 'Acquisition Function', 'True Mean Function', 'Data', 'Next Test Location'], 
                            loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fancybox=True, shadow=False, scatterpoints=1)
        plt.setp(legend.get_texts(), fontsize=8)
        legend.legendHandles[6]._sizes = [30]
        # Legend size for next point
        legend.legendHandles[7]._sizes = [30]
    elif bmao.get_prev_phase() == "low_density":
        # Plot points
        scat = boplt.plot_data(ax[0], bool_color_cycled=True, xlabel=None)
        next = boplt.plot_next(ax[1], x_next, y_next, markersize=800)

        # Low density, so plot on ax1
        plt.sca(ax[0])
        legend = plt.legend([exp_fill, unexp_fill, std_fill, mean_line, func_line, scat], 
                            ['Explained Deviation', 'Unexplained Deviation', 'Total Deviation', 'Estimated Mean Function', 'True Mean Function', 'Data'], 
                            loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fancybox=True, shadow=False, scatterpoints=1)
        plt.setp(legend.get_texts(), fontsize=8)
        legend.legendHandles[5]._sizes = [30]

        # Set the axis for the second one
        plt.sca(ax[1])
        legend = plt.legend([next], ['Next Test Location'],
                            loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=1, fancybox=True, shadow=False, scatterpoints=1)
        legend.legendHandles[0]._sizes = [30]
        plt.setp(legend.get_texts(), fontsize=8)        

    elif bmao.get_prev_phase() == "low_KL":
        boplt.plot_KL(ax[2], bool_bayesian=True, color='m', linestyle='--', upper=upper_KL)

        # Plot points
        scat = boplt.plot_data(ax[0], bool_color_cycled=True, xlabel=None)
        next = boplt.plot_next(ax[2], x_next, y_next, markersize=800)

        # Low density, so plot on ax2
        plt.sca(ax[0])
        legend = plt.legend([exp_fill, unexp_fill, std_fill, mean_line, func_line, scat], 
                            ['Explained Deviation', 'Unexplained Deviation', 'Total Deviation', 'Estimated Mean Function', 'True Mean Function', 'Data'], 
                            loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fancybox=True, shadow=False, scatterpoints=1)
        plt.setp(legend.get_texts(), fontsize=8)
        legend.legendHandles[5]._sizes = [30]

        # Set the axis for the second one
        plt.sca(ax[2])
        legend = plt.legend([next], ['Next Test Location'],
                            loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=1, fancybox=True, shadow=False, scatterpoints=1)
        legend.legendHandles[0]._sizes = [30]
        plt.setp(legend.get_texts(), fontsize=8)        

    # Remove y tick labels
    ax[0].yaxis.set_ticklabels([])
    # REmove x tick labels
    ax[0].xaxis.set_ticklabels([])
    ax[1].xaxis.set_ticklabels([])
    ax[2].xaxis.set_ticklabels([])

    # Set number of ticks
    ax[2].locator_params(nbins = 3, axis='y')

    plt.savefig("FullyBayesianBMAO1D_figures/"+str(k_fig)+"_c_predictive_.png", dpi=dpi)
    plt.close(fig)


constr1 = lambda x: x+3
constr2 = lambda x: -x+3

bmao = bo.optimizer.QuadraticBMAOptimizer(ndim = 1, 
                                          init_kernel_range=1.0, 
                                          n_int=100,
                                          precision_beta = 100.0,
                                          constraints = [constr1, constr2],
                                          kernel_type='Gaussian')

"""
bmao = bo.optimizer.QuadraticBMAOptimizer(ndim = 1, 
                                          init_kernel_range=1.0, 
                                          n_int=50,
                                          precision_beta = 100.0,
                                          constraints = [constr1, constr2],
                                          bounding_box = bounding_box,
                                          bool_compact = True,
                                          kernel_type='Gaussian')
"""
# Initialize x_next
x_next = np.array([2.5])
X = None

for k in xrange(20):
    # Get next
    x = x_next
    if k == 0:
        X = np.array([x_next])
    else:
        X = np.hstack([X, np.array([x_next])])
    
    # Make up errors
    r1 = (20.0+20*np.sin(x)[0])*(np.random.rand(1,1)[0,0]-0.5)
    r2 = (40.0+40*np.sin(x)[0])*(np.random.rand(1,1)[0,0]-0.5)
    r3 = (60.0+60*np.sin(x)[0])*(np.random.rand(1,1)[0,0]-0.5)

#    r1 = 0.0
#    r2 = 0.0
#    r3 = 0.0

    # Get y, grad, hess, and update corresponding lists
    f, grad, Hess = fun_all(x, r1, r2, r3)
    
    bmao.add_observation(x, f, grad, Hess)
    
    kappa = bmao.kappa_detail

    x_next, y_next = bmao.locate_next_point(bool_return_fval=True)

    # Plot result
    plot(bmao, X, k, x_next, y_next)

    #bmao.bma.predict_bayesian(X)
