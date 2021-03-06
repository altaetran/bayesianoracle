import numpy as np
import bayesianoracle as bo
import bayesianoracle.plot as boplotter

# Import function information
from function_data import *
execfile("function_data.py")

def plot(bmao, X, k_fig):
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
    import matplotlib as mpl
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
    fig = plt.figure(figsize=(8, 4), dpi=dpi)
    ax = plt.gca()

    boplt.plot_fun(ax, fun)
    boplt.plot_models(ax)
    boplt.plot_data(ax, bool_color_cycled=True)
    # Remove tick labels
    #ax.xaxis.set_ticklabels([])
    #ax.yaxis.set_ticklabels([])

    plt.tight_layout()
    plt.savefig("FullyBayesianBMADecomposition_figures/"+str(k_fig)+"_a_obs.png", dpi=dpi)
    plt.close(fig)

    ### Plot the kernel range likelihoods
    n_subplot = 3
    fig = plt.figure(figsize=(8, 6), dpi=dpi)
    gs = gridspec.GridSpec(n_subplot, 1, height_ratios=[10, 10, 1])
    
    # Create axis array
    ax = []
    for i in range(n_subplot):
        ax.append(plt.subplot(gs[i]))

    upper = 0.2

    boplt.plot_bayesian_likelihoods(ax[0], bool_posterior=False, title=r'prior for $\gamma, $'+r'$p_{x}(\gamma)$', bool_colorbar=False, xlabel=None, ylabel=r'$\gamma$', upper=upper)
    heatmap = boplt.plot_bayesian_likelihoods(ax[1], bool_posterior=True, title=r'posterior for $\gamma, $'+r'$p_{x}\left(\gamma\mid D\right)$', bool_colorbar=False, xlabel=r'$x$', ylabel=r'$\gamma$', upper=upper)
    boplt.plot_data_locations(ax[1], bool_color_cycled=True)
    ax[0].xaxis.set_ticklabels([])
    ax[1].xaxis.set_ticklabels([])
    ax[1].set_xlabel(r'$x$')

    # Custom colorbar on axis 2
    cbar = fig.colorbar(heatmap, cax=ax[2], orientation='horizontal')
    cbar.set_label('probability')
    #ax[2].axis('off')
    plt.tight_layout()
    #boplt.plot_bayesian_avg_kernel_ranges(ax[4], ylabel=r'expected $\gamma$')
    plt.savefig("FullyBayesianBMADecomposition_figures/"+str(k_fig)+"_b_likelihoods.png", dpi=dpi)
    plt.close(fig)

    ### Plot the bayesian quantities 
    fig = plt.figure(figsize=(8, 4), dpi=dpi)
    ax = plt.gca()

    func_line = boplt.plot_fun(ax, fun)
    mean_line = boplt.plot_means(ax)
    exp_fill, unexp_fill, std_fill = boplt.plot_confidence(ax, bool_bayesian=True)

    # Plot the acquisition function if prev phase was detail or explore

    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])        

    # Plot points
    scat = boplt.plot_data(ax, bool_color_cycled=True)
    legend = plt.legend([exp_fill, unexp_fill, std_fill, mean_line, func_line, scat], 
                        ['Explained Deviation', 'Unexplained Deviation', 'Total Deviation', 'Estimated Mean Function', 'True Mean Function', 'Data'], 
                        loc='upper center', bbox_to_anchor=(0.5, 1.075), ncol=2, fancybox=True, shadow=False, scatterpoints=1)
    legend.legendHandles[5]._sizes = [30]
    plt.setp(legend.get_texts(), fontsize=8)

    plt.savefig("FullyBayesianBMADecomposition_figures/"+str(k_fig)+"_c_predictive_.png", dpi=dpi)
    plt.close(fig)

    ### Plot the decomposition for different kernel ranges

    def plot_decomposition(kernel_range, label):
        boplt.get_pointwise_bma_evals(kernel_range=kernel_range)

        n_subplot = 6
        fig = plt.figure(figsize=(8, 12), dpi=dpi)
        gs = gridspec.GridSpec(n_subplot, 1, height_ratios=[1, 2, 1, 1, 1, 2])

        ax = []
        for i in range(n_subplot):
            ax.append(plt.subplot(gs[i]))

        boplt.plot_model_priors(ax[0], ylabel="correctness prior\n"+r'$p\left[\mathcal{M}\right](x)$', xlabel=None)
        boplt.plot_fun(ax[1], fun, xlabel=None)
        boplt.plot_visually_weighted_models(ax[1], 'prior', ylabel="prior mean\n"+r'$\mathbf{E}[y](x)$', xlabel=None)
        boplt.plot_data(ax[1], bool_color_cycled=True, xlabel=None)        
        boplt.plot_kernel_weights(ax[2], ylabel="effective\n sample size\n"+r'$N_{\gamma}(x)$', xlabel=None)
        boplt.plot_marginal_likelihoods(ax[3], xlabel=None, ylabel="integrated\n likelihood\n"+r'$p\left[D_{\gamma}\mid \mathcal{M}\right]$')
        boplt.plot_model_posteriors(ax[4], ylabel="posterior\n correctness\n"+r'$p\left[\mathcal{M}\mid D_{\gamma}\right](x)$', xlabel=None)
        boplt.plot_fun(ax[5], fun, xlabel=None)
        boplt.plot_visually_weighted_models(ax[5], 'posterior', ylabel="posterior mean\n"+r'$\mathbf{E}\left[y\mid D_{\gamma}\right]$', xlabel=None)
        boplt.plot_data(ax[5], bool_color_cycled=True, xlabel=r'$x$')

        # Hide tick lines
        ax[0].xaxis.set_ticklabels([])
        ax[1].xaxis.set_ticklabels([])
        ax[2].xaxis.set_ticklabels([])
        ax[3].xaxis.set_ticklabels([])
        ax[4].xaxis.set_ticklabels([])
        ax[5].xaxis.set_ticklabels([])

        # Align the y labels
        y_shift = 0.5
        x_shift = -0.085
        
        for i in xrange(n_subplot):
            ax[i].yaxis.set_label_coords(x_shift, y_shift)

        plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.05)
        plt.savefig("FullyBayesianBMADecomposition_figures/"+str(k_fig)+"_d_decomposition_"+label+".png", dpi=dpi)

        plt.close(fig)

    def plot_sidebyside_decomposition(kernel_range_1, kernel_range_2, label):

        n_subplot = 12
        fig = plt.figure(figsize=(12, 8), dpi=dpi)
        gs = gridspec.GridSpec(n_subplot/2, 2, height_ratios=[1, 2, 1, 1, 1, 2])

        ax = []
        for i in range(n_subplot):
            ax.append(plt.subplot(gs[i]))

        min_log_likelihood = 1e-10
        max_N_eff = 6

        boplt.get_pointwise_bma_evals(kernel_range=kernel_range_1)
        boplt.plot_model_priors(ax[0], ylabel=r'$\mathcal{M}$'+" prior\n"+r'$p_{x}\left(\mathcal{M}\mid \gamma\right)$', xlabel=None)
        boplt.plot_fun(ax[2], fun, xlabel=None)
        boplt.plot_visually_weighted_models(ax[2], 'prior', ylabel="prior mean\n"+r'$\mathbf{E}_{x}[y]$', xlabel=None)
        boplt.plot_data(ax[2], bool_color_cycled=True, xlabel=None)        
        boplt.plot_kernel_weights(ax[4], ylabel="effective\n sample size\n"+r'$N_{\gamma}(x)$', xlabel=None, y_range=[0,max_N_eff])
        boplt.plot_marginal_likelihoods(ax[6], xlabel=None, ylabel="integrated\n likelihood\n"+r'$p_{x}\left(\mathcal{D}\mid \mathcal{M}, \gamma\right)$', y_range=[min_log_likelihood,1])
        boplt.plot_model_posteriors(ax[8], ylabel=r'$\mathcal{M}$ '+"posterior\n"+r'$p\left(\mathcal{M}\mid \mathcal{D}, \gamma\right)$', xlabel=None)
        boplt.plot_fun(ax[10], fun, xlabel=None)
        boplt.plot_visually_weighted_models(ax[10], 'posterior', ylabel="posterior mean\n"+r'$\mathbf{E}_{x}\left[y\mid \mathcal{D}, \gamma\right]$', xlabel=None)
        boplt.plot_data(ax[10], bool_color_cycled=True, xlabel=r'$x$')
        
        ax[0].set_title(r'$\gamma='+str(kernel_range_1)+r'$')

        boplt.get_pointwise_bma_evals(kernel_range=kernel_range_2)
        boplt.plot_model_priors(ax[1], ylabel=None, xlabel=None)
        boplt.plot_fun(ax[3], fun, xlabel=None)
        boplt.plot_visually_weighted_models(ax[3], 'prior', ylabel=None, xlabel=None)
        boplt.plot_data(ax[3], bool_color_cycled=True, xlabel=None)        
        boplt.plot_kernel_weights(ax[5], ylabel=None, xlabel=None, y_range=[0,max_N_eff])
        boplt.plot_marginal_likelihoods(ax[7], xlabel=None, ylabel=None, y_range=[min_log_likelihood,1])
        boplt.plot_model_posteriors(ax[9], ylabel=None, xlabel=None)
        boplt.plot_fun(ax[11], fun, xlabel=None)
        boplt.plot_visually_weighted_models(ax[11], 'posterior', ylabel=None, xlabel=None)
        boplt.plot_data(ax[11], bool_color_cycled=True, xlabel=r'$x$')

        ax[1].set_title(r'$\gamma='+str(kernel_range_2)+r'$')
        
        # Hide tick labels except
        for k in range(n_subplot):
            ax[k].xaxis.set_ticklabels([])
        for k in range(n_subplot):
            if k % 2 == 1:
                ax[k].yaxis.set_ticklabels([])

        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.075, hspace=0.2)

        ax[2].yaxis.set_ticklabels([])
        ax[10].yaxis.set_ticklabels([])

        # Set number of ticks
        ax[0].locator_params(nbins = 3, axis='y')
        ax[1].locator_params(nbins = 3, axis='y')

        ax[4].locator_params(nbins = 3, axis='y')
        ax[5].locator_params(nbins = 3, axis='y')

        ax[8].locator_params(nbins = 3, axis='y')
        ax[9].locator_params(nbins = 3, axis='y')

        # Align the y labels
        y_shift = 0.5
        x_shift = -0.1
        
        for i in xrange(n_subplot):
            ax[i].yaxis.set_label_coords(x_shift, y_shift)

        plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.07)
        plt.savefig("FullyBayesianBMADecomposition_figures/"+str(k_fig)+"_d_sidebyside_decomposition_"+label+".png", dpi=dpi)

        plt.close(fig)

    plot_sidebyside_decomposition(0.2, 1.0, label='0020_0100')
    #plot_decomposition(kernel_range=0.1, label='0010')
    #plot_decomposition(kernel_range=0.25, label='0025')
    #plot_decomposition(kernel_range=1.0, label='0100')
    #plot_decomposition(kernel_range=2.0, label='0200')



bmao = bo.optimizer.QuadraticBMAOptimizer(ndim = 1, 
                                          init_kernel_range=1.0, 
                                          n_int=100,
                                          precision_beta = 1000.0,
                                          bias_lambda = 1.0,
                                          constraints = [constr1, constr2],
                                          bounding_box = bounding_box,
                                          bool_compact = True,
                                          kernel_type='Gaussian')

X = None
y_hist = np.array([])

for k in xrange(X_complete.shape[1]):
    # Get next
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
    
    bmao.add_observation(x, f, grad, Hess)

# Plot result
plot(bmao, X, k)

