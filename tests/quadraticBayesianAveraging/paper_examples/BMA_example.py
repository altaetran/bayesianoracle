import numpy as np
import bayesianoracle as bo

def plot(bmap, X, k_fig, x_next, kappa):
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

    # Get the bma
    bma = bmap.bma

    # Get number of models
    nModels = len(bma.quadratic_models)

    # Plot ranges
    x_min = -5.0
    x_max = 5.0
    y_min = -200
    y_max = 200

    colorcycle = ['r', 'b', 'g', 'm', 'y']

    # Plot for 1d
    num_points = 200
    # mini plot dx param
    dx = 0.5
    # miniplot num points
    mini_num_points = 50
    
    dpi = 300
    
    # Create x array to be plotted
    x_plot = np.linspace(x_min, x_max, num_points)
    x_grid = np.array([x_plot])

    # Get mean values at each of the grid points
    
    F = np.zeros(x_plot.shape)
    for i in range(len(x_plot)):
        F[i] = fun(np.array([x_plot[i]]))

    Z_rolled = bma.predict_with_unc(x_grid)
    model_weights, errors, N_eff, marginal_likelihoods = bma.estimate_model_weights(x_grid, return_likelihoods=True)
    Z = Z_rolled[0,:]

    # bma disagreement
    S = Z_rolled[1,:]
    # bma uncertainty
    U = Z_rolled[2,:]

    # Get full variance
    std = np.sqrt(np.square(S) + np.square(U))

    def plt_func(ax):
        # Plot function
        func_line = ax.plot(x_plot, F)
        # Set line properties 
        plt.setp(func_line, color="black", linewidth=2.0, alpha=0.5)

    def plt_data(ax, X):
        # Scatter of observations models (sampled models)
        for i in range(nModels):
            f = bma.quadratic_models[i].get_y()
            ax.scatter(X[0,i], f, s=100.0, c="black", alpha=0.5)

    def plt_acquisition(ax):
        # Get acquisition values:
        acq = bmao.calculate_discounted_mean(x_grid, kappa=kappa)
        
        acq_line = ax.plot(x_plot, acq)
        plt.setp(acq_line, color="red", linewidth = 2.0, linestyle='--')

        # Plot next point
        Z_next_rolled = bma.predict(np.array([x_next]))
        #ax.scatter(x_next,Z_next_rolled[0,:]+(y_max-y_min)/80., s=100.0, c="red",marker='v')

    def plt_mean(ax):
        # Plot prediction, and plot function
        pred_line = ax.plot(x_plot, Z)
        # Set line properties
        plt.setp(pred_line, color="blue", linewidth=2.0, alpha=0.5)


    def plt_confidence(ax):
        # Plot confidence bars for the two types of errors
        ax.fill_between(x_plot,Z-S,Z+S, facecolor='blue', interpolate=True, alpha=0.3)
        ax.fill_between(x_plot,Z-U,Z+U, facecolor='red', interpolate=True, alpha=0.3)
        # Full std bars
        upper_min = np.max(np.vstack([Z+S,Z+U]), axis=0)

        lower_min = np.min(np.vstack([Z-S,Z-U]), axis=0)
        ax.fill_between(x_plot,upper_min,Z+std, facecolor='black', interpolate=True, alpha=0.15)
        ax.fill_between(x_plot,Z-std,lower_min, facecolor='black', interpolate=True, alpha=0.15)

    def plt_minature(ax):
        # Plot minature cuvature plots for each model
        dx_mini_plot = np.linspace(-dx, dx, mini_num_points)
        dx_mini_grid = np.array([dx_mini_plot])

        # For the plotting, we will use the current colorcycle
        ax.set_color_cycle(colorcycle)

        for i in range(nModels):
            model_predictions = bma.model_predictions(x_grid)
            line = ax.plot(x_plot, model_predictions[i,:])

            # Set line properties for miniplot
            plt.setp(line, linewidth=3.0, alpha=0.5, linestyle='-',
                     dash_capstyle='round')

    ### Plot 1
    fig, ax = plt.subplots(dpi=dpi)

    plt_func(ax)
    plt_data(ax, X)
    plt_minature(ax)

    # Plot range
    plt.ylim([y_min, y_max])
    plt.xlim([x_min, x_max])
    plt.savefig("figures/"+str(k_fig)+"_func_obs_.png", dpi=dpi)

    ### Plot 2
    fig, ax = plt.subplots(dpi=dpi)

    plt_func(ax)
    plt_data(ax, X)
    plt_mean(ax)
    plt_confidence(ax)

    # Plot range
    plt.ylim([y_min, y_max])
    plt.xlim([x_min, x_max])
    plt.savefig("figures/"+str(k_fig)+"_bma_conf_.png", dpi=dpi)

    ### Plot 3
    fig, ax = plt.subplots(dpi=dpi)

    plt_func(ax)
    plt_data(ax, X)
    plt_mean(ax)
    plt_acquisition(ax)

    # Plot range
    plt.ylim([y_min, y_max])
    plt.xlim([x_min, x_max])
    plt.savefig("figures/"+str(k_fig)+"_bma_acq_.png", dpi=dpi)

    ### Informative Plot
    n_subplot = 5
    fig2 = plt.figure(figsize=(8, 8), dpi=dpi)
    gs = gridspec.GridSpec(n_subplot, 1, height_ratios=[3, 1, 1, 1, 1])
    
    # Create axis array
    ax = []
    for i in range(n_subplot):
        ax.append(plt.subplot(gs[i]))

    # Show the model priors
    log_model_priors = bma.estimate_log_model_priors(x_grid)
    ax[1]._get_lines.set_color_cycle(colorcycle)
    for i in range(nModels):
        p1_line = ax[1].plot(x_plot, np.exp(log_model_priors[i,:]))
        plt.setp(p1_line, linewidth=3.0, alpha=1.0, linestyle='-',
                 dash_capstyle='round')
    # Plot range
    ax[1].set_ylim([0, 1])
    ax[1].set_xlim([x_min, x_max])
    ax[1].set_ylabel('prior', fontsize=16)

    # Plot marginal likelihood proportionalities
    ax[2]._get_lines.set_color_cycle(colorcycle)
    for i in range(nModels):
        p2_line = ax[2].plot(x_plot, np.log(marginal_likelihoods[i,:]))
        plt.setp(p2_line, linewidth=3.0, alpha=1.0, linestyle='-',
                 dash_capstyle='round')

    #ax[2].set_ylim([0.00000001, 1])
    ax[2].set_ylim(-25,0)
    #ax[2].set_yscale('log')
    ax[2].set_xlim([x_min, x_max])
    ax[2].set_ylabel('ll', fontsize=16)
        
    # Show the model weights
    model_weights = bma.estimate_model_weights(x_grid)
    ax[3]._get_lines.set_color_cycle(colorcycle)
    for i in range(nModels):
        p3_line = ax[3].plot(x_plot, model_weights[i,:])
        plt.setp(p3_line, linewidth=3.0, alpha=1.0, linestyle='-',
                 dash_capstyle='round')
    # Plot range
    ax[3].set_ylim([0, 1])
    ax[3].set_xlim([x_min, x_max])
    ax[3].set_ylabel('posterior', fontsize=16)

    # Show the kernel weights
    kernel_weights = bma.calc_kernel_weights(x_grid)
    # N_eff
    N_eff = np.sum(kernel_weights, 0)
    #N_eff_line = ax[4].plot(x_plot, N_eff)
    #plt.setp(N_eff_line, linewidth = 3.0, alpha=1.0, linestyle='-',
    #         color='black')

    # Shade in the makeup of N_eff
    cumulant = N_eff*0.0

    for i in range(nModels):
        color = colorcycle[i]
        ax[4].fill_between(x_plot,cumulant,cumulant+kernel_weights[i,:], facecolor=color, interpolate=True, alpha=0.35)
        cumulant += kernel_weights[i,:]
    # Plot range
    ax[4].set_ylim([0, nModels])
    ax[4].set_xlim([x_min, x_max])
    ax[4].set_ylabel('N_eff', fontsize=16)


    ### Plot alpha weighted by posterior cuvature plots for each model
    model_predictions = bma.model_predictions(x_grid)

    # For the plotting, we will use the current colorcycle
    ax[0]._get_lines.set_color_cycle(colorcycle)
    color_cycle = ax[0]._get_lines.color_cycle

    for i in range(nModels):
        # Max alpha for the plots
        max_alpha = 0.7

        # Cited from http://matplotlib.1069221.n5.nabble.com/Varying-alpha-in-ListedColormap-not-working-td39950.html

        x = x_plot
        y = model_predictions[i,:]
        scaling = model_weights[i,:]

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1],points[1:]], axis=1)

        # Values to scale by
        smin = 0
        smax = 1
        
        # Inline function to convert scaling value to alpha value in [0,1]
        alpha = lambda s0:(s0-smin)/(smax-smin)

        # Create a (rgba) color description for each line segment
        cmap = []

        # Get the next color in the colorcycle
        color = next(color_cycle)
        converter = cl.ColorConverter()
        # Get the rgb color tuple and convert it to a list
        color_rgba = list(converter.to_rgb(color))
        # Add placeholder alpha to get color_rgba
        color_rgba.append(0.0)

        for a in segments:
            # The x-value for the segment
            x0 = a.mean(0)[0]
            # With scaling value of
            s0 = np.interp(x0, x, scaling)
            # And alpha value of 
            a0 = alpha(s0)

            # Pop the previous alpha and add the current alpha
            color_rgba[3] = a0*max_alpha
            
            # Add the rgba entry to the colormap. Make sure to copy the list
            # Using list slicing.
            cmap.append(color_rgba[:])
        
        # Create the line collection objecft
        lc = LineCollection(segments)
        lc.set_color(cmap)

        # Set line properties. 
        # Dashed line
        lc.set_dashes('-')
        # line width 3.0
        lc.set_linewidth(3.0)
            
        ax[0].add_collection(lc)

        """
        # Normal plot
        #p0_line = ax[0].plot(x_plot, model_predictions[i,:])
        # Set line properties for miniplot
        #plt.setp(p0_line, linewidth=3.0, alpha=model_weights[i,:], linestyle='--',
        #         dash_capstyle='round')
        """

    # Now plot scatter of observations
    for i in range(nModels):
        f = bma.quadratic_models[i].get_y()
        ax[0].scatter(X[0,i],f,s=100.0, c="black", alpha=0.5)

    # Plot bma predictions
    pred_line = ax[0].plot(x_plot,Z)
    
    # Line properties
    plt.setp(pred_line, color="black", alpha=0.3, linewidth=3.0)

    # Set the plot range
    ax[0].set_xlim(x_min, x_max)
    ax[0].set_ylim(y_min, y_max)

    # set x axis
    ax[4].set_xlabel("x",fontsize=16)

    plt.savefig("figures/"+str(k_fig)+"_bma_model_breakdown.png", dpi=dpi)

    fig3, ax = plt.subplots(2, sharex=True)
    
    ax[0].plot(x_plot, N_eff)

    ### Plot of errors
    for i in range(nModels):
        p1_line = ax[1].plot(x_plot, errors[i,:]+10**-32)
        plt.setp(p1_line, linewidth=3.0, alpha=1.0, linestyle='-',
                 dash_capstyle='round')
        
    # Set log scale
    ax[1].set_yscale('log')
    plt.savefig("figures/"+str(k_fig)+"_bma_error_breakdown.png", dpi=dpi)
    
    
    ### Likelihood plots
    fig4, ax = plt.subplots(2, sharex=True)
    kernel_grid = np.logspace(-4.0, 1, num=50)

    # Get the likelihoods 
    unreg_loglikelihood = np.array([bma.loglikelihood(kernel_range, regularization=False, skew=False) for kernel_range in kernel_grid])
    reg_loglikelihood = np.array([bma.loglikelihood(kernel_range) for kernel_range in kernel_grid])

    # Plot the two terms
    ll0 = ax[0].plot(kernel_grid, unreg_loglikelihood)
    ax[0].set_xscale('log')
    ll1 = ax[1].plot(kernel_grid, reg_loglikelihood)
    ax[1].set_xscale('log')

    plt.setp(ll0, color="red", linewidth=3.0, alpha=0.5, linestyle='-',
             dash_capstyle='round')
    plt.setp(ll1, color="red", linewidth=3.0, alpha=0.5, linestyle='-',
             dash_capstyle='round')

    ax[1].set_xlabel("kernel range",fontsize=16)

    plt.savefig("figures/"+str(k_fig)+"_bma_loglikelihood.png", dpi=dpi)

    """
    ### Likelihood plots
    fig4, ax = plt.subplots(2)
    kernel_grid = np.logspace(-4.0, 1.0, num=50)

    unreg_loglikelihood = np.array([bma.loglikelihood(kernel_range, regularization=False) for kernel_range in kernel_grid])
    unreg_loglikelihood = np.array([bma.loglikelihood(kernel_range, regularization=False) for kernel_range in kernel_grid])

    # Plot the two terms
    ll1 = ax.plot(kernel_grid, unreg_loglikelihood)
    ax.set_xscale('log')


    plt.savefig("figures/"+str(k_fig)+"_bma_loglikelihood.png")
    """
# Create the function

def fun(x):
    #return (np.square(np.square(x))+10.0*np.sin(x)+10.0*np.sin(5.0*x))[0]
    return (np.square(np.square(x))+10.0*np.sin(x))[0]
    """
    if x<1.0:
        return 1.0*np.square(x-0.5)[0]
    else:
        return 1.0*0.25+10.0*(np.square(x))[0]-10.0
        """
h = 0.000001

def fun_all(x,r1=0.0, r2=0.0, r3=0.0, bool_zero=False):
    """
    r1 : additive error on f
    r2 : additive error on grad
    r3 : additive error on Hess
    """
    f = fun(x) + r1
    der = (fun(x+h)-fun(x-h))/(2*h)
    derder = (fun(x+h)+fun(x-h)-2*fun(x))/(h**2)
    grad = np.array([der]) + r2
    Hess = np.array([[derder]]) + r3
    if bool_zero:
        Hess = Hess*0.0
        grad = grad*0.0
    return f, grad, Hess

# Create optimizer object
bmao = bo.optimizer.QuadraticBMAOptimizer(ndim = 1, precision_beta=100.0)

# Create the points for function evaluation
X_next = np.array([[0.0,0.5,2.5,1.5,0.4]])
X = None

# Error  values
errors = [(30.0, np.array([10.0]), np.array([[50.0]])),
          (-10.0, np.array([-30.0]), np.array([[-20.0]])),
          (20.0, np.array([-10.0]), np.array([[100.0]])),
          (5.0, np.array([20.0]), np.array([[10.0]])),
          (3.0, np.array([-4.0]), np.array([[-30.0]]))]

for k in xrange(X_next.shape[1]):
    x_next = X_next[:,k]
    x = x_next
    if k == 0:
        X = np.array([x_next])
    else:
        X = np.hstack([X, np.array([x_next])])

    # Get y, grad, hess, and update corresponding lists
    f, grad, Hess = fun_all(x, 0.0, 0.0, 0.0)

    # Add errors
    f += errors[k][0]
    grad += errors[k][1]
    Hess += errors[k][2]

    bmao.add_observation(x, f, grad, Hess)
    
    kappa = bmao.kappa_detail
    bmao.set_kernel_range(2.0)

    # Plot result
    plot(bmao, X, k, x_next, kappa)

