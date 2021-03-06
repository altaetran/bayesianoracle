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
    from matplotlib.collections import LineCollection
    from matplotlib import colors as cl
    from matplotlib import gridspec

    # Get number of models
    nModels = len(bma.quadratic_models)

    # Plot ranges
    x_min = -3.0
    x_max = 3.0
    y_min = -100
    y_max = 100

    # Plot for 1d
    num_points = 100
    # mini plot dx param
    dx = 0.5
    # miniplot num points
    mini_num_points = 50

    # Create x array to be plotted
    x_plot = np.linspace(x_min, x_max, num_points)
    x_grid = np.array([x_plot])

    # Get mean values at each of the grid points
    Z = np.zeros(x_plot.shape)
    S = np.zeros(x_plot.shape)
    
    F = np.zeros(x_plot.shape)
    for i in range(len(x_plot)):
        F[i] = fun(np.array([x_plot[i]]))

    Z_rolled = bma.predict_with_unc(x_grid)
    Z = Z_rolled[0,:]
    # bma disagreement
    S = Z_rolled[1,:]
    # bma uncertainty
    U = Z_rolled[2,:]

    ### Plot 1
    fig, ax = plt.subplots()

    # Plot prediction, and plot function
    pred_line = ax.plot(x_plot,Z)
    func_line = ax.plot(x_plot,F)
    
    # Set line properties
    plt.setp(pred_line, color="blue", linewidth=2.0)
    plt.setp(func_line, color="black", linewidth=2.0)

    # Scatter of observations
    for i in range(nModels):
        f = bma.quadratic_models[i][0]
        ax.scatter(X[0,i],f,s=100.0, c="black", alpha=0.5)

    # Plot disagreement bars
    ax.fill_between(x_plot,Z-S,Z+S, facecolor='blue', interpolate=True, alpha=0.3)
    ax.fill_between(x_plot,Z-U,Z+U, facecolor='red', interpolate=True, alpha=0.3)

    # Plot minature cuvature plots for each model
    dx_mini_plot = np.linspace(-dx, dx, mini_num_points)
    dx_mini_grid = np.array([dx_mini_plot])
    for i in range(nModels):
        model_predictions = bma.model_predictions(dx_mini_grid+X[0,i])
        line = ax.plot(dx_mini_plot+X[0,i], model_predictions[i,:])
        # Set line properties for miniplot
        plt.setp(line, color="green", linewidth=3.0, alpha=0.5, linestyle='--',
                 dash_capstyle='round')
    
    # Plot range
    plt.ylim([y_min, y_max])
    plt.xlim([x_min, x_max])
    plt.savefig("figures/"+str(k_fig)+"_bma_vs_func_.png")

    ### Plot 2
    n_subplot = 4
    fig2 = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(4, 1, height_ratios=[3, 1, 1, 1])
    
    # Create axis array
    ax = []
    for i in range(n_subplot):
        ax.append(plt.subplot(gs[i]))
    #fig, ax = plt.subplots(4, sharex=True)

    # Show the model priors
    model_priors = bma.estimate_model_priors(x_grid)
    for i in range(nModels):
        p1_line = ax[1].plot(x_plot, model_priors[i,:])
        plt.setp(p1_line, linewidth=3.0, alpha=1.0, linestyle='-',
                 dash_capstyle='round')
    # Plot range
    ax[1].set_ylim([0, 1])
    ax[1].set_xlim([x_min, x_max])
        
    # Show the model weights
    model_weights = bma.estimate_model_weights(x_grid)
    for i in range(nModels):
        p2_line = ax[2].plot(x_plot, model_weights[i,:])
        plt.setp(p2_line, linewidth=3.0, alpha=1.0, linestyle='-',
                 dash_capstyle='round')
    # Plot range
    ax[2].set_ylim([0, 1])
    ax[2].set_xlim([x_min, x_max])

    # Show the kernel values 
    kernel_weights = bma.calc_relevance_weights(x_grid)
    for i in range(nModels):
        p3_line = ax[3].plot(x_plot, kernel_weights[i,:])
        plt.setp(p3_line, linewidth=3.0, alpha=1.0, linestyle='-',
                 dash_capstyle='round')

    ### Plot alpha weighted by posterior cuvature plots for each model
    model_predictions = bma.model_predictions(x_grid)

    # For the plotting, we will use the current colorcycle
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
        lc.set_dashes('--')
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
        f = bma.quadratic_models[i][0]
        ax[0].scatter(X[0,i],f,s=100.0, c="black", alpha=0.5)

    # Plot bma predictions
    pred_line = ax[0].plot(x_plot,Z)
    
    # Line properties
    plt.setp(pred_line, color="black", alpha=0.3, linewidth=3.0)

    # Set the plot range
    ax[0].set_xlim(x_min, x_max)
    ax[0].set_ylim(y_min, y_max)

    plt.savefig("figures/"+str(k_fig)+"_bma_model_breakdown.png")

    fig3, ax = plt.subplots(2, sharex=True)
    model_weights, errors, N_eff = bma.estimate_model_weights(x_grid, return_errors=True)
    
    ax[0].plot(x_plot, N_eff)

    ### Plot of errors
    for i in range(nModels):
        p1_line = ax[1].plot(x_plot, errors[i,:])
        plt.setp(p1_line, linewidth=3.0, alpha=1.0, linestyle='-',
                 dash_capstyle='round')
        
    # Set log scale
    ax[1].set_yscale('log')
    plt.savefig("figures/"+str(k_fig)+"_bma_error_breakdown.png")
    


# Create 1 dimensional bma process object
bma = bo.process_objects.QuadraticBMAProcess(ndim = 1)

# Create the function

def fun(x):
    return (np.square(np.square(x))+10.0*np.sin(x))[0]

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
    b = grad - Hess.dot(x)
    if bool_zero:
        Hess = Hess*0.0
        grad = grad*0.0
    [A, b, d] = bo.process_objects.der_to_quad(x, f, grad, Hess)
    return f, A, b, d, x, A

"""
x = 2.0
r1 = 40.0*(np.random.rand(1,1)[0]-0.5)
r2 = 20.0*(np.random.rand(1,1)[0]-0.5)
r3 = 60.0*(np.random.rand(1,1)[0]-0.5)
f, A, b, d, a, A = fun_all(np.array([x]), r1, r2, r3)
bma.add_model(f, A, b, d, a, A)

# X is a n x p stack of points
a = [[x]]
X = a

for i in range(2):
    x = 2.0
    r1 = 40.0*(np.random.rand(1,1)[0]-0.5)
    r2 = 20.0*(np.random.rand(1,1)[0]-0.5)
    r3 = 60.0*(np.random.rand(1,1)[0]-0.5)
    f, A, b, d, a, A = fun_all(np.array([x]), r1, r2, r3)
    bma.add_model(f, A, b, d, a, A)

    # X is a n x p stack of points
    a = [[x]]
    X = np.hstack([X, a])
"""

X = np.array([[]])

for i in range(2):
    x = 0.5+5.0*(np.random.rand(1,1)[0,0]-0.5)

    r1 = (50.0+50*np.sin(x))*(np.random.rand(1,1)[0]-0.5)
    r2 = 50.0*(np.random.rand(1,1)[0]-0.5)
    r3 = 100.0*(np.random.rand(1,1)[0]-0.5)
#    r1 = 0
#    r2 = 0
#    r3 = 0

    f, A, b, d, a, A = fun_all(np.array([x]), r1, r2, r3)
    bma.add_model(f, A, b, d, a, A)

    #f, A, b, d, a, A = fun_all(np.array([x]), r1, r2, r3, bool_zero=True)
    #bma.add_model(f, A, b, d, a, A)

    a = [[x]]
    if X.shape[0]==0:
        X = a
    else:
        X = np.hstack([X, a])
        
    #X = np.hstack([X, a])


print('lol')
print(bma.pdf(np.array([[0.0, 0.0, 0.0]]),np.array([0, 10, 20])))

print('uh')
plot(bma, X, 1)

bma.cross_validate()

plot(bma, X, 2)

#bma.zero_errors=True

#plot(bma, X, 2)

print(bma.predict(np.array([[0.0]]), bool_weights=False))

X_test = np.array(np.array([[0.0],[0.01], [-0.1]]).T)

print(bma.predict(X_test, bool_weights=False))
