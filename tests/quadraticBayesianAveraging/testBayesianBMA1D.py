import numpy as np
import bayesianoracle as bo
import bayesianoracle.plot as boplotter

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

    # Plot ranges
    x_range = [-5.0, 5.0]
    y_range = [-200.0, 200.0]

    # Number of x points to plot
    num_points = 200

    dpi=600

    boplt = boplotter.Plotter1D(x_range=x_range, y_range=y_range, num_points=num_points)
    boplt.set_bma(bma)
    boplt.get_bayesian_bma_evals()

    ### Plot the data and the models
    fig, ax = plt.subplots()

    boplt.plot_fun(ax, fun)
    boplt.plot_data(ax)
    boplt.plot_models(ax)
    plt.savefig("figures/"+str(k_fig)+"_a_obs_.png", dpi=dpi)

    ### Plot the bayesian quantities 
    fig, ax = plt.subplots()

    boplt.plot_fun(ax, fun)
    boplt.plot_data(ax)
    boplt.plot_means(ax)
    boplt.plot_confidence(ax, bool_bayesian=True)

    plt.savefig("figures/"+str(k_fig)+"_b_predictive_.png", dpi=dpi)

    ### Plot the kernel range likelihoods
    n_subplot = 2
    fig = plt.figure(figsize=(8, 8), dpi=dpi)
    gs = gridspec.GridSpec(n_subplot, 1, height_ratios=[3, 1])
    
    # Create axis array
    ax = []
    for i in range(n_subplot):
        ax.append(plt.subplot(gs[i]))


    boplt.plot_bayesian_likelihoods(ax[0])
    boplt.plot_bayesian_avg_kernel_ranges(ax[1])
    plt.savefig("figures/"+str(k_fig)+"_c_likelihoods_.png", dpi=dpi)


# Create enriched process
ebma = bo.process_objects.EnrichedQuadraticBMAProcess(ndim = 1)

# Create the function

def fun(x):
    return (np.square(np.square(x))+10.0*np.sin(x)+10.0*np.sin(5.0*x))[0]
    #return (np.square(np.square(x))+10.0*np.sin(x))[0]
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

bmao = bo.optimizer.QuadraticBMAOptimizer(ndim = 1, init_kernel_var = 10.0, init_kernel_range=1.0, precision_beta = 100.0,
                                          kernel_type='triweight')

# Initialize x_next
x_next = np.array([3.0])

X_next = np.array([[2.5, 2.0, 1.5, 1.0, 0.8, 0.5]])
X = None

for k in range(X_next.shape[1]):
    # Get next
    x_next = X_next[:,k]
    x = x_next
    if k == 0:
        X = np.array([x_next])
    else:
        X = np.hstack([X, np.array([x_next])])
    
    # Make up errors
    r1 = (20.0+20*np.sin(x)[0])*(np.random.rand(1,1)[0,0]-0.5)
    r2 = (40.0+40*np.sin(x)[0])*(np.random.rand(1,1)[0,0]-0.5)
    r3 = (60.0+60*np.sin(x)[0])*(np.random.rand(1,1)[0,0]-0.5)

    r1 = 0.0
    r2 = 0.0
    r3 = 0.0

    # Get y, grad, hess, and update corresponding lists
    f, grad, Hess = fun_all(x, r1, r2, r3)
    
    bmao.add_observation(x, f, grad, Hess)
    
    kappa = bmao.kappa_detail

    bmao.set_kernel_range(0.1)

    # Plot result
    plot(bmao, X, k, x_next, kappa)

    #bmao.bma.predict_bayesian(X)
