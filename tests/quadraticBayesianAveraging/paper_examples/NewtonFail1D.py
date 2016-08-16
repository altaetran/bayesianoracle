import numpy as np
import bayesianoracle as bo
import bayesianoracle.plot as boplotter

# Import function information
from function_data import *
execfile("function_data.py")

def plot(quad, X, y_hist, k_fig, x_next, y_next):
    """ Auxillary plotting function
    
    Parameters
    ----------
    quad : quadratic model
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

    ### Plot the data and the models
    fig, ax = plt.subplots()

    func_line = boplt.plot_fun(ax, fun)
    mean_line = boplt.plot_quad(ax, quad)
    scat = boplt.plot_data(ax, X, y_hist, xlabel=r'$x$')
    next = boplt.plot_next(ax, x_next, y_next)

    # Create legend
    legend = plt.legend([mean_line, func_line, scat, next], 
                        ['Noisy Quadratic Approximation', 'True Mean Function', 'Data', 'Next Test Location'],
                        loc='upper center', bbox_to_anchor=(0.5, 1.075), ncol=2, fancybox=True, shadow=False, scatterpoints=1)
    legend.legendHandles[2]._sizes = [30]
    legend.legendHandles[3]._sizes = [30]
    plt.setp(legend.get_texts(), fontsize=12)

    plt.savefig("NewtonFail1D_figures/"+str(k_fig)+"_predictive.png", dpi=dpi)
    plt.close(fig)

# Initialize x_nextp
x_next = np.array([2.5])
X = None
y_hist = np.array([])

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

    # Get y, grad, hess, and update corresponding lists
    f, grad, Hess = fun_all(x, r1, r2, r3)

    y_hist = np.append(y_hist, f)
    
    A, b, d = bo.process_objects.der_to_quad(x, f, grad, Hess)
    
    quad = bo.process_objects.QuadraticModel(f, A, b, d ,x)

    # Get the next locations
    x_next = quad.get_min_location()

    # Take it to be the within the bounds
    if x_next > ub:
        x_next = np.array([ub])
    elif x_next < lb:
        x_next = np.array([lb])
        
    # Get the predicted y_val at the next position
    y_next = quad.predict(x_next)
    
    # Plot result
    plot(quad, X, y_hist, k, x_next, y_next)

