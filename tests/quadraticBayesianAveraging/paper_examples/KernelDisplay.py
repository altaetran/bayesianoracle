import numpy as np
import bayesianoracle as bo
import bayesianoracle.plot as boplotter

# Import function information
from function_data import *
import os
os.system("function_data.py")

def plot_kernels(bmao, X, x0, kernel_ranges, colors):
    """ Auxillary plotting function
    
    Parameters
    ----------
    bmao : Bayesian model averaging optimization process
    X  : The values that have been previously traversed
    x0 : location at which we want to evaluate the kernel
    kernel_ranges : (list of scalars) the desired kernel_widths to be plotted
    """

    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib import colors as cl
    from matplotlib import gridspec

    boplt = boplotter.Plotter1D(x_range=x_range, y_range=y_range, num_points=num_points*10)
    boplt.set_bma(bmao.bma)

    ### Plot the data and the models
    fig, ax = plt.subplots()

    legend_elements = []
    legend_texts = []
    
    for i in xrange(len(kernel_ranges)):
        # Plot the kernels
        kernel_line = boplt.plot_kernel(ax, x0, kernel_ranges[i], color=colors[i], ylabel='data relevance')

        # Plot data 
        data_scat = boplt.plot_kernel_at_data(ax, x0, kernel_ranges[i], color=colors[i], bool_color_cycled=True, xlabel=r'$x^\prime$')
        
        """
        # Add the information for legends
        legend_elements.append(kernel_line)
        legend_texts.append(r"$K_\theta(x)$ for $\theta = "+str(kernel_ranges[i])+"$")

        # Get the data element in the legend
        legend_elements.append(data_scat)
        legend_texts.append(r"$K_\theta(x_i)$ for $\theta = "+str(kernel_ranges[i])+"$")
        """
        legend_elements.append((kernel_line, data_scat))
        legend_texts.append(r"$\theta = " + str(kernel_ranges[i])+"$")

    #boplt.plot_data_locations(ax, color='black', alpha=0.3, linestyle='--', zorder=1)

    # Create the x0 line
    boplt.draw_vertical_line(ax, x0, r'$x='+str(x0[0])+'$', color='#FF9900')

    # Reverse the legend texts and elements
    legend_elements = reversed(legend_elements)
    legend_texts = reversed(legend_texts)

    # Create the legend
    legend = plt.legend(legend_elements, 
                        legend_texts,
                        loc='center right', bbox_to_anchor=(1.05, 0.5), ncol=1, fancybox=True, shadow=False, scatterpoints=1)

    # Change the sizes of the scatter dots in legend
    """
    for i in xrange(len(kernel_ranges)):
        legend.legendHandles[2*i+1]._sizes = [30]
    """

    plt.setp(legend.get_texts(), fontsize=12)    

    plt.savefig("KernelDisplay_figures/Kernels.png", dpi=dpi)
    plt.close(fig)


bmao = bo.optimizer.QuadraticBMAOptimizer(ndim = 1, 
                                          init_kernel_range=0.25, 
                                          n_int=50,
                                          precision_beta = 1000.0,
                                          constraints = [constr1, constr2],
                                          bounding_box = bounding_box,
                                          bool_compact = True,
                                          kernel_type='Gaussian')

# Center of the kernel
x0 = np.array([-0.5])

# Initialize x_next
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
           
    y_hist = np.append(y_hist, f)

    # Add the observations to the bmao
    bmao.add_observation(x, f, grad, Hess)

kernel_ranges = [2.0, 0.25, 0.1]
colors = ['#FFBBBB', '#FF7777', '#FF0000']
#colors = ['#00FF99', '#66FFCC', '#99FFCC']

#colors = ['#FFD494','#FFB870','#FF9900']
colors = ['#DDDDDD','#AAAAAA','#666666']
#colors = ['#707070','#333333','#000000']
# Plot the Kernels
plot_kernels(bmao, X, x0, kernel_ranges, colors)
