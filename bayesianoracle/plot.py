import numpy as np
import scipy
import matplotlib.pyplot as plt

from matplotlib.collections import LineCollection
from matplotlib import colors as cl
from matplotlib import gridspec, markers

class Plotter1D(object):
    def __init__(self,
                 x_range=[-2.0, 2.0],
                 y_range=[-10.0, 10.0],
                 num_points=200,
                 global_linewidth=4.0,
                 colorcycle=['r', 'b', 'g', 'm', 'y', 'c']):
        self.x_min = x_range[0]
        self.x_max = x_range[1]
        self.y_min = y_range[0]
        self.y_max = y_range[1]

        self.num_points = num_points

        self.x_plot = np.linspace(self.x_min, self.x_max, self.num_points)
        self.x_grid = np.array([self.x_plot])

        self.bma = None
        self.n_models = 0

        # Plotting options
        self.global_linewidth = global_linewidth
        self.colorcycle = colorcycle
        
    def set_bma(self, bma):
        self.bma = bma
        self.model_predictions = bma.model_predictions(self.x_grid)
        self.n_models = len(bma.quadratic_models)

    def plot_fun(self, ax, fun, alpha=0.7, linestyle='--'):
        """ Plots the function, fun, along the x range 

        Parameters
        ----------
        ax : (matplotlib axis) to plot along
        fun : (function handle) - funtion to plot
        alpha : (float in range [0,1]) transparency of line
        """
        # Set current plt axis to ax
        plt.sca(ax)

        # Get function values along x_grid
        F = np.zeros(self.x_plot.shape)
        for i in range(len(self.x_plot)):
            F[i] = fun(np.array([self.x_plot[i]]))

        # Plot function
        func_line, = ax.plot(self.x_plot, F)
        # Set line properties 
        plt.setp(func_line, color='black', linewidth=self.global_linewidth, alpha=alpha, linestyle=linestyle,
                 dash_capstyle='round')

        # Set range
        plt.ylim([self.y_min, self.y_max])
        plt.xlim([self.x_min, self.x_max])

        return func_line

    def plot_data(self, ax, alpha=0.5):
        # Set current plt axis to ax
        plt.sca(ax)

        # Scatter of observations models (sampled models)
        for i in range(self.n_models):
            a = self.bma.quadratic_models[i].get_a()
            f = self.bma.quadratic_models[i].get_y()
            ax.scatter(a, f, s=100.0, c="black", alpha=alpha, zorder=3)

        # Set range
        plt.ylim([self.y_min, self.y_max])
        plt.xlim([self.x_min, self.x_max])

    def get_bayesian_bma_evals(self):
        Z_rolled, kernel_ranges, likelihoods, avg_kernel_ranges = self.bma.predict_bayesian(self.x_grid, return_likelihoods=True)

        # Save the predictions
        self.bayesian_means = Z_rolled[0,:]
        self.bayesian_unexp_std = Z_rolled[1,:]
        self.bayesian_exp_std = Z_rolled[2,:]
        self.bayesian_N_eff = Z_rolled[3,:]
        self.bayesian_kernel_ranges = kernel_ranges
        self.bayesian_likelihoods = likelihoods
        self.bayesian_avg_kernel_ranges = avg_kernel_ranges

        # Get the full standard deviation
        self.bayesian_std = np.sqrt(np.square(self.bayesian_unexp_std) + 
                                   np.square(self.bayesian_exp_std))

    def get_pointwise_bma_evals(self, kernel_range=-1.0):
        if kernel_range == -1.0:
            # Use built in kernel range
            Z_rolled = self.bma.predict_with_unc(self.x_grid)
            model_weights, errors, N_eff, marginal_likelihoods = self.bma.estimate_model_weights(self.x_grid, return_likelihoods=True)
            kernel_weights = self.bma.calc_kernel_weights(self.x_grid)
        else:
            # Otherwiuse use given kernel range
            Z_rolled = self.bma.predict_with_unc(self.x_grid, kernel_range=kernel_range)
            model_weights, errors, N_eff, marginal_likelihoods = self.bma.estimate_model_weights(self.x_grid, return_likelihoods=True, kernel_range=kernel_range)
            kernel_weights = self.bma.calc_kernel_weights(self.x_grid, kernel_range=kernel_range)

        # Unroll and save
        self.pointwise_means = Z_rolled[:,0]
        self.pointwise_unexp_std = Z_rolled[1,:]
        self.pointwise_exp_std = Z_rolled[2,:]
        self.pointwise_N_eff = Z_rolled[3,:]

        # Get the full standard deviation
        self.pointwise_std = np.sqrt(np.square(self.pointwise_unexp_std) + np.square(self.pointwise_exp_std))
        
        # Save other values
        self.pointwise_model_weights = model_weights
        self.pointwise_errors = errors
        self.pointwise_N_eff = N_eff
        self.pointwise_marginal_likelihoods = marginal_likelihoods
        self.pointwise_kernel_weights = kernel_weights

        # Get log_model_priors
        self.pointwise_log_model_priors = self.bma.estimate_log_model_priors(self.x_grid)

    def plot_means(self, ax, bool_bayesian=True, alpha=0.7, xlabel='x'):
        if bool_bayesian:
            means = self.bayesian_means
        else:
            means = self.pointwise_means

        # Set current plt axis to ax
        plt.sca(ax)
            
        func_line, = ax.plot(self.x_plot, means)
        # Set line properties 
        plt.setp(func_line, color="black", linewidth=self.global_linewidth, alpha=alpha)

        # Set range
        plt.ylim([self.y_min, self.y_max])
        plt.xlim([self.x_min, self.x_max])

        ax.set_xlabel(xlabel, fontsize=12)

        return func_line

    def plot_discounted_means(self, ax, kappa, bool_bayesian=True, linestyle='--', alpha=0.9, color='magenta', xlabel='x'):
        if bool_bayesian:
            means = self.bayesian_means
            std = self.bayesian_unexp_std
        else:
            means = self.pointwise_means
            std = self.pointwise_unexp_std

        # Set current plt axis to ax
        plt.sca(ax)
            
        func_line, = ax.plot(self.x_plot, means - kappa*std)
        # Set line properties 
        plt.setp(func_line, color=color, linewidth=self.global_linewidth, linestyle=linestyle, alpha=alpha,
                 dash_capstyle='round')

        # Set range
        plt.ylim([self.y_min, self.y_max])
        plt.xlim([self.x_min, self.x_max])

        ax.set_xlabel(xlabel, fontsize=12)

        return func_line

    def plot_confidence(self, ax, bool_bayesian=True, xlabel='x', exp_color='blue', unexp_color='red', std_color='black'):
        if bool_bayesian:
            means = self.bayesian_means
            unexp_std = self.bayesian_unexp_std
            exp_std = self.bayesian_exp_std
            std = self.bayesian_std
        else:
            means = self.pointwise_means
            unexp_std = self.pointwise_unexp_std
            exp_std = self.pointwise_exp_std
            std = self.pointwise_std

        # Set current plt axis to ax
        plt.sca(ax)

        # Plot confidence bars for the two types of errors
        exp_line = ax.fill_between(self.x_plot,means-exp_std, means+exp_std, facecolor=exp_color, interpolate=True, alpha=0.3)
        unexp_line = ax.fill_between(self.x_plot,means-unexp_std, means+unexp_std, facecolor=unexp_color, interpolate=True, alpha=0.3)

        # Full std bars
        upper_min = np.max(np.vstack([means+exp_std, means+unexp_std]), axis=0)
        lower_min = np.min(np.vstack([means-exp_std, means-unexp_std]), axis=0)
        std_line1 = ax.fill_between(self.x_plot, upper_min, means+std, facecolor=std_color, interpolate=True, alpha=0.15)
        std_line2 =ax.fill_between(self.x_plot, means-std, lower_min, facecolor=std_color, interpolate=True, alpha=0.15)

        # Set range
        plt.ylim([self.y_min, self.y_max])
        plt.xlim([self.x_min, self.x_max])

        ax.set_xlabel(xlabel, fontsize=12)

        # Create items for use as legend
        exp_fill = plt.Rectangle((0, 0), 1, 1, fc=exp_color)
        unexp_fill = plt.Rectangle((0, 0), 1, 1, fc=unexp_color)
        std_fill = plt.Rectangle((0, 0), 1, 1, fc=std_color)

        return exp_fill, unexp_fill, std_fill
        
    def plot_bayesian_likelihoods(self, ax, xlabel='x', ylabel='kernel range', nLevels=20, kernel_ranges_range=[1.0e-2,1.0e2]):
        # Set current plt axis to ax
        plt.sca(ax)

        padded_kernel_ranges = self.bayesian_kernel_ranges
        padded_likelihoods = self.bayesian_likelihoods
        # Pad the data to fill the full range
        if self.bayesian_kernel_ranges.min() > kernel_ranges_range[0]:
            # Left pad
            padded_kernel_ranges = np.insert(padded_kernel_ranges, 0, kernel_ranges_range[0])
            padded_likelihoods = np.insert(padded_likelihoods, 0, 0.0, axis=0)
        if self.bayesian_kernel_ranges.max() < kernel_ranges_range[1]:
            # Right pad
            padded_kernel_ranges = np.append(padded_kernel_ranges, kernel_ranges_range[1])
            padded_likelihoods = np.insert(padded_likelihoods, -1, 0.0, axis=0)            

        # Plot the countour
        heatmap = ax.contourf(self.x_plot, padded_kernel_ranges, padded_likelihoods, nLevels, cmap=plt.cm.bone)
        ax.set_yscale('log')

        # Set range
        plt.xlim([self.x_min, self.x_max])
        plt.ylim(kernel_ranges_range)

        # Set the labels
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # Get the horizontal colorbar
        cbar = plt.colorbar(heatmap, orientation='horizontal', pad=0.1)
        cbar.ax.set_xlabel('posterior probability')

        #plt.ylim([self.bayesian_kernel_ranges.min, self.bayesian_kernel_ranges.max])
        
    def plot_bayesian_avg_kernel_ranges(self, ax, xlabel='x', ylabel='avg kernel range', alpha=1.0, linestyle='-', kernel_ranges_range=[0.0,2.0]):
        # Set current plt axis to ax
        plt.sca(ax)

        # Set colorcylce
        ax._get_lines.set_color_cycle(self.colorcycle)

        # Plot model priors
        p_line, = ax.plot(self.x_plot, self.bayesian_avg_kernel_ranges)
        plt.setp(p_line, color='black', linewidth=self.global_linewidth, alpha=alpha, linestyle=linestyle,
                 dash_capstyle='round')

        # Plot range
        plt.xlim([self.x_min, self.x_max])
        plt.ylim(kernel_ranges_range)

        # Set labels
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)

        return p_line

    def plot_N_eff(self, ax, bool_bayesian=True, alpha=0.7, xlabel='x', ylabel='N_eff'):
        if bool_bayesian:
            N_eff = self.bayesian_N_eff
        else:
            N_eff = self.pointwise_N_eff

        # Set current plt axis to ax
        plt.sca(ax)
            
        N_eff_line, = ax.plot(self.x_plot, N_eff)
        # Set line properties 
        plt.setp(N_eff_line, color="black", linewidth=self.global_linewidth, alpha=alpha)

        # Set range
        plt.ylim([0.0, np.max(N_eff)])
        plt.xlim([self.x_min, self.x_max])

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)

        return N_eff_line

    def plot_models(self, ax, alpha=0.7, linestyle='-'):
        # Set current plt axis to ax
        plt.sca(ax)

        # For the plotting, we will use the current colorcycle
        ax.set_color_cycle(self.colorcycle)

        lines = []

        for i in range(self.n_models):
            model_predictions = self.bma.model_predictions(self.x_grid)
            line, = ax.plot(self.x_plot, model_predictions[i,:])

            # Set line properties for the quadratic line
            plt.setp(line, linewidth=self.global_linewidth, alpha=alpha, linestyle=linestyle,
                     dash_capstyle='round')

            lines.append(line)

        # Set range
        plt.xlim([self.x_min, self.x_max])
        plt.ylim([self.y_min, self.y_max])

        return lines

    def plot_model_priors(self, ax, bool_bayesian=False, alpha=1.0, linestyle='-', ylabel='prior'):
        """ Not possible for Bayesian yet
        Plot model prior weights
        """
        # Get the model priors
        if bool_bayesian:
            log_model_priors = self.bayesian_log_model_priors
        else:
            log_model_priors = self.pointwise_log_model_priors

        # Set current plt axis to ax
        plt.sca(ax)

        # Set colorcylce
        ax._get_lines.set_color_cycle(self.colorcycle)

        prior_lines = []

        # Plot model priors
        for i in xrange(self.n_models):
            p_line, = ax.plot(self.x_plot, np.exp(log_model_priors[i,:]))
            plt.setp(p_line, linewidth=self.global_linewidth, alpha=alpha, linestyle=linestyle,
                     dash_capstyle='round')

            prior_lines.append(p_line)

        # Plot range
        plt.xlim([self.x_min, self.x_max])
        plt.set_ylim([0, 1])
        ax.set_ylabel(ylabel, fontsize=12)

        return prior_lines

    def plot_model_posteriors(self, ax, bool_bayesian=False, alpha=1.0, linestyle='-', ylabel='posterior'):
        """ Not possible for Bayesian yet
        Plot model posterior weights
        """
        # Get the model priors
        if bool_bayesian:
            model_posteriors = self.bayesian_model_weights
        else:
            model_posteriors = self.pointwise_model_weights

        # Set current plt axis to ax
        plt.sca(ax)

        # Set colorcylce
        ax._get_lines.set_color_cycle(self.colorcycle)

        posterior_lines = []

        # Plot model priors
        for i in xrange(self.n_models):
            p_line, = ax.plot(self.x_plot, model_posteriors[i,:])
            plt.setp(p_line, linewidth=self.global_linewidth, alpha=alpha, linestyle=linestyle,
                     dash_capstyle='round')
            
            posterior_lines.append(p_line)

        # Plot range
        plt.xlim([self.x_min, self.x_max])
        plt.ylim([0, 1])
        ax.set_ylabel(ylabel, fontsize=12)

        return posterior_lines

    def plot_marginal_likelihoods(self, ax, bool_bayesian=False, alpha=1.0, linestyle='-', ylabel='ML'):
        # Get the model priors
        if bool_bayesian:
            marginal_likelihoods = self.bayesian_marginal_likelihoods
        else:
            marginal_likelihoods = self.pointwise_marginal_likelihoods
        
        # Set current plt axis to ax
        plt.sca(ax)

        # Set colorcycle
        ax._get_lines.set_color_cycle(self.colorcycle)

        marginal_likelihood_lines = []

        for i in range(self.n_models):
            p_line, = plt.plot(self.x_plot, np.log(marginal_likelihoods[i,:]))
            plt.setp(p_line, linewidth=global_linewidth, alpha=alpha, linestyle=linestyle,
                     dash_capstyle='round')

            marginal_likelihood_lines.append(p_line)


        #ax[2].set_yscale('log')
        plt.xlim([self.x_min, self.x_max])
        ax.set_ylabel(ylabel, fontsize=12)

        return marginal_likelihood_lines

    def plot_kernel_weights(self, ax, bool_bayesian=False, alpha=0.35, ylabel='N_eff'):
        # Get the model priors
        if bool_bayesian:
            kernel_weights = self.bayesian_kernel_weights
        else:
            kernel_weights = self.pointwise_kernel_weights

        # Set current plt axis to ax
        plt.sca(ax)

        # N_eff built from the kernel weights
        N_eff = np.sum(kernel_weights, 0)
        cumulant = N_eff*0.0

        for i in range(self.n_models):
            color = self.colorcycle[i]
            ax.fill_between(self.x_plot, cumulant, cumulant+kernel_weights[i,:], facecolor=color, interpolate=True, alpha=alpha)
            # Add to the cumulant
            cumulant += kernel_weights[i,:]

        # Plot range
        plt.ylim([0.0, self.n_models])
        plt.xlim([self.x_min, self.x_max])
        ax.set_ylabel(ylabel, fontsize=12)

    def plot_visually_weighted_models(self, ax, weight_type='posterior', xlabel='x',
                                      pred_alpha=0.5, pred_linestyle='-', pred_color='black',
                                      model_linestyle='-', model_alpha=0.7):
        if weight_type == 'posterior':
            weights = self.pointwise_model_weights
        elif weight_type == 'prior':
            weights = np.exp(self.pointwise_log_model_priors)

        # Set current plt axis to ax
        plt.sca(ax)

        # For the plotting, we will use the current colorcycle
        ax._get_lines.set_color_cycle(self.colorcycle)
        # Get an actual color cycle oject
        color_cycle = ax._get_lines.color_cycle

        for i in xrange(self.n_models):
            # Max alpha for the plots
            max_alpha = model_alpha

            # Cited from http://matplotlib.1069221.n5.nabble.com/Varying-alpha-in-ListedColormap-not-working-td39950.html

            x = self.x_plot
            y = self.model_predictions[i,:]
            scaling = weights[i,:]

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

            # Create the line collection object
            lc = LineCollection(segments)
            lc.set_color(cmap)

            # Set line properties. 
            # Dashed line
            lc.set_dashes(model_linestyle)
            # line width 3.0
            lc.set_linewidth(self.global_linewidth)

            ax.add_collection(lc)

        # Plot the predictions
        predictions = np.sum(np.multiply(self.model_predictions, weights),axis=0)
        p_line = plt.plot(self.x_plot, predictions)
        plt.setp(p_line, color=pred_color, linewidth=self.global_linewidth, alpha=pred_alpha, linestyle=pred_linestyle,
                 dash_capstyle='round')

        ax.set_xlabel(xlabel)

    def plot_next(self, ax, x_next, y_next, edgecolor='magenta', facecolor='black', edgealpha=0.9, facealpha=0.7, markersize=200):
        # Set current plt axis to ax
        plt.sca(ax)

        # Custom aligned hovering triangle marker
        verts = list(zip([-0.75,0.75,0.0, -0.75, 0.75],[1.4,1.4,0.4, 1.4, 1.4]))

        print(x_next)
        print(y_next)
        # Create the marker above the y_next value for the face
        ax.scatter(x_next, y_next, s=markersize, edgecolors='none', facecolors=facecolor, alpha=facealpha,
                   linewidth=0.01*markersize,
                   marker=(verts,0),
                   zorder=3)
        
        # Create the marker for the edge
        ax.scatter(x_next, y_next, s=markersize, edgecolors=edgecolor, facecolors='none', alpha=edgealpha,
                   linewidth=0.01*markersize,
                   marker=(verts,0),
                   zorder=3)
