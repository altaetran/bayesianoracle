import numpy as np
import scipy
import matplotlib.pyplot as plt

from matplotlib.collections import LineCollection
from matplotlib import colors as colors
from matplotlib import gridspec, markers, ticker

class Plotter1D(object):
    def __init__(self,
                 x_range=[-2.0, 2.0],
                 y_range=[-10.0, 10.0],
                 num_points=200,
                 global_linewidth=4.0,
                 global_linewidth_smaller=2.0,
                 colorcycle=['#FF0000', '#FF00FF', '#9933FF', '#6600CC', '#3366FF', '#00CCFF', '#00FFCC', '#FFFF66', '#FF9933']):
        #, '#FF33AD'
        self.x_min = x_range[0]
        self.x_max = x_range[1]
        self.y_min = y_range[0]
        self.y_max = y_range[1]

        self.num_points = num_points

        self.x_plot = np.linspace(self.x_min, self.x_max, self.num_points)
        self.x_grid = np.array([self.x_plot])

        self.y_plot = np.linspace(self.y_min, self.y_max, self.num_points)

        self.bma = None
        self.n_models = 0

        # Plotting options
        self.global_linewidth = global_linewidth
        self.global_linewidth_smaller = global_linewidth_smaller
        self.colorcycle = colorcycle
        
    def set_bma(self, bma):
        self.bma = bma
        self.model_predictions = bma.model_predictions(self.x_grid)
        self.n_models = len(bma.quadratic_models)

    def plot_fun(self, ax, fun, alpha=0.7, linestyle='--', xlabel=r'$x$', ylabel=None):
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

        if xlabel is not None:
            ax.set_xlabel(xlabel, multialignment='center')
        if ylabel is not None:
            ax.set_ylabel(ylabel, multialignment='center')

        return func_line

    def plot_data(self, ax, X=None, y=None, color='black', alpha=0.5, xlabel=r'$x$', ylabel=None, bool_color_cycled=False):
        # Set current plt axis to ax
        plt.sca(ax)

        # Scatter of observations models (sampled models)
        if X is None:
            X = self.bma.get_X_stack()
            y = self.bma.get_y_stack()

        if bool_color_cycled:
            # Set color cycle
            ax._get_lines.set_color_cycle(self.colorcycle)
            # Get an actual color cycle oject
            color_cycle = ax._get_lines.color_cycle
            
            for i in xrange(X.shape[1]):
                cur_color = next(color_cycle)
                ax.scatter(X[:,i], y[i], s=100.0, color=cur_color, alpha=alpha, zorder=3)
                
            # Figure legend object
            scat = ax.scatter(-1e100, 1e100, s=100.0, c=color, alpha=alpha, zorder=1, facecolors='none')
        else:
            scat = ax.scatter(X, y, s=100.0, c=color, alpha=alpha, zorder=3)

        # Set range
        plt.ylim([self.y_min, self.y_max])
        plt.xlim([self.x_min, self.x_max])

        if xlabel is not None:
            ax.set_xlabel(xlabel, multialignment='center')
        if ylabel is not None:
            ax.set_ylabel(ylabel, multialignment='center')

        # Return the last data point
        return scat

    def plot_kernel(self, ax, x, kernel_range, alpha=1.0, color='red', linestyle='-', xlabel=r'$x$', ylabel=None):
        # Set current plt axis to ax
        plt.sca(ax)

        # Get kernel evaluations
        kernel_plot = self.bma.kernel_func(np.linalg.norm(self.x_grid-x[:, None], axis=0), kernel_range)

        # Plot kernel evaluations
        kernel_line, = ax.plot(self.x_plot, kernel_plot)
        # Set line properties 
        plt.setp(kernel_line, color=color, linewidth=self.global_linewidth, alpha=alpha, linestyle=linestyle,
                 dash_capstyle='round')
        
        # Set range
        plt.ylim([0, np.max([1, np.max(kernel_plot)])])
        plt.xlim([self.x_min, self.x_max])

        # Set labels
        if xlabel is not None:
            ax.set_xlabel(xlabel, multialignment='center')
        if ylabel is not None:
            ax.set_ylabel(ylabel, multialignment='center')

        return kernel_line
        
    def plot_kernel_at_data(self, ax, x, kernel_range, X=None, alpha=0.5, color='red', xlabel=r'$x$', ylabel=None, bool_color_cycled=True):
        # Set current plt axis to ax
        plt.sca(ax)

        # Get the data if not given
        if X is None:
            X = self.bma.get_X_stack()

        # Get the kernel values of the data
        y = self.bma.calc_relevance_weights(np.array([x]).T, kernel_range)

        if bool_color_cycled:
            # Set color cycle
            ax._get_lines.set_color_cycle(self.colorcycle)
            # Get an actual color cycle oject
            color_cycle = ax._get_lines.color_cycle
            
            for i in xrange(X.shape[1]):
                cur_color = next(color_cycle)
                ax.scatter(X[:,i], y[i], s=100.0, color=cur_color, alpha=alpha, zorder=3)
                
            # Figure legend object
            kernel_scat = ax.scatter(-1e100, 1e100, s=100.0, c=color, alpha=alpha, zorder=1, facecolors='none')
        else:
            kernel_scat = ax.scatter(X, y, s=100.0, c=color, alpha=alpha, zorder=3)

        # Set range
        plt.xlim([self.x_min, self.x_max])

        # Set labels
        if xlabel is not None:
            ax.set_xlabel(xlabel, multialignment='center')
        if ylabel is not None:
            ax.set_ylabel(ylabel, multialignment='center')

        # Return the last data point
        return kernel_scat

    def plot_data_locations(self, ax, X=None, alpha=1.0, color='black', linestyle=':', xlabel=r'$x$', zorder=1, bool_color_cycled=False):
        """
        Does not change the plot range
        """
        # Set current plt axis to ax
        plt.sca(ax)
        
        # Get the data if not given
        if X is None:
            X = self.bma.get_X_stack()

        # Number of points for plotting in the y direction
        num_points = 100

        # First get the y lim and create y range for plotting
        cur_y_min, cur_y_max = ax.get_ylim()
        y_plot = np.linspace(cur_y_min, cur_y_max, num_points)

        if bool_color_cycled:
            # Set color cycle
            ax._get_lines.set_color_cycle(self.colorcycle)
            # Get an actual color cycle oject
            color_cycle = ax._get_lines.color_cycle

            # Plot vertical lines
            for i in xrange(X.shape[1]):
                loc_line, = plt.plot(np.tile(X[:,i], num_points), y_plot, zorder=zorder)
                plt.setp(loc_line, linewidth=self.global_linewidth_smaller, alpha=alpha, linestyle=linestyle,
                         dash_capstyle='round')
        else:
            # Plot vertical lines
            for i in xrange(X.shape[1]):
                loc_line, = plt.plot(np.tile(X[:,i], num_points), y_plot, zorder=zorder)
                plt.setp(loc_line, color=color, linewidth=self.global_linewidth_smaller, alpha=alpha, linestyle=linestyle,
                         dash_capstyle='round')

        # Return a single instance of loc_line
        return loc_line

    def plot_quad(self, ax, quad, alpha=1.0, color='magenta', linestyle='--', xlabel=r'$x$'):
        """ Plots the function, fun, along the x range 

        Parameters
        ----------
        ax : (matplotlib axis) to plot along
        alpha : (float in range [0,1]) transparency of line
        """
        # Set current plt axis to ax
        plt.sca(ax)

        # Get function values along x_grid
        F = quad.predict_batch(self.x_grid)

        # Plot function
        func_line, = ax.plot(self.x_plot, F)
        # Set line properties 
        plt.setp(func_line, color=color, linewidth=self.global_linewidth, alpha=alpha, linestyle=linestyle,
                 dash_capstyle='round')

        # Set range
        plt.ylim([self.y_min, self.y_max])
        plt.xlim([self.x_min, self.x_max])

        ax.set_xlabel(xlabel)

        return func_line

    def plot_model_mean(self, ax, model_ind, alpha=1.0, color='magenta', linestyle='-', xlabel=r'$x$'):
        return self.plot_quad(ax, self.bma.quadratic_models[model_ind], alpha=alpha, color=color, linestyle=linestyle, xlabel=xlabel)

    def plot_model(self, ax, model_ind, kernel_range=-1, bool_dataless=False, alpha=1.0, color='magenta', linestyle='--', xlabel=r'$x$'):
        # Set current plt axis to ax
        plt.sca(ax)

        # Create meshgrid 
        x_mesh, y_mesh = np.meshgrid(self.x_plot, self.y_plot)

        model_probs_mesh = np.array([[]])

        # Evaluate on meshgrid
        for k in xrange(x_mesh.shape[1]):
            x_test = np.array([x_mesh[k,:]])
            y_test = y_mesh[k,:]

            bma_prob, model_probs, model_weights = self.bma.pdf(x_test, y_test, kernel_range=kernel_range, bool_return_model_pdfs=True, bool_dataless=bool_dataless)
            model_prob = model_probs[model_ind,:]

            if k == 0:
                model_probs_mesh = np.array([model_prob])
            else:
                model_probs_mesh = np.vstack([model_probs_mesh, model_prob])
                
        # Create custom colormap from white to color
        #colormap = colormap_fade_buffer('white', color)
        colormap = colormap_fade_buffer(color,'white')                

        # Contour plot
        upper = np.ceil(np.max(model_probs_mesh)*10.0)/10.0

        # Create levels with 5 levels total
        v = np.linspace(0.00, upper, 6, endpoint=True)
        heatmap = plt.contourf(x_mesh, y_mesh, model_probs_mesh, v, cmap=colormap)

        # Set range
        plt.ylim([self.y_min, self.y_max])
        plt.xlim([self.x_min, self.x_max])

        # Get the horizontal colorbar
        cbar = plt.colorbar(heatmap, orientation='horizontal', pad=0.1)
        cbar.ax.set_xlabel('probability')
        tick_locator = ticker.MaxNLocator(nbins=5)
        cbar.locator = tick_locator
        cbar.update_ticks()

        ax.set_xlabel(xlabel)

    def get_bayesian_bma_evals(self):
        Z_rolled, kernel_ranges, likelihoods, avg_kernel_ranges, normalized_priors, integrated_relevance_weights = self.bma.predict_bayesian(self.x_grid, return_likelihoods=True)

        # Save the predictions
        self.bayesian_means = Z_rolled[0,:]
        self.bayesian_unexp_std = Z_rolled[1,:]
        self.bayesian_exp_std = Z_rolled[2,:]
        self.bayesian_N_eff = Z_rolled[3,:]
        self.bayesian_kernel_ranges = kernel_ranges
        self.bayesian_likelihoods = likelihoods
        self.bayesian_avg_kernel_ranges = avg_kernel_ranges
        self.bayesian_normalized_priors = normalized_priors
        self.integrated_relevance_weights = integrated_relevance_weights

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

    def plot_means(self, ax, bool_bayesian=True, alpha=0.7, xlabel=r'$x$'):
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

    def plot_discounted_means(self, ax, kappa, bool_bayesian=True, linestyle='--', alpha=0.9, color='magenta', xlabel=r'$x$'):
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

    def plot_confidence(self, ax, bool_bayesian=True, xlabel=r'$x$', exp_color='blue', unexp_color='red', std_color='black'):
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
        exp_line = ax.fill_between(self.x_plot,means-exp_std, means+exp_std, facecolor=exp_color, interpolate=True, alpha=0.2)
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
        exp_fill = plt.Rectangle((0, 0), 1, 1, fc=exp_color, alpha=0.5)
        unexp_fill = plt.Rectangle((0, 0), 1, 1, fc=unexp_color, alpha=0.5)
        std_fill = plt.Rectangle((0, 0), 1, 1, fc=std_color, alpha=0.3)

        return exp_fill, unexp_fill, std_fill
        
    def plot_bayesian_likelihoods(self, ax, bool_posterior=True, bool_colorbar=True, xlabel=r'$x$', ylabel='kernel range', color='#663300', kernel_ranges_range=[1.0e-2,1.0e1], colorbarlabel='probability'):
        # Set current plt axis to ax
        plt.sca(ax)

        padded_kernel_ranges = self.bayesian_kernel_ranges

        if bool_posterior:
            padded_likelihoods = self.bayesian_likelihoods
        else:
            prior = self.bayesian_normalized_priors
            # Now tile it
            padded_likelihoods = np.array([prior,]*self.num_points).transpose()
            
        # Pad the data to fill the full range
        if self.bayesian_kernel_ranges.min() > kernel_ranges_range[0]:
            # Left pad
            padded_kernel_ranges = np.insert(padded_kernel_ranges, 0, kernel_ranges_range[0])
            padded_likelihoods = np.insert(padded_likelihoods, 0, 0.0, axis=0)
        if self.bayesian_kernel_ranges.max() < kernel_ranges_range[1]:
            # Right pad
            padded_kernel_ranges = np.append(padded_kernel_ranges, kernel_ranges_range[1])
            padded_likelihoods = np.insert(padded_likelihoods, -1, 0.0, axis=0)            

        colormap = colormap_fade_buffer('white', color) 

        upper = np.ceil(np.max(padded_likelihoods)*10.0)/10.0
        v = np.linspace(0.00, upper, 11, endpoint=True)
        # Plot the countour
        heatmap = ax.contourf(self.x_plot, padded_kernel_ranges, padded_likelihoods, v, cmap=colormap)
        ax.set_yscale('log')

        # Set range
        plt.xlim([self.x_min, self.x_max])
        plt.ylim(kernel_ranges_range)

        # Set labels
        if xlabel is not None:
            ax.set_xlabel(xlabel, multialignment='center')
        if ylabel is not None:
            ax.set_ylabel(ylabel, multialignment='center')

        # Get the horizontal colorbar if said
        if bool_colorbar:
            cbar = plt.colorbar(heatmap, orientation='horizontal', pad=0.1)
            cbar.ax.set_xlabel(colorbarlabel)

        return heatmap
        
    def plot_bayesian_avg_kernel_ranges(self, ax, xlabel=r'$x$', ylabel='avg kernel range', alpha=1.0, linestyle='-', kernel_ranges_range=[0.0,1.0], color='#663300'):
        # Set current plt axis to ax
        plt.sca(ax)

        # Set colorcylce
        ax._get_lines.set_color_cycle(self.colorcycle)

        # Plot the average kernel range
        p_line, = ax.plot(self.x_plot, self.bayesian_avg_kernel_ranges)
        plt.setp(p_line, color=color, linewidth=self.global_linewidth, alpha=alpha, linestyle=linestyle,
                 dash_capstyle='round')

        # Plot range
        plt.xlim([self.x_min, self.x_max])
        plt.ylim(kernel_ranges_range)

        # Set labels
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)

        return p_line

    def plot_N_eff(self, ax, bool_bayesian=True, alpha=0.7, xlabel=r'$x$', ylabel='N_eff'):
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

    def plot_model_priors(self, ax, bool_bayesian=False, alpha=1.0, linestyle='-', xlabel=r'$x$', ylabel='prior'):
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
        ax.set_ylim([0, 1])

        # Set labels
        if xlabel is not None:
            ax.set_xlabel(xlabel, multialignment='center')
        if ylabel is not None:
            ax.set_ylabel(ylabel, multialignment='center')

        return prior_lines

    def plot_model_posteriors(self, ax, bool_bayesian=False, alpha=1.0, linestyle='-', xlabel=r'$x$', ylabel='posterior'):
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

        # Set labels
        if xlabel is not None:
            ax.set_xlabel(xlabel, multialignment='center')
        if ylabel is not None:
            ax.set_ylabel(ylabel, multialignment='center')

        return posterior_lines

    def plot_marginal_likelihoods(self, ax, bool_bayesian=False, alpha=1.0, linestyle='-', xlabel=None, ylabel='marginal likelihoods'):
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
            p_line, = plt.plot(self.x_plot, marginal_likelihoods[i,:])
            plt.setp(p_line, linewidth=self.global_linewidth, alpha=alpha, linestyle=linestyle,
                     dash_capstyle='round')

            marginal_likelihood_lines.append(p_line)

        # Use log scale
        ax.set_yscale('log')
        # Set number of ticks
        up = 0
        lb = np.floor(np.min(np.log10(marginal_likelihoods)))
        ax.set_yticks(np.logspace(lb, up, num=5))
        # Set minor ticks off
        plt.minorticks_off()

        # Set range
        plt.xlim([self.x_min, self.x_max])

        # Set labels
        if xlabel is not None:
            ax.set_xlabel(xlabel, multialignment='center')
        if ylabel is not None:
            ax.set_ylabel(ylabel, multialignment='center')

        return marginal_likelihood_lines

    def plot_kernel_weights(self, ax, bool_bayesian=False, alpha=0.35, xlabel=None, ylabel='effective sample size'):
        # Get the model priors
        if bool_bayesian:
            kernel_weights = self.integrated_relevance_weights
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
        plt.ylim([0.0, np.max(cumulant)])
        plt.xlim([self.x_min, self.x_max])

        # Set labels
        if xlabel is not None:
            ax.set_xlabel(xlabel, multialignment='center')
        if ylabel is not None:
            ax.set_ylabel(ylabel, multialignment='center')

    def plot_visually_weighted_models(self, ax, weight_type='posterior', xlabel=r'$x$',
                                      ylabel=None,
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
            converter = colors.ColorConverter()
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

        # Set labels
        if xlabel is not None:
            ax.set_xlabel(xlabel, multialignment='center')
        if ylabel is not None:
            ax.set_ylabel(ylabel, multialignment='center')

    def plot_next(self, ax, x_next, y_next, edgecolor='magenta', facecolor='black', edgealpha=0.9, facealpha=0.7, markersize=200):
        # Set current plt axis to ax
        plt.sca(ax)

        # Custom aligned hovering triangle marker
        verts = list(zip([-0.75, 0.75, 0.0, -0.75, 0.75], [1.4, 1.4, 0.4, 1.4, 1.4]))
        # Non hovering triangle marker
        centered_verts = list(zip([-0.75, 0.75, 0.0, -0.75, 0.75], [0.5, 0.5, -0.5, 0.5, 0.5]))

        # Create the marker above the y_next value for the face
        ax.scatter(x_next, y_next, s=markersize, edgecolors='none', facecolors=facecolor, alpha=facealpha,
                   linewidth=0.01*markersize, marker=(verts,0), zorder=3)
        
        # Create the marker for the edge
        ax.scatter(x_next, y_next, s=markersize, edgecolors=edgecolor, facecolors='none', alpha=edgealpha,
                   linewidth=0.01*markersize, marker=(verts,0), zorder=3)

        # Create imaginary markers for the legend

        # Create the marker above the y_next value for the face
        marker = ax.scatter(-1e100, -1e100, s=markersize, edgecolors=edgecolor, facecolors=facecolor, alpha=edgealpha,
                          linewidth=0.005*markersize, marker=(centered_verts,0), zorder=3)
        
        # return the combined object for the legend
        return marker

    def draw_vertical_line(self, ax, x_pos, marker_label, color='black', alpha=0.7, linestyle='--', alignment='center', multialignment='center'):
        # Set current axis
        plt.sca(ax)

        vertical_line = plt.plot(np.tile(x_pos, self.y_plot.shape[0]), self.y_plot)
        plt.setp(vertical_line, color=color, linewidth=self.global_linewidth, alpha=alpha, linestyle=linestyle,
                 dash_capstyle='round')

        # Create label
        cur_y_min, cur_y_max = ax.get_ylim()
        plt.text(x_pos, cur_y_max*1.02, marker_label, ha=alignment, multialignment=multialignment)
    
def colormap_fade_buffer(color1='white', color2='red'):
    # Get color converter
    converter = colors.ColorConverter()

    color1_rgba = converter.to_rgba(color1)
    color2_rgba = converter.to_rgba(color1)
    color3_rgba = converter.to_rgba(color2)
    color4_rgba = converter.to_rgba(color2)

    color_rgbas = [color1_rgba, color2_rgba, color3_rgba, color4_rgba]

    color_ranges = [0, 0.1, 0.9, 1.0]
    
    matrices = []
    for i in range(4):
        matrices.append(np.zeros((len(color_ranges), 3)))

    # Populate matrices
    for i in range(4):
        # Initialize beginning and end entires
        matrix = matrices[i]
        # Begin x value
        matrix[0,0] = 0.0
        # End x value
        matrix[-1,0] = 1.0
        # Unused matrix values
        matrix[0,1] = 0.0
        matrix[-1,-1] = 1.0
        for j in range(len(color_ranges)-1):
            # Set colors for transitions
            matrix[j,2] = color_rgbas[j][i]
            matrix[j+1,1] = color_rgbas[j+1][i]
            # Set x position
            matrix[j,0] = color_ranges[j]
            
    # Make cdict
    cdict = {'red' : matrices[0], 'green' : matrices[1], 'blue' : matrices[2]}

    colormap = colors.LinearSegmentedColormap('custommap', cdict, N=256)

    return colormap


