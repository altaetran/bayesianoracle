import numpy as np
import scipy
from collections import Counter
from . import process_objects
from . import misc

class QuadraticBMAOptimizer(process_objects.EnrichedQuadraticBMAProcess):
    def __init__(self, 
                 ndim,
                 kernel_type='Gaussian',
                 kernel_prior_type='Gamma',
                 init_kernel_range=1.0,
                 init_kernel_var=1e2,
                 kappa_explore=1.0,
                 kappa_detail=0.10,
                 kernel_mult=1.0,
                 kernel_mult_explore=2.0,
                 min_trust=0.01,
                 precision_alpha=1.0,
                 precision_beta=10.0,
                 verbose=True):
        """
        Class initializer. 

        args:
        -----
        ndim              : (scalar) number of dimensions of the location space
        init_kernel_range : (scalar) initial guess of the kernel range
        init_kernel_var   : (scalar) guess of the spread in the kernel range
                            values
        kappa_explore     : (scalar) the kappa value to use in the exploration
                            phase
        kappa_detail      : (scalar) the kappa value to use in the detail phase
        kernel_mult       : (scalar) the value used to determine the trust radius
                            from the current estimate of the kernel width
                            i.e. trust = kernel_width*kernel_mult
        min_trust         : (scalar) minimum trust radius size.
        precision_alpha   : (scalar) the alpha value for the gamma prior on precision
        precision_beta    : (scalar) the beta value for the gamma prior on precision
        verbose           : (boolean) report progress throughout optimization?
        """

        super(QuadraticBMAOptimizer, self).__init__(ndim=ndim,
                                                    verbose=verbose,
                                                    init_kernel_range=init_kernel_range,
                                                    kernel_type=kernel_type)

        self.set_precision_prior_params(precision_alpha, precision_beta)

        self.init_kernel_range = init_kernel_range
        self.min_trust = min_trust
        self.kappa_explore = kappa_explore
        self.kappa_detail = kappa_detail

        # Set the kernel prior based on specifications
        if kernel_prior_type == 'Gamma':
            self.set_gamma_kernel_prior(init_kernel_range, init_kernel_var)
        elif kernel_prior_type == 'InvGamma':
            self.set_invgamma_kernel_prior(init_kernel_range, init_kernel_var)
        elif kernel_prior_type == 'LogNormal':
            self.set_lognormal_kernel_prior(init_kernel_range, init_kernel_var)
        else:
            assert False,'kernel_prior_type must be a \'Gamma\', \'InvGamma\', or \'LogNormal\''

        # trust radius is kernel_mult * kernel_range
        self.kernel_mult = kernel_mult
        self.kernel_mult_explore = kernel_mult_explore
        
        # Number of iterations with detail points "nearby" above which an
        # exploration step will be taken
        self.thresh_factor = 0.2 # Percetage of current kernel to warrant a "close step"
        self.num_near_thresh = 5

        self.__init_iteration_variables()

    def __init_iteration_variables(self):
        """
        Initializes the variables that are used for the phase control
        """
        self.iteration = 0  # Iteration number
        self.min_hist = None # (iteration dimensional vector) of min values
        self.x_min_hist = None  # (n x iteration matrix) of min locations
        self.run_count = 0  # Iterations since beginning of run
        self.prev_num_near = 0  # Previous count of iterations since x_min hasn't changed
        self.prev_phase = 'low_density'  # Previous phase
        self.phase='detail'  # Current phase

    def set_kappa_detail(self, kappa):
        """ 
        Setter for the kappa value for the detail steps

        args:
        -----
        kappa : (scalar) value for the kappa parameter in the discounted means
        """
        self.kappa_detail = kappa

    def set_kappa_explore(self, kappa):
        """ 
        Setter for the kappa value for the detail steps

        args:
        -----
        kappa : (scalar) value for the kappa parameter in the discounted means
        """
        self.kappa_explore = kappa

    def locate_next_point(self):
        """
        Locates the next point to begin optimization. Hyperparameters
        are optimized in this function.
        """

        self.optimize_hyperparameters()

        # Use the default trust calculation if trusts are zero
        trust = self.kernel_mult*self.bma.kernel_range
        trust_explore = self.kernel_mult_explore*self.bma.kernel_range
        #trust = self.init_kernel_range
        #trust_explore = self.init_kernel_range*self.kernel_mult_explore
        self.trust_detail = np.max([trust, self.min_trust])
        self.trust_explore = np.max([trust_explore, self.min_trust])
        self.trust_low_density = np.max([trust_explore, self.min_trust])
        
        # Set the near threshold 
        self.near_thresh = self.trust_detail*self.thresh_factor

        # Locate current location of the minimum
        if self.verbose:
            print("BayesianOracle>> locating current estimate of the minimum location")
        x_min = self.locate_min_point()  # located expected minimum
        min_val = self.predict(np.array([x_min]).T)[0,0]

        # Add location of estimated minimum to the x_min_hist
        if self.iteration == 0:
            self.min_hist = np.array([min_val])
            self.x_min_hist = np.array([x_min]).T
        else:
            self.x_min_hist = np.hstack([self.x_min_hist, 
                                         np.array([x_min]).T])
            self.min_hist = np.vstack([self.min_hist, np.array([min_val])])

        # Default case if no other information is available
        if (self.iteration == 0) or (self.run_count < 1):
            # Increment detail run count and iteration
            self.run_count += 1
            self.iteration += 1
            if self.verbose:
                print("BayesianOracle>> requesting DETAIL point")
            return self.locate_detail_point()

        num_near = self.prev_num_near

        # Get the distance between new min and old min
        dist_x_min = scipy.linalg.norm(self.x_min_hist[:,-1]-self.x_min_hist[:,-2])
        diff_min = self.min_hist[-1]-self.min_hist[-2]

        # If close enough, then increment num near
        if dist_x_min < self.near_thresh:
            num_near += 1
        else:
            # Reset run otherwise
            self.run_count = 0
            num_near = 0

        if self.verbose:
            print("BayesianOracle>> minimum estimate summary:")
            print("BayesianOracle>> distance from previous minimum location: %.5f" % dist_x_min)
            print("BayesianOracle>> change in minimum value from previous iteration %.5f" % diff_min)
            print("BayesianOracle>> number of iterations with small steps relative to kernel width: " + str(num_near))
            print("")


        if num_near < self.num_near_thresh:
            self.phase = 'detail'
        elif self.prev_phase == 'detail':
            self.phase = 'exploration'
        elif self.prev_phase == 'exploration':
            self.phase = 'low_density'
        else:
            self.phase = 'detail'

        """
        if self.prev_phase == 'detail':
            self.phase = 'exploration'
        elif self.prev_phase == 'exploration':
            self.phase = 'low_density'
        else:
            self.phase = 'detail'
        """
        self.iteration += 1
        self.prev_num_near = num_near
        self.run_count += 1
        self.prev_phase = self.phase

        if self.phase == 'detail':
            if self.verbose:
                print("BayesianOracle>> requesting DETAIL point")
            return self.locate_detail_point()
        elif self.phase == 'exploration':
            if self.verbose:
                print("BayesianOracle>> requesting EXPLORATION point")
            return self.locate_exploration_point()
        else:
            if self.verbose:
                print("BayesianOracle>> requesting RANDOM LOW DENSITY point")
            return self.locate_low_density_point()

    def locate_low_density_point(self):
        def fun(X):
            return self.calculate_N_eff(X)

        return self.locate_acquisition_point(trust_radius=self.trust_low_density, fun=fun)

    def locate_exploration_point(self):
        def fun(X):
            return self.calculate_discounted_mean(X, self.kappa_explore)
 
        return self.locate_acquisition_point(trust_radius=self.trust_explore, fun=fun)

    def locate_detail_point(self):
        def fun(X):
            return self.calculate_discounted_mean(X, self.kappa_detail)

        return self.locate_acquisition_point(trust_radius=self.trust_detail, fun=fun)

    def locate_min_point(self):
        def fun(X):
            return self.calculate_discounted_mean(X, 0.0)

        return self.locate_acquisition_point(trust_radius=self.trust_detail, fun=fun)

    def __gen_n_seed_around(self, x0, n_seed, trust_radius):
        """
        Creates n_seed points sampled uniformly from a ball about x0
        of radius trust_radius

        args:
        -----

        x0           : (n dimensional vector) containing the location 
                       center
        n_seed       : (scalar) number of points to be sampled 
        trust_radius : (scalar) radius of the ball

        returns:
        --------
        (n x n_seed matrix) of n_seed locations within trust_radius
        distance from x0.
        """
        # Generate random lengths in [0,trust_radius]
        U = scipy.random.random(size=(n_seed, 1))
        lengths = trust_radius*(U**(1.0/self.ndim))
        # Get uniformly distributed directions
        directions = scipy.random.randn(n_seed, self.ndim)
        row_norms = np.sum(directions**2,axis=1)**(1./2)

        # Normalize the directions, then multiply by the lengths
        row_mult = (row_norms / lengths.T).T
        X_search = (directions / row_mult) + x0

        # Finally transpose to get a n x n_seed matrix
        X_search = X_search.T

        return X_search

    def locate_acquisition_point(self, trust_radius=1.0, fun=None):
        """
        Locates a new location within trust_radius distance of any previous 
        tested locations for future function evaluations. The new location
        is a minimizer of the acquisition function, fun.
        
        Prints out indicators in the case of the verbose option.
        
        args:
        -----
        trust_radius : (scalar) the maximum distance between any new proposed
                       location and any previouslly attempted locations
        fun          : (function handle) the function to be minimized

        returns:
        --------
        (n dimensional vector) of the new proposed location. 
        """

        if self.verbose:
            misc.tic()

        # Get number of models
        n_models = self.get_n_models()

        # (n x n_models) stack of observations
        X_stack = self.get_X_stack()

        # Seed the trust region and find the best point
        n_seed = 1000*self.ndim

        # create a list of model indices to sample from, then sample it
        index_list = np.arange(0,n_models)
        index_samples = np.random.choice(index_list, (n_seed,), replace=True)

        # Get the counts of the indices
        counter = Counter(index_samples)

        X_search = np.hstack([self.__gen_n_seed_around(X_stack[:,i], counter[i], trust_radius)
                              for i in range(n_models)])

        # Add the previous minima locations as additional seeds if avail
        if not (self.x_min_hist is None):
            X_search = np.hstack([X_search, self.x_min_hist])

        # Minimize discounted means
        discounted_means = fun(X_search)
        j_min = np.argmin(discounted_means)
        x_search = X_search[:, j_min]
        X_min = discounted_means[j_min]

        def trust_check(x):
            # Trust to any point in 
            return trust_radius - np.min(np.linalg.norm(X_stack - x[:,None], axis=0))

        def dm(x):
            # Get the vectorized form into the correct n x 1 form
            return fun(np.array([x]).T)
        
        # dd trust radius constraint
        constraints = {'type' : 'ineq', 'fun': trust_check}

        # Create results 
        result = scipy.optimize.minimize(dm, x_search,
                                         constraints=constraints,
                                         #jac=True,
                                         options={'maxiter':100},
                                         method='COBYLA')
        
        if self.verbose:
            print("BayesianOracle>> found in %.1f seconds" % misc.toc())

        # Check if the objective function decreased from random search
        b_success = (X_min > result.fun) and (trust_check(result.x) >= 0)

        # If additional optimization was successful, return result, otherwise return x_search
        if b_success:
            x_final = result.x  # Get the optimization result
            print("BayesianOracle>> additional optimization successful")
            print("BayesianOracle>> minimum of %.5f at" % result.fun)
        else:
            x_final = x_search  # Otherwise get randomized search results
            print("BayesianOracle>> additional optimization failed")
            print("BayesianOracle>> minimum of %.5f at" % X_min)

        print("BayesianOracle>> " + str(x_final))

        # Get information about the current location
        m, u, s, n = self.predict_with_unc(np.array([x_final]).T)
        print("BayesianOracle>> mean: %.5f," %m + 
              " unexplained std: %.5f" %u + 
              " explained std: %.5f," %s + 
              " effective sample size: %.2f" %n)
        print("")

        return x_final
