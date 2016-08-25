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
                 init_kernel_var=-1,
                 n_int=50,
                 constraints=[],
                 bool_compact=False,
                 bounding_box=[],
                 kappa_explore=0.80,
                 kappa_detail=0.20,
                 kernel_mult=1.0,
                 kernel_mult_explore=1.0,
                 min_trust=0.01,
                 precision_alpha=2.0,
                 precision_beta=10.0,
                 bias_lambda=1.0,
                 bool_sample_low_density=False,
                 bool_exploration=True,
                 verbose=True):
        """
        Class initializer. 

        args:
        -----
        ndim              : (scalar) number of dimensions of the location space
        init_kernel_range : (scalar) initial guess of the kernel range
        init_kernel_var   : (scalar) guess of the spread in the kernel range
                            values
        n_int             : (integer) max number of samples to use for integration
                            over the kernel range
        constraints       : (list of functions) each included function imposes the constraint that
                            function(x) >= 0. Only works if constraint set is compact.
                            Each function should be vectorized to work
                            on multiple inputs.
        bool_compact      : (boolean) true if the constraint set is compact. 
                            in which case no trust radius is used. If bool_compact
                            is True, then bounding box needs to be specified
        bounding_box      : (ndim x 2 matrix). The i^th row has two values
                            li, ui, where we are sure that the constraint set
                            lies within li and ui in the i^th coordinate.
        kappa_explore     : (scalar) the kappa value to use in the exploration
                            phase
        kappa_detail      : (scalar) the kappa value to use in the detail phase
        kernel_mult       : (scalar) the value used to determine the trust radius
                            from the current estimate of the kernel width
                            i.e. trust = kernel_width*kernel_mult
        min_trust         : (scalar) minimum trust radius size.
        precision_alpha   : (scalar) the alpha value for the gamma prior on precision
        precision_beta    : (scalar) the beta value for the gamma prior on precision
        bool_sample_low_density : (boolean) True if low density points are to be sampled
                            during exploration
        verbose           : (boolean) report progress throughout optimization?
        """

        super(QuadraticBMAOptimizer, self).__init__(ndim=ndim,
                                                    verbose=verbose,
                                                    init_kernel_range=init_kernel_range,
                                                    n_int=n_int,
                                                    kernel_type=kernel_type)
        self.precision_alpha = precision_alpha
        self.precision_beta = precision_beta
        self.set_precision_prior_params(precision_alpha, precision_beta)
        self.set_bias_prior_params(bias_lambda)

        self.init_kernel_range = init_kernel_range
        self.min_trust = min_trust
        self.kappa_explore = kappa_explore
        self.kappa_detail = kappa_detail

        # Set the kernel prior based on specifications
        if init_kernel_var == -1:
            # Use default variance of 25% of init_kernel_rnage
            # init_kernel_var = init_kernel_range*0.25
            # Use default of setting alpha = 3 and determinig scale
            alpha = 2.0
            beta = alpha / init_kernel_range
            init_kernel_var = alpha / beta**2

        if kernel_prior_type == 'Gamma':
            self.set_gamma_kernel_prior(init_kernel_range, init_kernel_var)
        elif kernel_prior_type == 'InvGamma':
            self.set_invgamma_kernel_prior(init_kernel_range, init_kernel_var)
        elif kernel_prior_type == 'LogNormal':
            self.set_lognormal_kernel_prior(init_kernel_range, init_kernel_var)
        else:
            assert False,'kernel_prior_type must be a \'Gamma\', \'InvGamma\', or \'LogNormal\''

        # Set up the contraints information
        self.constraints = constraints
        self.bool_compact = bool_compact
        self.bounding_box = bounding_box

        # trust radius is kernel_mult * kernel_range
        self.kernel_mult = kernel_mult
        self.kernel_mult_explore = kernel_mult_explore
        
        # Number of iterations with detail points "nearby" above which an
        # exploration step will be taken
        self.thresh_factor = 0.05 # Percetage of current kernel to warrant a "close step"
        self.num_near_thresh = 2

        self.bool_sample_low_density = bool_sample_low_density
        self.bool_exploration = bool_exploration

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
        self.prev_phase = 'detail'  # Previous phase
        self.phase='detail'  # Current phase

    def set_precision_beta(self, beta):
        self.precision_beta = beta
        self.set_precision_prior_params(self.precision_alpha, self.precision_beta)

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

    def get_kappa_detail(self):
        return self.kappa_detail
    def get_kappa_explore(self):
        return self.kappa_explore

    def locate_next_point(self, bool_return_fval=False):
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
            # Update phase
            self.prev_phase = 'detail'
            return self.locate_detail_point(bool_return_fval=bool_return_fval)

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


        self.phase = 'detail'
        if num_near > self.num_near_thresh:
            if self.prev_phase == 'detail':
                self.phase = 'low_KL'
            elif self.prev_phase == 'low_KL':
                self.phase = 'detail'

        self.iteration += 1
        self.prev_num_near = num_near
        self.run_count += 1
        self.prev_phase = self.phase

        if self.phase == 'detail':
            if self.verbose:
                print("BayesianOracle>> requesting DETAIL point")
            return self.locate_detail_point(bool_return_fval)
        elif self.phase == 'exploration':
            if self.verbose:
                print("BayesianOracle>> requesting EXPLORATION point")
            return self.locate_exploration_point(bool_return_fval)
        elif self.phase == 'low_KL':
            if self.verbose:
                print("BayesianOracle>> requesting LOW KL point")
            return self.locate_low_KL_point(bool_return_fval)

    def get_prev_phase(self):
        return self.prev_phase

    def locate_low_density_point(self, bool_return_fval=False):
        def fun(X):
            return self.calculate_N_eff_bayesian(X)

        return self.locate_acquisition_point(trust_radius=self.trust_low_density, fun=fun, bool_return_fval=bool_return_fval)

    def locate_low_KL_point(self, bool_return_fval=False):
        def fun(X):
            return self.calculate_KL_bayesian(X)

        return self.locate_acquisition_point(trust_radius=self.trust_low_density, fun=fun, bool_return_fval=bool_return_fval)
    
    def locate_exploration_point(self, bool_return_fval=False):
        def fun(X):
            return self.calculate_discounted_mean(X, self.kappa_explore)
 
        return self.locate_acquisition_point(trust_radius=self.trust_explore, fun=fun, bool_return_fval=bool_return_fval)

    def locate_detail_point(self, bool_return_fval=False):
        def fun(X):
            return self.calculate_discounted_mean(X, self.kappa_detail)

        return self.locate_acquisition_point(trust_radius=self.trust_detail, fun=fun, bool_return_fval=bool_return_fval)

    def locate_min_point(self, bool_return_fval=False):
        def fun(X):
            return self.calculate_discounted_mean(X, 0.0)

        return self.locate_acquisition_point(trust_radius=self.trust_detail, fun=fun, bool_return_fval=bool_return_fval)

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

    def __check_constraints(self, X):
        """
        X : (n x p) matrix of p points for which constraint checking is 
            desired.
        returns
        -------
        (1 x p vector of booleans) with True if the j^th point satisifes 
        the constraints
        """
        # Initialize all to true
        constraint_check = np.tile(True, X.shape[1])
        for i in xrange(len(self.constraints)):
            # Check which points satisfy the i^th constraint
            constraint_check &= (self.constraints[i](X) >= 0)
        
        return constraint_check

    def __gen_n_seed_in_constraints(self, n_seed):
        X_search = np.array([[]])

        n_test = n_seed

        while X_search.shape[1] < n_seed:
            # Generate random seed in bounding box
            # Correct for 1 dim
            if self.ndim == 1:
                cur_samples = np.array([np.random.uniform(self.bounding_box[:,0], self.bounding_box[:,1], n_seed)])
            else:
                cur_samples = np.random.uniform(self.bounding_box[:,0], self.bounding_box[:,1], n_seed)
            # Check if sample satisfies the constraints
            constraint_check = self.__check_constraints(cur_samples)
            # Only take the samples that satisfy the constraints
            X_search = np.hstack([X_search, cur_samples[:, constraint_check]])
        
        # Remove excess
        return X_search[:, 0:n_seed]

    def locate_acquisition_point(self, trust_radius=1.0, fun=None, bool_return_fval=False):
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

        if self.bool_compact:
            # Use uniform sampling from the constraints
            X_search = self.__gen_n_seed_in_constraints(n_seed)
        else:
            # Otherwise use trust radius
            X_search = np.hstack([self.__gen_n_seed_around(X_stack[:,i], counter[i], trust_radius)
                              for i in range(n_models)])

        # Add the previous minima locations as additional seeds if avail
        if not (self.x_min_hist is None):
            X_search = np.hstack([X_search, self.x_min_hist])

        # Minimize fun over the randomly selected points
        X_search_fun = fun(X_search)
        j_min = np.argmin(X_search_fun)
        x_search = X_search[:, j_min]
        X_min = X_search_fun[j_min]

        def trust_check(x):
            # Trust to any point in 
            return trust_radius - np.min(np.linalg.norm(X_stack - x[:,None], axis=0))

        def dm(x):
            # Get the vectorized form into the correct n x 1 form
            return fun(np.array([x]).T)
        
        # dd trust radius constraint if not compact
        if self.bool_compact:
            secondary_constraints = []
            for i in xrange(len(self.constraints)):
                # Create the constraint, knowing that x needs to be vectorized
                # since this is not done in scipy.optimize.minimize
                secondary_constraints.append({'type' : 'ineq', 'fun': 
                                              (lambda x : self.constraints[i](np.array([x]).T))})
        else:
            secondary_constraints = {'type' : 'ineq', 'fun': trust_check}

        # Further optimization with constraints  
        result = scipy.optimize.minimize(dm, x_search,
                                         constraints=secondary_constraints,
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
        m, u, s, n, h = self.predict_bayesian(np.array([x_final]).T, return_H=True)
        print("BayesianOracle>> mean: %.5f," %m + 
              " unexplained std: %.5f" %u + 
              " explained std: %.5f," %s + 
              " effective sample size: %.2f" %n)
        print("")

        if bool_return_fval:
            return x_final, dm(x_final)
        else:
            return x_final
