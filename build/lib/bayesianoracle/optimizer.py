import numpy as np
import scipy
from collections import Counter
from . import process_objects
from . import misc

class QuadraticBMAOptimizer(process_objects.EnrichedQuadraticBMAProcess):
    def __init__(self, 
                 ndim,
                 init_kernel_range=1.0,
                 init_kernel_var=10.0,
                 kappa_explore=2.0,
                 kappa_detail=0.5,
                 kernel_mult=2.0,
                 precision_alpha=1.0,
                 precision_beta=100.0,
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
        precision_alpha   : (scalar) the alpha value for the gamma prior on precision
        precision_beta    : (scalar) the beta value for the gamma prior on precision
        verbose           : (boolean) report progress throughout optimization?
        """

        super(QuadraticBMAOptimizer, self).__init__(ndim=ndim,
                                                    verbose=verbose,
                                                    init_kernel_range=init_kernel_range)

        self.set_precision_prior_params(precision_alpha, precision_beta)
        
        self.kappa_explore = kappa_explore
        self.kappa_detail = kappa_detail

        # Set the default gamma kernel prior
        self.set_gamma_kernel_prior(init_kernel_range, init_kernel_var)

        # trust radius is kernel_mult * kernel_range
        self.kernel_mult = kernel_mult
        
        self.iteration = 0

        # Number of iterations under the detail regime
        self.detail_run_count = 0
        # Number of iterations with detail points "nearby" above which an
        # exploration step will be taken
        self.near_num_thresh = 10000

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

    def locate_next_point(self, trust_detail=0.0, trust_explore=0.0):
        """
        Locates the next point to begin optimization. Hyperparameters
        are optimized in this function.
        
        args:
        -----
        trust_detail  : (scalar) trust radius to use for detail steps.
        trust_explore : (scalar) trust radius to use for exploration steps.

        if either is set to 0.0, then the default calculation using the 
        current kernel_range to set the trust_radius for the variables
        set to 0.0
        """
        self.optimize_hyperparameters()

        # Use the default trust calculation if trusts are zero
        if trust_detail == 0.0:
            trust_detail = self.kernel_mult*self.bma.kernel_range
        if trust_explore == 0.0:
            trust_explore = self.kernel_mult*self.bma.kernel_range

        # Set the trust radii
        self.trust_detail = trust_detail
        self.trust_explore = trust_explore
        
        # Set the near threshold 
        self.near_thresh = self.trust_detail/5.0

        # Locate current location of the minimum
        if self.verbose:
            print("BayesianOracle>> locating current estimate of the minimum location")
            self.locate_min_point()

        if self.iteration == 0:
            # Increment detail run count and iteration
            self.detail_run_count += 1
            self.iteration += 1
            if self.verbose:
                print("BayesianOracle>> requesting detail point")
            return self.locate_detail_point()

        # Return detail point if only 1 previous detail point has been generated
        if self.detail_run_count <= 1:
            self.detail_run_count +=1
            self.iteration +=1
            if self.verbose:
                print("BayesianOracle>> requesting detail point")
            return self.locate_detail_point()
        # Get all past detail_run_count iterations worth of data. X has all data with columns
        # Corresponding to spatial dimensions
        X = np.vstack([x for (x, f, g, H, b, d, Hchol) in self.data[-self.detail_run_count:]]).T

        # Get pdist matrix
        dists = scipy.spatial.distance.pdist(X, 'euclidean')
        
        # Get the number of distances below thresh
        n_below_thresh = sum([dist < self.near_thresh for dist in dists])

        # If below thresh, then do detail, otherwise do explore
        if n_below_thresh < self.near_num_thresh:
            self.detail_run_count += 1
            self.iteration += 1
            if self.verbose:
                print("BayesianOracle>> requesting detail point")
                print("BayesianOracle>> locating acqusition point... ")
            return self.locate_detail_point()
        else:
            # reset detail run count
            self.detail_run_count = 0
            self.iteration +=1
            if self.verbose:
                print("BayesianOracle>> requesting exploration point")
                print("BayesianOracle>> locating acqusition point... ")
            return self.locate_exploration_point()

    def locate_exploration_point(self):
        return self.locate_acquisition_point(trust_radius=self.trust_explore, kappa=self.kappa_explore)

    def locate_detail_point(self):
        return self.locate_acquisition_point(trust_radius=self.trust_detail, kappa=self.kappa_detail)

    def locate_min_point(self):
        return self.locate_acquisition_point(trust_radius=self.trust_detail, kappa=0.0)

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

    def locate_acquisition_point(self, trust_radius=1.0, kappa=1.0):
        """
        Locates a new location within trust_radius distance of any previous 
        tested locations for future function evaluations. The new location
        is a minimizer of the acquisition function, which takes in parameter
        kappa. Prints out indicators in the case of the verbose option.
        
        args:
        -----
        trust_radius : (scalar) the maximum distance between any new proposed
                       location and any previouslly attempted locations
        kappa        : (scalar) the parameter for the acqusition function.
                       In the case of discounted means, the kappa value
                       determines the level to which uncertainty is used
                       in the acquisition funcftion.

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
        n_seed = 10000

        # create a list of model indices to sample from, then sample it
        index_list = np.arange(0,n_models)
        index_samples = np.random.choice(index_list, (n_seed,), replace=True)

        # Get the counts of the indices
        counter = Counter(index_samples)

        X_search = np.hstack([self.__gen_n_seed_around(X_stack[:,i], counter[i], trust_radius)
                              for i in range(n_models)])

        # Maximize expected improvement over search points
        discounted_means = self.calculate_discounted_mean(X_search, kappa)
        j_min = np.argmin(discounted_means)
        x_search = X_search[:, j_min]
        X_min = discounted_means[j_min]


        def trust_check(x):
            # Trust to any point in 
            return trust_radius - np.min(np.linalg.norm(X_stack - x[:,None], axis=0))

        def dm(x):
            # Get the vectorized form into the correct n x 1 form
            return self.calculate_discounted_mean(np.array([x]).T, kappa)

        
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
            print("BayesianOracle>> additional optimization successful")
            print("BayesianOracle>> minimum of %.5f at" % result.fun)
            print("BayesianOracle>> "+str(result.x))
            m, s, u, n = self.predict_with_unc(np.array([result.x]).T)
            print("BayesianOracle>> mean: %.5f," %m + 
                  " explained std: %.5f," %s + 
                  " unexplained std: %.5f" %u + 
                  " effective sample size: %.2f" %n)
            print("")
            return result.x
        else:
            print("BayesianOracle>> additional optimization failed")
            print("BayesianOracle>> minimum of %.5f at" % X_min)
            print("BayesianOracle>> "+str(x_search))
            m, s, u, n = self.predict_with_unc(np.array([x_search]).T)
            print("BayesianOracle>> mean: %.5f," %m + 
                  " explained std: %.5f," %s + 
                  " unexplained std: %.5f" %u + 
                  " effective sample size: %.2f" %n)
            print("")
            return x_search
