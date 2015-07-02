import numpy as np
import scipy
from collections import Counter
from . import process_objects
from . import misc

class QuadraticBMAOptimizer(process_objects.EnrichedQuadraticBMAProcess):
    def __init__(self, 
                 ndim,
                 verbose=True):
        super(QuadraticBMAOptimizer, self).__init__(ndim=ndim,
                                                    verbose=verbose)
        
        self.kappa_explore = 0.1
        self.kappa_detail = 0.1
        self.trust_explore = 3.0
        self.trust_detail = 3.0

        # trust radius is kernel_mult * kernel_range
        self.kernel_mult = 2.0
        
        self.iteration = 0
        # Number of iterations under the detail regime
        self.detail_run_count = 0
        # Number of iterations with detail points "nearby" above which an
        # exploration step will be taken
        self.near_num_thresh = 10000
        # Threshold for what is "near"

    def set_kappa_detail(self, kappa):
        """ 
        Setter for the kappa value for the detail steps

        args
        kappa : value for the kappa parameter
        """
        self.kappa_detail = kappa
    def set_kappa_explore(self, kappa):
        """ 
        Setter for the kappa value for the detail steps

        args
        kappa : value for the kappa parameter
        """
        self.kappa_explore = kappa

    def locate_next_point(self, trust_detail=0.0, trust_explore=0.0):
        """
        Locates the next point to begin optimization
        """

        # Optimize
        self.optimize_hyperparameters()

        # Use the defaults if trusts are zero
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
            print("bayesianoracle>> locating current estimate of the minimum location")
            self.locate_min_point()
            print("bayesianoracle>> ")

        if self.iteration == 0:
            # Increment detail run count and iteration
            self.detail_run_count += 1
            self.iteration += 1
            if self.verbose:
                print("bayesianoracle>> requesting detail point")
            return self.locate_detail_point()

        # Return detail point if only 1 previous detail point has been generated
        if self.detail_run_count <= 1:
            self.detail_run_count +=1
            self.iteration +=1
            if self.verbose:
                print("bayesianoracle>> requesting detail point")
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
                print("bayesianoracle>> requesting detail point")
                print("BayesianOracle>> locating acqusition point... ")
            return self.locate_detail_point()
        else:
            # reset detail run count
            self.detail_run_count = 0
            self.iteration +=1
            if self.verbose:
                print("bayesianoracle>> requesting exploration point")
                print("BayesianOracle>> locating acqusition point... ")
            return self.locate_exploration_point()

    def locate_exploration_point(self):
        return self.locate_acquisition_point(trust_radius=self.trust_explore, kappa=self.kappa_explore)

    def locate_detail_point(self):
        return self.locate_acquisition_point(trust_radius=self.trust_detail, kappa=self.kappa_detail)

    def locate_min_point(self):
        return self.locate_acquisition_point(trust_radius=self.trust_detail, kappa=0.0)

    def locate_acquisition_point(self, trust_radius=1.0, kappa=1.0):
        """ Suggest a new point, X, that minimizes the expected improvement

        Parameters
        ----------
        tr : Trust radius for the EI minimization. Defaults to 10. All solutions
             proposed must be within trust_radius distance from x0 

        Output
        -------
        X_guess : shape = (n_features,) """

        if self.verbose:
            misc.tic()

        # Get number of models
        n_models = self.get_n_models()

        # (n x n_models) stack of observations
        X_stack = self.get_X_stack()

        # Seed the trust region and find the best point
        n_seed = 10000

        def gen_n_seed_around(x0, n_seed, trust_radius):
            """
            Creates n_seed points sampled uniformly from a ball about x0
            of radius trust_radius
            
            args
            
            x0   : (n dimensional vector) containing the location center
            n_seed : (scalar) number of points to be sampled 
            trust_radius : (scalar) radius of the ball
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

        # create a list of model indices to sample from, then sample it
        index_list = np.arange(0,n_models)
        index_samples = np.random.choice(index_list, (n_seed,), replace=True)

        # Get the counts of the indices
        counter = Counter(index_samples)

        X_search = np.hstack([gen_n_seed_around(X_stack[:,i], counter[i], trust_radius)
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

        
        constraints = {'type' : 'ineq', 'fun': trust_check}
        result = scipy.optimize.minimize(dm, x_search,
                                         constraints=constraints,
                                         #jac=True,
                                         options={'maxiter':100},
                                         method='COBYLA')
        
        if self.verbose:
            print("BayesianOracle>> found in " + str(misc.toc()) + " seconds")

        # Check if the objective function was decresased
        b_success = (X_min > result.fun) and (trust_check(result.x) >= 0)
        #b_success = False
        # If additional optimization was successful, return result, otherwise return x_search
        if b_success:
            print("BayesianOracle>> additional optimization successful")
            print("BayesianOracle>> minimum of %.5f at" % result.fun)
            print(result.x)
            m, s, u = self.predict_with_unc(np.array([result.x]).T)
            print("bayesianoracle>> mean: %.5f," %m + " explained std: %.5f," %s + " unexplained std: %.5f" %u)
            #print(self.predict(np.array([result.x]).T, bool_weights=True))
            return result.x
        else:
            print("BayesianOracle>> additional optimization failed")
            print("BayesianOracle>> minimum of %.5f at" % X_min)
            print(x_search)
            m, s, u = self.predict_with_unc(np.array([x_search]).T)
            print("bayesianoracle>> mean: %.5f," %m + " explained std: %.5f," %s + " unexplained std: %.5f" %u)
            #print(self.predict(np.array([x_search]).T, bool_weights=True))
            return x_search
