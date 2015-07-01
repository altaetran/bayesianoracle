import numpy as np
import scipy
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

        # Cross validate BMA
        self.cross_validate_kernel_range()

        # Use the defaults if trusts are zero
        if trust_detail == 0.0:
            trust_detail = self.bma.kernel_range
        if trust_explore == 0.0:
            trust_explore = self.bma.kernel_range

        # Set the trust radii
        self.trust_detail = trust_detail
        self.trust_explore = trust_explore
        
        # Set the near threshold 
        self.near_thresh = self.trust_detail/5.0

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

        # Locate current location of the minimum
        if self.verbose:
            print("bayesianoracle>> locating current estimate of the minimum location")
            self.locate_min_point()

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
        # Get the anchor point to be the point that minimizes observed f
        f_vals = [f for (x, f, g, H, b, d, Hchol) in self.data]
        x_anchor = self.data[np.argmin(f_vals)][0]

        return self.locate_acquisition_point(x0 = x_anchor, trust_radius=self.trust_explore, kappa=self.kappa_explore)

    def locate_detail_point(self):
        # Get the anchor point to be the point that minimizes observed f
        f_vals = [f for (x, f, g, H, b, d, Hchol) in self.data]
        x_anchor = self.data[np.argmin(f_vals)][0]

        return self.locate_acquisition_point(x0 = x_anchor, trust_radius=self.trust_detail, kappa=self.kappa_detail)
        #return self.locate_acquisition_point(trust_radius=self.trust_detail, kappa=self.kappa_detail)

    def locate_min_point(self):
        # Get the anchor point to be the point that minimizes observed f
        f_vals = [f for (x, f, g, H, b, d, Hchol) in self.data]
        x_anchor = self.data[np.argmin(f_vals)][0]

        return self.locate_acquisition_point(x0 = x_anchor, trust_radius=2.0, kappa=0.0)

    def locate_acquisition_point(self, x0=None, trust_radius=1.0, kappa=1.0):
        """ Suggest a new point, X, that minimizes the expected improvement

        Parameters
        ----------
        x0 : Initial starting point for discounted mean minimization. Setting x0 to None
             defaults the starting point to the last most iteration
             x0 must be n x 1 matrix
        tr : Trust radius for the EI minimization. Defaults to 10. All solutions
             proposed must be within trust_radius distance from x0 

        Output
        -------
        X_guess : shape = (n_features,) """

        if self.verbose:
            misc.tic()
            
        # Use default of last iteration if x0 is None type
        if x0 is None:
            x0 = self.data[-1][0]
            
        # Set kernel range of BMA to be half of the trust radius
        #self.set_kernel_range(trust_radius/20.0)

        # Convert to vector
        x0 = np.squeeze(np.asarray(x0))

        # Seed the trust region and find the best point
        nseed = 10000

        # Generate random lengths in [0,trust_radius]
        U = scipy.random.random(size=(nseed, 1))
        lengths = trust_radius*(U**(1.0/self.ndim))
        # Get uniformly distributed directions
        directions = scipy.random.randn(nseed, self.ndim)
        row_norms = np.sum(directions**2,axis=1)**(1./2)

        # Normalize the directions, then multiply by the lengths
        row_mult = (row_norms / lengths.T).T
        X_search = (directions / row_mult) + x0
        
        # Finally transpose to get a n x nseed matrix
        X_search = X_search.T

        # Maximize expected improvement over search points
        discounted_means = self.calculate_discounted_mean(X_search, kappa)
        j_min = np.argmin(discounted_means)
        x_search = X_search[:, j_min]
        X_min = discounted_means[j_min]

        def trust_check(x):
            return trust_radius - np.linalg.norm(x-x0)

        def dm(x):
            # Get the vectorized form into the correct n x 1 form
            return self.calculate_discounted_mean(np.array([x]).T, kappa)

        
        constraints = {'type' : 'ineq', 'fun': trust_check}
        result = scipy.optimize.minimize(dm, x_search,
                                         constraints=constraints,
                                         #jac=True,
                                         options={'maxiter':50},
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
