import numpy as np
import scipy
from . import process_objects
from . import misc

class SequentialOptimizer(process_objects.EnrichedGaussianProcess):
    def __init__(self, 
                 ndim,
                 max_der_order=2,
                 constraints=None,
                 amplitude_prior=None,
                 scale_priors=None,
                 noisy=True,
                 kernel_type='SquaredExponential',
                 verbose=True):
        super(SequentialOptimizer, self).__init__(ndim,
                                                  max_der_order=max_der_order,
                                                  constraints=constraints,
                                                  amplitude_prior=amplitude_prior,
                                                  scale_priors=scale_priors,
                                                  noisy=noisy,
                                                  kernel_type=kernel_type,
                                                  verbose=verbose)

    def locate_acquisition_point(self, x0=None, trust_radius=1.5):
        """ Suggest a new ponit, X, that minimizes the expected improvement

        Parameters
        ----------
        x0 : Initial starting point for EI maximization. Setting x0 to None
             defaults the starting point to the last most iteration
        tr : Trust radius for the EI maximization. Defaults to 0.5. All solutions
             proposed must be within tr distance from x0 

        Output
        -------
        X_guess : shape = (n_features,) """

        xi = .01
        order = 0

        if self.verbose:
            print("BayesianOracle>> locating acqusition point... ")
            misc.tic()
            
        X = self.data[0]['X']
        # Use default of last iteration if x0 is None type
        if x0 is None:
            x0 = X[X.shape[0]-1,:]

        # Seed the trust region and find the best point
        nseed = 2000*self.ndim

        # Generate random lengths in [0,trust_radius]
        U = scipy.random.random(size=(nseed, 1))
        lengths = trust_radius*(U**(1.0/self.ndim))
        # Get uniformly distributed directions
        directions = scipy.random.randn(nseed, self.ndim)
        row_norms = np.sum(directions**2,axis=1)**(1./2)

        # Normalize the directions, then multiply by the lengths
        row_mult = (row_norms / lengths.T).T
        X_search = (directions / row_mult) + x0
        
        # Maximize expected improvement over search points
        ei = -self.calculate_EI(X_search, xi, order)
        x_search = X_search[np.argmin(ei)]

        def minus_ei(x):
            return -float(self.calculate_EI(np.array([x]), xi, order))

        result = scipy.optimize.minimize(minus_ei, x_search,
                                         bounds=self.constraints,
                                         options={'maxiter':1000, 'gtol':1e-5},
                                         method='TNC')

        if self.verbose:
            print("BayesianOracle>> found in " + str(misc.toc()) + " seconds")
            
        return x_search

class SequentialLeastSquaresOptimizer(process_objects.MultiIndependentTaskGaussianProcess):
    def __init__(self, 
                 ndim,
                 neq,
                 max_der_order=2,
                 constraints=None,
                 amplitude_prior=None,
                 scale_priors=None,
                 noisy=False,
                 kernel_type='Matern52',
                 verbose=True):

        super(SequentialLeastSquaresOptimizer, self).__init__(ndim,
                                                              neq,
                                                              max_der_order=2,
                                                              noisy=noisy,
                                                              constraints=constraints,
                                                              kernel_type=kernel_type,
                                                              verbose=verbose)

    def locate_acquisition_point(self, x0, trust_radius=1.5):
        """ Suggest a new ponit, X, that minimizes the expected improvement

        Parameters
        ----------
        x0 : Initial starting point for EI maximization. Setting x0 to None
             defaults the starting point to the last most iteration
        tr : Trust radius for the EI maximization. Defaults to 0.5. All solutions
             proposed must be within tr distance from x0 

        Output
        -------
        X_guess : shape = (n_features,) """
        
        kappa = -0.5

        if self.verbose:
            print("BayesianOracle>> locating acqusition point... ")
            misc.tic()

        # Seed the trust region and find the best point
        nseed = 2000*self.ndim

        # Generate random lengths in [0,trust_radius]
        U = scipy.random.random(size=(nseed, 1))
        lengths = trust_radius*(U**(1.0/self.ndim))
        # Get uniformly distributed directions
        dirnections = scipy.random.randn(nseed, self.ndim)
        row_norms = np.sum(directions**2,axis=1)**(1./2)

        # Normalize the directions, then multiply by the lengths
        row_mult = (row_norms / lengths.T).T
        X_search = (directions / row_mult) + x0
        
        # Maximize expected improvement over search points
        ei = self.discounted_mean(X_search, kappa)
        x_search = X_search[np.argmin(ei)]

        def ei(x):
            return float(self.discounted_mean(np.array([x]), kappa))

        result = scipy.optimize.minimize(ei, x_search,
                                         bounds=self.constraints,
                                         options={'maxiter':1000, 'gtol':1e-5},
                                         method='TNC')

        if self.verbose:
            print("LOL")
            print("BayesianOracle>> found in " + str(misc.toc()) + " seconds")
            print(result)

        return result.x
        return x_search

class QuadraticBMAOptimizer(process_objects.EnrichedQuadraticBMAProcess):
    def __init__(self, 
                 ndim,
                 verbose=True):
        super(QuadraticBMAOptimizer, self).__init__(ndim=ndim,
                                                    verbose=verbose)
        
        self.kappa_explore = 100
        self.kappa_detail = -0.1
        self.trust_explore = 5.0
        self.trust_detail = 1.0
        
        self.iteration = 0
        # Number of iterations under the detail regime
        self.detail_run_count = 0
        # Number of iterations with detail points "nearby" above which an
        # exploration step will be taken
        self.near_num_thresh = 3
        # Threshold for what is "near"
        self.near_thresh = self.trust_detail/10

    def locate_next_point(self):
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
        X = np.hstack([x for (x, f, g, H) in self.data[-self.detail_run_count:]]).T
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
            return self.locate_detail_point()
        else:
            # reset detail run count
            self.detail_run_count = 0
            self.iteration +=1
            if self.verbose:
                print("bayesianoracle>> requesting exploration point")
            return self.locate_exploration_point()

    def locate_exploration_point(self):
        # Get the anchor point to be the point that minimizes observed f
        f_vals = [f for (x, f, g, H) in self.data]
        x_anchor = self.data[np.argmin(f_vals)][0]

        return self.locate_acquisition_point(x0 = x_anchor, trust_radius=self.trust_explore, kappa=self.kappa_explore)

    def locate_detail_point(self):
        return self.locate_acquisition_point(trust_radius=self.trust_detail, kappa=self.kappa_detail)

    def locate_acquisition_point(self, x0=None, trust_radius=10, kappa = 1.0):
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
            print("BayesianOracle>> locating acqusition point... ")
            misc.tic()
            
        # Use default of last iteration if x0 is None type
        if x0 is None:
            x0 = self.data[-1][0]
            
        # Convert to vector
        x0 = np.squeeze(np.asarray(x0))
        print(x0)

        # Seed the trust region and find the best point
        nseed = 100*self.ndim

        # Generate random lengths in [0,trust_radius]
        U = scipy.random.random(size=(nseed, 1))
        lengths = trust_radius*(U**(1.0/self.ndim))
        # Get uniformly distributed directions
        directions = scipy.random.randn(nseed, self.ndim)
        row_norms = np.sum(directions**2,axis=1)**(1./2)

        # Normalize the directions, then multiply by the lengths
        row_mult = (row_norms / lengths.T).T
        X_search = (directions / row_mult) + x0

        # Maximize expected improvement over search points
        discounted_means = self.calculate_discounted_mean(X_search, kappa)
        x_search = X_search[np.argmin(discounted_means)]

        def dm(x):
            return float(self.calculate_discounted_mean(np.array([x]), kappa) )

        def trust_check(x):
            return trust_radius - np.linalg.norm(x-x0)

        constraints = {'type' : 'ineq', 'fun': trust_check}
        result = scipy.optimize.minimize(dm, x_search,
                                         constraints=constraints,
                                         options={'maxiter':10000, 'gtol':1e-5},
                                         method='L-BFGS-B')

        if self.verbose:
            print("BayesianOracle>> found in " + str(misc.toc()) + " seconds")
            
        # If additional optimization was successful, return result, otherwise return x_search
        if result.success:
            print("BayesianOracle>> additional optimization successful")
            return result.x
        else:
            print("BayesianOracle>> additional optimization failed")
            return x_search
