import numpy as np
import gptools
import scipy
# For deep copy
import copy
# Distribution priors
from . import priors
from . import misc
# Optimizer objects

class EnrichedGaussianProcess(object):
    """ Base class for holding an extended GaussianProcess object """
    def __init__(self, ndim,
                 max_der_order=2,
                 constraints=None,
                 amplitude_prior=None,
                 scale_priors=None,
                 noisy=False,
                 kernel_type='SquaredExponential',
                 verbose=True):
        # Check that the constraints dimensions is consistent
        if (not (constraints is None)) and (len(constraints) != ndim):
            raise Exception("GaussianProcess: constraints inconsistent with input dimension")
 
        # Default cosntraints
        if constraints is None:
            constraints = [(-1e8, 1e8) for _ in range(ndim)]
        # Default priors
        if amplitude_prior is None:
            #amplitude_prior = priors.UniformPrior([0, 1e2])
            #amplitude_prior = priors.ExponentialPrior(100)
            amplitude_prior = priors.LogNormalPrior(0,1)
        if scale_priors is None:
            scale_priors = [priors.LogNormalPrior(0, 1) for _ in range(ndim)]
        

        # Set dimension size, constraints, and maximum derivative order
        self.ndim = ndim
        self.max_der_order = max_der_order
        self.constraints = constraints
        self.verbose=verbose
        self.amplitude_prior = amplitude_prior
        self.scale_priors = scale_priors
        self.kernel_type = kernel_type
        self.noisy = noisy

        # initialize kernel
        self.init_kernels()

        # Create initial data segment
        self.data_init = []
        for k in range(self.max_der_order+1):
            self.data_init.append({'X':None, 'q':[], 'q_err':[]})

        # Intialize data segment
        self.data = copy.deepcopy(self.data_init)

        # declare fit status
        self.is_fit = False

    def init_kernels(self):
        """ Initializes the kernel functions for the Gaussian Process, and 
        initializes the ac tual gaussian process """

        if self.kernel_type == 'SquaredExponential':
            kernel = gptools.SquaredExponentialKernel(self.ndim,
                                                      hyperprior=[self.amplitude_prior,] + self.scale_priors)
        elif self.kernel_type == 'Matern52':
            kernel = gptools.Matern52Kernel(self.ndim,
                                            hyperprior=[self.amplitude_prior,] + self.scale_priors)
        else:
            raise Exception("Requested Kernel not implemented")

        
        if self.noisy:
            # If requested, set noise priors and create a diagonal noise kernel
            noise_pX = priors.LogNormalPrior(0, 1)
            noise_pG = priors.LogNormalPrior(0, 1)
            noise_pH = priors.LogNormalPrior(0, 1)

            # Exponential priors allow the noise to be zero, but also large if necessary
            #noise_pX = priors.ExponentialPrior(1)
            #noise_pG = priors.ExponentialPrior(1)
            #noise_pH = priors.ExponentialPrior(1)

            # Set up diagonal noise kernel for the value
            noise_kernel = gptools.DiagonalNoiseKernel(self.ndim,
                                                       n=0,
                                                       initial_noise=1e-3,
                                                       fixed_noise=False, 
                                                       noise_bound=(1e-3, np.inf),
                                                       hyperprior=[noise_pX])
            if self.max_der_order > 0:
                # If needed, set up gradient noise kernel
                grad_noise_kernel = gptools.DiagonalNoiseKernel(self.ndim,
                                                                n=1,
                                                                initial_noise=1e-3,
                                                                fixed_noise=False, 
                                                                noise_bound=(1e-3, np.inf),
                                                                hyperprior=[noise_pG])
                noise_kernel = noise_kernel.__add__(grad_noise_kernel)
            if self.max_der_order > 1:
                # If needed, set up hessian noise kernel
                hess_noise_kernel = gptools.DiagonalNoiseKernel(self.ndim,
                                                                n=2,
                                                                initial_noise=1e-3,
                                                                fixed_noise=False, 
                                                                noise_bound=(1e-3, np.inf),
                                                                hyperprior=[noise_pH])
                noise_kernel = noise_kernel.__add__(hess_noise_kernel)
        else:
            noise_kernel = None

        self.gp_ = gptools.GaussianProcess(kernel, noise_k=noise_kernel)

    def add_data(self, X, y, y_err=None, der_order=0):
        """

        Parameters
        ----------
        X     : array-like, shape = [n_samples, n_features]
                Training vector, where n_samples is the number of samples and
                n_features is the number of features.

        y     : list of objects (func observations, gradients, hessians, etc)
        y_err : list of errors, the element types of y_err must match that of y 
        der_order : derivative order of the input data """

        # Use default error of zero
        if y_err is None:
            y_err = [0] * len(y)

        if X.shape[0] != len(y):
            raise Exception('X and y have different number of points')

        # Store X and y and y_err
        if self.data[der_order]['X'] is None and self.data[der_order]['q'] == []:
            self.data[der_order]['X'] = X
        else:
            self.data[der_order]['X'] = np.vstack((self.data[der_order]['X'], X))

        self.data[der_order]['q'] = self.data[der_order]['q'] + y
        self.data[der_order]['q_err'] = self.data[der_order]['q_err'] + y_err

    def clear_gp_data(self):
        """ Clears the data from the Gaussian Process, without touching the 
        fitted parameters """

        self.gp_.y = np.array([], dtype=float)
        self.gp_.X = None
        self.gp_.n = None
        self.gp_.err_y = scipy.array([], dtype=float)
        self.gp_.K_up_to_date = False
 
    def clear_data(self):
        """ Reinitializes the data segment """

        self.data = copy.deepcopy(self.data_init)
     
    def optimize_hyperparameters(self, n_starts):
        """ Optimize the hyperparameters. If the number number of starts is not
        sufficient to achieve a minimum log likelihood threshold, 
        it is increased until a max amount """

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
                
            n_starts_mult = 2
            n_starts_max = 50
            ll_min = -1000*self.ndim
            
            # Compute the first fit
            ll = self.gp_.optimize_hyperparameters(random_starts=n_starts)
            if n_starts == 0:
                n_starts = 1
            # Iterate until minimal log likelhood or max n_starts is achieved
            while(ll < ll_min) and n_starts < n_starts_max:
                n_starts *= n_starts_mult
                ll = self.gp_.optimize_hyperparameters(random_starts=n_starts)

            return ll

    def fit(self, n_starts):
        """Fit the estimator with the stored data """
        
        # Reinitialize gaussian process
        self.clear_gp_data()

        # Loop through data and and add it to the GP
        for m in range(self.max_der_order+1):
            X = self.data[m]['X']
            y = self.data[m]['q']
            y_err = self.data[m]['q_err']
            # Add data to GP if data is available
            if not (X is None):
                self.gp_.add_data_list(X, y, err_y=y_err, n=m)

        # Fit gaussian process with partial data
        if self.verbose:
            misc.tic()

        self.optimize_hyperparameters(n_starts)
        
        if self.verbose:
            print("BayesianOracle>> hyperparameter optimization finished in " + str(misc.toc()) + " seconds")
            print("BayesianOracle>> current log-likelihood: %.4f" % self.gp_.ll)
            print("BayesianOracle>> current hyperparameters:")
            print(self.gp_.k.params)
            if self.noisy:
                print("BayesianOracle>> current noise kernel hyperparameters:")
                print(self.gp_.noise_k.params)
                
        # Update fit status
        self.is_fit = True
        return self.gp_.ll

    def predict(self, X, der_order=0):
         """ Performs a standard prediciton using the underlying Gaussian Process
         
         Parameters
         ----------
         X : numpy array, shape = (n_points, ndim)
         where n_samples is the number of requested samples
         
         Output
         ------
         mean : numpy array, shape = (n_points,). Contains prediction means
         std  : numpy array, shape = (n_points,). Contains prediction standard deviations """

         if not self.is_fit:
             raise Exception("Gaussian Process was not fit before calling the predict method")

         # Batch process for speed. 
         batch_sz = 100
         n_points = X.shape[0]
         mean, std = self.gp_.predict(X[0:np.min([n_points, batch_sz]),:], n=0, return_mean=True, return_std=True)
         # !!! ISSUE: imaginary output of gptools.GaussianProcess.predict()
         mean = np.real(mean)
         std = np.real(std)

         for i in range(1,int(np.ceil(float(n_points)/batch_sz))):
             # Calculate the underlying prediction
             batch_mean, batch_std = self.gp_.predict(X[i*batch_sz:np.min([n_points, (i+1)*batch_sz]),:], n=0, return_mean=True, return_std=True)
             # Stack up results
             mean = np.real(np.hstack((mean, batch_mean)))
             std = np.real(np.hstack((std, batch_std)))
         return mean, std 

    def calculate_EI(self, X, xi=0.01, order=1):
        """ Calculates the generalized expected improvement

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Grid of candidate parameters at which to compute the expected
            improvement
        xi: minimum expected improvement
        order : determines which generalized expected improvement to calculate

        Returns
        -------
        ei : array shape=(n,)
            The expected improvement (EI) at each of the
            candidates

        References
        ----------
        .. [1] Jones, D R., M. Schonlau, and W. J. Welch. "Efficient global
           optimization of expensive black-box functions." J. Global Optim.
           13.4 (1998): 455-492.
           
           https://www.cs.ubc.ca/~hoos/Publ/HutEtAl09b.pdf
        """

        # Get the prediction mean, std
        mean, std = self.predict(X)

        # Calculate the z score
        if order == 1 or order == 2:
            u = (np.min(self.data[0]['q']) - mean - xi) / std
            ncdf = scipy.stats.norm.cdf(u)
            npdf = scipy.stats.norm.pdf(u)
    
            if order == 1:
                # Calculate the order 1 expected improvement
                res = np.real(std * (u * ncdf + npdf))
            elif order == 2:
                # Calculate the order 2 expected improvement
                res = np.real(std)**2*((u**2+1)*ncdf + u*npdf)
        elif order == 0:
            # Just calculate the mean
            res = -np.real(mean)
        else:
            raise NotImplementedError('expected improvement order must be 0, 1 or 2')
            res = None
        return res
        
class MultiIndependentTaskGaussianProcess(object):
    def __init__(self, 
                 ndim,
                 neq,
                 max_der_order=2,
                 constraints=None,
                 noisy=False,
                 kernel_type='SquaredExponential',
                 verbose=True):
        if constraints is None:
            constraints = [(0, 1) for _ in range(ndim)]

        self.ndim = ndim
        self.neq = neq
        self.constraints = constraints
        self.max_der_order = max_der_order
        self.noisy=noisy
        self.kernel_type = kernel_type
        self.verbose = verbose

        # Initialize Gaussian Processes
        self.multi_gp = []
        for i in range(self.neq):
            amplitude_prior = priors.UniformPrior([0,1e2])
            scale_priors = [priors.LogNormalPrior(0, 1) for i in range(self.ndim)]
            self.multi_gp.append(EnrichedGaussianProcess(self.ndim, 
                                                         max_der_order=self.max_der_order,
                                                         constraints=self.constraints,
                                                         amplitude_prior=amplitude_prior,
                                                         scale_priors=scale_priors,
                                                         kernel_type=self.kernel_type, 
                                                         noisy=self.noisy,
                                                         verbose=self.verbose))

    def add_residual_data(self, res_idx, X, y, y_err=None, order=0):
        """ Add data for a specific residual to the corresponding Gaussian Process 

        Parameters
        ----------
        res_idx : index of the residual to which the input data belongs 
        X     : array-like, shape = [n_samples, n_features]
                Training vector, where n_samples is the number of samples and
                n_features is the number of features.

        y     : list of objects (func observations, gradients, hessians, etc)
        y_err : list of errors, the element types of y_err must match that of y 
        der_order : derivative order of the input data """

        if not ((res_idx < self.neq) and (res_idx >= 0)):
            raise Exception("index  add_residual_data must be an integer between 0 and " + str(neq-1))

        self.multi_gp[res_idx].add_data(X, y, y_err, order)
        
    def clear_residual_data(self, res_idx):
        """ Clears all residual data """
        self.multi_gp[res_idx].clear_data()
        
    def fit(self, n_starts):
        """ Optimizes the hyerparameters of each Gaussian Process. """ 

        for res_idx in range(self.neq):
            ll = self.multi_gp[res_idx].fit(n_starts)
    
    def predict(self, X):
        """ Returns the sum of residual squares mean and standard deviation.

        Parameters
        ----------
        X : numpy array, shape = (n_points, ndim)
        where n_samples is the number of requested samples
        
        Output
        ------
        mean : numpy array, shape = (n_points,). Contains prediction means
        std  : numpy array, shape = (n_points,). Contains prediction standard deviations """

        ignore_mu_sigma_term = False
        
        mu = []
        Sigma_diag = []
        # Acquire mu and diagonal Sigma elements for the cross covariations
        for res_idx in range(self.neq):
            mean, std = self.multi_gp[res_idx].gp_.predict(X)
            mu.append(mean.tolist())
            Sigma_diag.append(std.tolist())
        # Now, each point gets one column, rather than one row
        mu = np.array(mu)
        Sigma_diag = np.array(Sigma_diag)
        # Get the sum of squares along each of the residuals
        mu_sum_sq = np.apply_along_axis(np.linalg.norm, 0, mu)**2
        Sigma_sum_sq = np.apply_along_axis(np.linalg.norm, 0, Sigma_diag)**2
        # Add the terms in place to get the mean.
        mean = mu_sum_sq
        if not ignore_mu_sigma_term:
            mean += Sigma_sum_sq

        # Get higher order matrix multiplications for the quadratic std
        Sigma_Sigma = np.einsum('ij,ji->i', Sigma_diag.T, Sigma_diag)
        mu_Sigma_mu = np.einsum('ij,ji->i', mu.T, Sigma_diag*mu)
        # Add up the results in place
        sigma = Sigma_Sigma
        sigma *= 2
        mu_Sigma_mu *= 4
        sigma += mu_Sigma_mu
        # Get the square root
        sigma **= 0.5
        return mean, sigma

    def discounted_mean(self, X, kappa=1.0):
        """ Calculates the expected improvement in the Chi Square variable
        past a specific threshold 

        Parameters
        ----------

        X : numpy array, shape = (n_points, ndim)
        where n_samples is the number of requested samples
        kappa : the discount factor. 

        Output
        ------
        discounted mean : numpy array, shape = (n_points,). Contains the discounted means
            for the points in X """

        mean, std = self.predict(X)
        std *= kappa
        result = mean
        result += std
        return result

class QuadraticModel(object):
    def __init__(self, y, A, b, d, a):
        """
        Initializer class for a quadratic model of the form

        0.5*x^T A x + b^T x + d
        
        args:
        -----
        y : (scalar) observed valued of y at the observation location
        A : (n x n matrix) observed Hessian of y at the observation location
        b : (n dimensional vector) in quad equation. Related to gradient
        d : (scalar) in quad equation. Related to y
        a : (n dimensional vector) observation location.
        """
        self.y = y
        self.A = A
        self.b = b
        self.d = d
        self.a = a

    def predict(self, x):
        """
        Predicts the result of the deterministic model at the single input
        location
        """
        return 0.5*(self.A.dot(x)).T.dot(x) + self.b.T.dot(x) + self.d
        
    def get_y(self):
        return self.y
    
    def get_a(self):
        return self.a

class QuadraticBMAProcess(object):
    def __init__(self, 
                 ndim,
                 verbose=True,
                 hessian_distances=False,
                 zero_errors=False,
                 kernel_type='Gaussian',
                 bool_variable_kernel=True,
                 bool_sample_lambdas=False):

        self.ndim = ndim
        self.verbose = verbose

        self.quadratic_models = []  # Init models list
        self.kernel_type = kernel_type
        self.kernel_range = 1.00;  # Default kernel range

        self.bool_variable_kernel = bool_variable_kernel

        self.__init_priors()  # Initialize priors
        self.__init_kernel_func()  # Initialize kernel function
        self.lambda_samples = []  # (n_models x n_samples matrix) of lam samples
        self.n_samples = 1000  # Num lambda samples each gen

        # Additional two testing options
        self.hessian_distances = hessian_distances
        self.zero_errors = zero_errors
        self.bool_sample_lambdas = bool_sample_lambdas

    def __init_priors(self):
        # Prior on the lambda (product of independent gammas)
        self.lambda_alpha = 3.0
        self.lambda_beta = 1.0
        self.lambda_prior = priors.GammaPrior(self.lambda_alpha, self.lambda_beta)

        # Set prior on precision
        self.precision_alpha = 1.00  # Use 1 or 2 for differing results
        self.precision_beta = 100.0

        # Set the kernel prior to a default gamma
        self.kernel_prior = priors.GammaPrior()

    def __init_kernel_func(self):
        """
        Initializes the default kernel function to be used. 
        """
        if self.kernel_type == 'Gaussian':
            def gaussian_kernel(dist, kernel_range):
                return np.exp(-0.5*np.square(dist/kernel_range))

            self.set_kernel_func(gaussian_kernel)            
        elif kernel_prior_type == 'triweight':
            def triweight_kernel(dist, kernel_range):
                if (dist > self.kernel_range):
                    return 0.0
                else:
                    return (np.square(1.0-np.square(dist/kernel_range)))**3
            
            self.set_kernel_func(triweight_kernel)            
        else:
            assert False,'kernel__type must be a \'Gaussian\', or \'Triweight\''

    def set_kernel_func(self, kernel_func):
        """
        Setter for the kernel function. Must be a radially symmetric kernel.
        The actual function is vectorized by numpy before being used later.
        
        args:
        -----
        kernel_func : (function(dist(scalar), kernel_range(scalar)))
                      where dist is the distance between the points
                      and where kernel_range is the kernel_range of the
                      kernel
        """
        self.kernel_func = np.vectorize(kernel_func)

    def set_precision_prior_params(self, precision_alpha, precision_beta):
        """
        Setter for the precision parameters. The precision prior is a gamma 
        distribution with alpha = precision_alpha, and beta = precision_beta
        
        args:
        -----
        precision_alpha : (scalar) alpha value
        precision_beta  : (scalar) beta value
        """
        self.precision_alpha = precision_alpha
        self.precision_beta = precision_beta

    def set_kernel_range(self, kernel_range):
        """ 
        Setter for the kernel range. The kernel range is the effective
        width of the kernel used in the bayesian model averaging process

        args:
        -----
        kernel_range : (scalar) desired kernnel_range.
        """
        self.kernel_range = kernel_range
        if len(self.quadratic_models) > 0:
            self.generate_renorm_factors()

    def set_kernel_prior(self, kernel_prior):
        """
        Setter for the kernel prior. The kernel prior is the distribution
        used to set a prior on the kernel width which is ultimately learned
        via MAP estimation.

        args:
        -----
        kernel_prior : (Prior object) from priors.py
        """
        self.kernel_prior = kernel_prior

    def set_precision_beta(self, precision_beta):
        """
        Setter for the precision beta value. 
        Note, does nto regenerate new lambda samples

        args:
        -----
        precision_beta : desired beta value for the precision gamma distribution
        """
        self.precision_beta = precision_beta
        
    def get_X_stack(self):
        """ 
        Getter for the X stack for the positional data 

        returns: 
        --------
        (n x n_models matrix) of the n_models data points.
        """
        return np.vstack([qm.get_a() for qm in self.quadratic_models]).T
        
    def get_y_stack(self):
        """
        Getter for the y stack of function observations

        returns:
        --------
        (n dimensional vector) of the n_models function observations
        """
        return np.hstack([qm.get_y() for qm in self.quadratic_models])

    def generate_lambda_samples(self):
        """ 
        Generates and stores the lambda samples for estimating the 
        prior distribution. Each sample is stored in self.lambda_samples
        with each column corresponding to one sample        
        """
        n_samples = self.n_samples
        n_models = len(self.quadratic_models)

        self.lambda_samples = []  # Reset samples

        if self.bool_sample_lambdas:
            # Generate samples of the possible lambdas. 
            for i in xrange(n_models):
                # Get n_samples samples of Lambda values for the i^th model
                self.lambda_samples.append(self.lambda_prior.rvs(size=n_samples))

            # self.lambda_samples is a (n_models x n_samples matrix)
            self.lambda_samples = np.vstack(self.lambda_samples)
        else:
            self.lambda_samples.append(1.0)
            self.lambda_samples = np.array([self.lambda_samples]).T

    def add_model(self, y, A, b, d, a, Hchol):
        """ 
        Quadratic model of the form 0.5 x'Ax + b'x + d anchored at point a
        Currently assumes that each model has no affect on its likelihood
        Hchol is inverse Hessian
        """

        # Append model data to the set of models
        qm = QuadraticModel(y, A, b, d, a)
        self.quadratic_models.append(qm)
        self.set_kernel_range(self.kernel_range)

    def __get_model_distances(self, X):
        """
        Gets the distances between the input positions and all of the
        model positions
        
        args:
        -----
        X : (n x p matrix) of p positions
        
        returns:
        --------
        (n_models x p) matrix of distances between the n_model observation
        locations and the p input locations
        """
        distances_sq = None

        if self.hessian_distances:
            # Determine the squared distances using Hessian distances
            distances_sq = [np.square(np.linalg.norm(Hchol.dot(X-a[:,None]), axis=0)) 
                           for (y, A, b, d, a, Hchol) in self.quadratic_models] 
            distances_sq = np.vstack(distances_sq)
        else:
            # Determine the squared distances without Hessian distances
            distances_sq = [np.square(np.linalg.norm(X-qm.get_a()[:,None], axis=0)) 
                           for qm in self.quadratic_models] 
            distances_sq = np.vstack(distances_sq)

        return distances_sq

    def estimate_model_priors(self, X):
        """
        Estimates the model priors at the positions x using the 
        existing lambda samples

        args:
        -----
        X : (n x p matrix) of p positions
        
        returns:
        --------
        (n_models x p matrix) of the n_models model prior values
        at each of the p positions

        """
        p = X.shape[1]
        n_models = len(self.quadratic_models)
        distances_sq = self.__get_model_distances(X)  # Model distances
        
        # Keep track of the sum of the model priors across the lambda samples
        model_prior_avg = np.zeros((n_models, p))

        n_samples = self.lambda_samples.shape[1] # Get num lambda samples

        # Iterate through each lambda sample and calculate its contribution
        # to the resulting prior
        for i in xrange(n_samples):
            lambda_sample = self.lambda_samples[:, i] # Specific lambda vector
            # Get energies corresponding to this lambda vec and normalize
            energies = -np.multiply(lambda_sample[:, np.newaxis], distances_sq)
            log_normalization = scipy.misc.logsumexp(energies, axis=0)
            normalized_energies = energies - log_normalization[None,:]

            prior_lambda = np.exp(normalized_energies)
            
            # Add to the model prior sum for this lambda
            model_prior_avg += prior_lambda/n_samples

        model_priors = model_prior_avg

        return model_priors

    def estimate_log_model_priors(self, X):
        """
        Estimates the model priors at the positions X using the 
        existing lambda samples

        args:
        -----
        X : (n x p matrix) of p positions

        returns:
        --------
        (n_models x p matrix) of the n_models log model prior values
        at each of the p positions
        """
        return np.log(self.estimate_model_priors(X))

    def calc_variable_kernel_weights(self, X, renorm):
        """ 
        Calculates the kernel weights at the p positions in X

        args:
        -----
        X : n x p matrix containing the p locations at which the weights 
            are desired.
    
        returns: 
        --------
        (n_models x p matrix) of the kernel weights
        """
        if self.hessian_distances:
            # Calculate the relevence weights for each model, and then stack them vertically
            kernel_weights = [self.kernel_func(np.linalg.norm(Hchol.dot(X-a[:,None]), axis=0), 
                                               self.kernel_range) 
                              for (y, A, b, d, a, Hchol) in self.quadratic_models]
            kernel_weights = np.vstack(kernel_weights)  # Reformat
        else:
            # Calculate the relevence weights for each model, and then stack them vertically
            kernel_weights = [self.kernel_func(np.linalg.norm(X-self.quadratic_models[i].get_a()[:,None], axis=0),
                                               self.kernel_range*renorm[i])
                              for i in xrange(len(self.quadratic_models))]
            kernel_weights = np.vstack(kernel_weights)  # Reformat
            kernel_weights / np.sum(np.power(1/renorm,self.ndim))
        return kernel_weights

    def calc_kernel_weights(self, X):
        """ 
        Calculates the kernel weights at the p positions in X

        args:
        -----
        X : n x p matrix containing the p locations at which the weights 
            are desired.
    
        returns: 
        --------
        (n_models x p matrix) of the kernel weights
        """
        if self.hessian_distances:
            # Calculate the relevence weights for each model, and then stack them vertically
            kernel_weights = [self.kernel_func(np.linalg.norm(Hchol.dot(X-a[:,None]), axis=0), 
                                               self.kernel_range) 
                              for (y, A, b, d, a, Hchol) in self.quadratic_models]
            kernel_weights = np.vstack(kernel_weights) # Reformat
        else:
            # Calculate the relevence weights for each model, and then stack them vertically
            kernel_weights = [self.kernel_func(np.linalg.norm(X-qm.get_a()[:,None], axis=0), 
                                               self.kernel_range)
                              for qm in self.quadratic_models]
            kernel_weights = np.vstack(kernel_weights) # Reformat

        return kernel_weights

    def generate_renorm_factors(self):
        """
        Generates the renorm factors for the variable kernel
        method. Only geenrates if the variable kernel method
        is active
        """
        if not self.bool_variable_kernel:
            return
        
        n_models = len(self.quadratic_models)
        X_obs = self.get_X_stack()
        init_kernel_weights = self.calc_kernel_weights(X_obs)
        # Get kernel sums 
        init_kernel_sum = np.sum(init_kernel_weights,axis=0)
        # Get log densities
        log_init_density = np.log(init_kernel_sum)-np.log(n_models)-self.ndim*np.log(self.kernel_range)
        # Get densities
        init_density = np.exp(log_init_density)
        # Sum to get the geometric means
        log_geom = np.sum(log_init_density)/n_models
        geom = np.exp(log_geom)
        # Generate renorm factors
        renorm_factors = np.power(init_density/geom, -1.0)
        self.renorm_factors = renorm_factors  # Save renorm factors
#        print('han')
#        print(renorm_factors)


    def calc_relevance_weights(self, X):
        """ 
        Calculates the relevance weights at the position x. Currently
        this is simply the kernel weights.

        args:
        -----
        X : (n x p matrix) containing the p locations at which the weights 
            are desired.
    
        returns: 
        --------
        (n_models x p matrix) of the relevance weights
        """
        if self.bool_variable_kernel:
            return self.calc_variable_kernel_weights(X, self.renorm_factors)
        else:
            return self.calc_kernel_weights(X)

    def estimate_model_weights(self, X, return_errors=False, return_likelihoods=False):
        """
        Predicts the model weights

        args:
        -----
        X                  : (n x p matrix) containing p locations at which 
                             weights are desired
        return_errors      : return errors and effective sample size in 
                             addition to model weights?
        return_likelihoods : return errors and effective sample size
                             and likelihoods in addition to model
                             weights?
        returns:
        --------
        (n_models x p) matrix of model weights at the p locations

        (optional) (n_models x p) matrix of the n_models errors at the p 
        locations

        (optional) (1 x p) matrix of the effective sample sizes at the p
        locations

        (optional) (1 x p) matrix of the likelihoods of the bayesian
        model averaging process at the p locations
        """
        n_models = len(self.quadratic_models)
        p = X.shape[1]  # Number of positions to check

        # Get the relevence weights (nModels x p)
        relevance_weights = self.calc_relevance_weights(X)

        def Q(x):
            """
            Array of each model's prediction on x
            """
            return np.vstack([qm.predict(x) for qm in self.quadratic_models])

        # gets a n_models x n_models matrix with quadratic of model i on observation x_j
        model_means_at_obs = np.hstack([Q(qm.get_a()) for qm in self.quadratic_models])

        # Get the observations made at each model
        y_obs = np.hstack([qm.get_y() for qm in self.quadratic_models])
        
        # Get the J matrix
        J_mat = model_means_at_obs - y_obs[None,:]
        # Get the J Matrix elementwise square
        J_mat_sq = np.square(J_mat)

        # Calculate the effective number of samples
        N_eff = np.sum(relevance_weights, axis=0)

        # Create the n_models x p error matrix
        # element i,j -> error of model i at j^th position
        errors = np.vstack([np.hstack([np.dot(relevance_weights[:,j],J_mat_sq[i,:])
                                       for j in xrange(p)]) for i in xrange(n_models)])

        # SET ERRORS TO ZERO
        if self.zero_errors:
            errors = errors*0

        # Calculate marginal likelihoods
        marginal_likelihoods = np.power(1+errors/(2*self.precision_beta), -(self.precision_alpha+N_eff/2.0))

        # Get the log model priors
        log_model_priors = self.estimate_log_model_priors(X)

        # Calculate log marignal likelihoods
        log_marginal_likelihoods = np.multiply(-(self.precision_alpha+N_eff/2.0),
                                                np.log(1+errors/(2*self.precision_beta)))

        log_unnorm_weights = np.add(log_marginal_likelihoods, log_model_priors)

        log_normalization = scipy.misc.logsumexp(log_unnorm_weights, axis=0)
        
        model_weights = np.exp(log_unnorm_weights - log_normalization[None,:])

        # Conditional return values
        if return_errors:
            return model_weights, errors, N_eff
        if return_likelihoods:
            return model_weights, errors, N_eff, np.exp(log_marginal_likelihoods)
        # Default return values
        return model_weights

    def pdf_data_given_x_kernel_range(x, kernel_range):
        return

    def model_predictions(self, X):
        """
        For each quadratic model, calculate the prediction of the model
        on the input data points
        
        args:
        -----
        X : (n x p matrix) of the p locations for which predictions
            are desired.

        returns:
        --------
        (n_models x p matrix) of model predictions. Each row corresponds 
        to a specific model's predictions
        """
        n = X.shape[0]
        p = X.shape[1]
        
        # Get model predictions
        def Q(x):
            return np.vstack([qm.predict(x) for qm in self.quadratic_models])

        # Gets a n_models x p matrix with quadratic of model i on observation x_j
        model_means = np.hstack([Q(X[:,j]) for j in xrange(p)])
        
        return model_means

    def predict(self, X, bool_weights=False):
        """
        Predictions without calculating explained variance
        
        args:
        -----
        X : (n x p matrix) of the p locations for which predictions
        are desired.

        returns:
        --------
        (2 x p matrix) of predictions. 1st row is the means. 2nd row
        is the unexplaiend standard deviation.
        """
        
        # Get model weights
        model_weights = self.estimate_model_weights(X)

        model_means = self.model_predictions(X) 
        # multiply by model weights to get prediction

        bma_mean = np.sum(np.multiply(model_weights, model_means), axis = 0)

        disagreement = np.sum(np.multiply(model_weights, np.square(model_means)), axis = 0) - np.square(bma_mean)
        disagreement[disagreement<0.0] = 0.0
        bma_disagreement = np.sqrt(disagreement)

        if bool_weights:
            print("BayesianOracle>> model weights")
            print(model_weights)
        return np.vstack([bma_mean, bma_disagreement])                

    def predict_with_unc(self, X):
        """
        Full BMA prediction for a set of input locations. Returns mean 
        prediction, unexplained standard deviation, and explained
        standard devaition
       
        args:
        -----
        X       : (n x p matrix) of p locations
        
        returns:
        --------
        (4 x p matrix) of predictions. 1st row is means. 2nd row is 
        unexplained standard deviation. 3rd row is explained standard
        deviation. 4th row is the effective sample size.
        """
        # Minimum N_eff to prevent dividing by zero
        soft_N_eff = 10e-8

        model_weights, errors, N_eff = self.estimate_model_weights(X, return_errors=True)

        N_eff = N_eff+soft_N_eff  # Update soft minimum for sample size
        model_means = self.model_predictions(X)  # Individual model predictions
        
        # Get prediction means over the models via weighted average.
        bma_mean = np.sum(np.multiply(model_weights, model_means), axis = 0)

        # Get the expected disagreement over the models
        disagreement = np.sum(np.multiply(model_weights, np.square(model_means)), axis = 0) - np.square(bma_mean)
        disagreement[disagreement<0.0] = 0.0
        unexp_std = np.sqrt(disagreement)
        
        # Calculate the uncertainty of each model
        alpha_n = self.precision_alpha+0.5*N_eff
        prefactor = np.divide(2*alpha_n, 2*alpha_n-2)
        divfactor = alpha_n
        postfactor = (self.precision_beta+0.5*errors) / divfactor[None,:]
        model_unc = postfactor * prefactor[None,:]
        bma_unc = np.sum(np.multiply(model_weights, model_unc), axis = 0)
        exp_std = np.sqrt(bma_unc)

        exp_std[exp_std == np.inf] = 10e40  # Protect against inf
        unexp_std[unexp_std == np.inf] = 10e40  # Protect against inf

        
        return np.vstack([bma_mean, unexp_std, exp_std, N_eff])

    def pdf(self, X, y):
        """ 
        Computes the pdf of observing the values of y at the locations X

        args:
        -----
        X : (n x p matrix) of p data points
        y : (p dimsional vector) of function values at each of the p points

        returns:
        --------
        p dimensional numpy vector of probabilities of the given observations
        """        
        # Get model weights, and errors
        model_weights, errors, N_eff = self.estimate_model_weights(X, return_errors=True)

        # Get the model means (n_model x p)
        model_means = self.model_predictions(X)
        
        # Get the t distribution scale parameter (sigma^2)
        divfactor = (self.precision_alpha+0.5*N_eff)
        postfactor = (self.precision_beta+0.5*errors) / divfactor[None,:]
        
        # Sigma for the student t posterior distributions are postfactor square root
        sigma = np.sqrt(postfactor)

        # Get nu (1 x p) vector
        nu = 2*self.precision_alpha+N_eff

        # Get z scores  (nModels x p)
        z = (y[None, :] - model_means) / sigma
        
        model_probs = scipy.stats.t.pdf(z, nu[None, :])

        # Calculate the bma probability        
        bma_probs = np.sum(np.multiply(model_weights, model_probs), axis = 0)

        return bma_probs

    def log_naive_pdf(self, X):
        """ 
        Computes the log pdf of observing the values of y at the locations X
        WITHOUT any training data

        args:
        -----
        X : (n x p matrix) of p data points
        y : (p dimsional vector) of function values at each of the p points

        returns:
        --------
        p dimensional numpy vector of probabilities of the given observations
        """        
        p = X.shape[1]

        def Q(x):
            """
            Array of each model's prediction on x
            """
            return np.vstack([qm.predict(x) for qm in self.quadratic_models])

        # Get the n_models x p matrix of log model priors
        log_prior = self.estimate_log_model_prior(X)

        # gets a n_models x p matrix with quadratic of model i on observation x_j
        model_means_at_X = np.hstack([Q(X[i,:]) for i in xrange(p)])
        log_prob = np.log(1 + np.power(y[None, :] - model_means_at_X, 2)
                          / (2*self.precision_beta))
        log_prob *= -self.precision_alpha

        # Calculate probabiltiy dot product using log sum
        comb_log_prob = log_prob + log_prior
        
        # Get the log sum exp across the models (likelihood for each 
        # individual data point)
        indv_likelihood = scipy.misc.logsumexp(comb_log_prob, axis=0)

        # Get the likelihood across all the data
        log_likelihood = np.sum(indv_likelihood)

        return log_likelihood

    def estimate_skewness(self, kernel_range=-1):
        """
        Calculates the skewness of the data involved
        """
        if kernel_range != -1:
            old_kernel_range = self.kernel_range 
            self.set_kernel_range(kernel_range)
            bool_restore = True
        else:
            bool_restore = False

        predictions = self.predict_with_unc(self.get_X_stack())
        y = self.get_y_stack()

        sample_size = len(self.quadratic_models)
        
        # Get the true variance of the points
        var = np.square(predictions[1, :]) + np.square(predictions[2, :])

        # Get residuals
        resi = predictions[0, :] - y

        if sample_size != 1:
            # Get third moment
            numerator = np.sum(np.power(resi, 3))/sample_size
            denominator = np.power((np.sum(var)/(sample_size-1)), 3.0/2.0)
        
            skew = np.divide(numerator, denominator)

        else:
            skew = 0
        
        if bool_restore:
            self.kernel_range = old_kernel_range

        return skew

    def loglikelihood_data(self, X, y):
        """
        Calculates the loglikelihood of observing the values y at the 
        location X without the contribution of the prior.
        
        args:
        -----
        X : (n x p matrix) of the p locations
        y : p dimensional vector witth corresponding observations 

        returns: 
        --------
        (scalar) log likelihood of the input set
        """
        return np.sum(np.log(self.pdf(X, y)))

    def cv_loglikelihood(self, kernel_range):
        """
        Calculates the cross validated log likelihood of the data
        included in the current model
        
        returns:
        --------
        (scalar) cross validated log likelihood of the contained data
        """
        old_kernel_range = self.kernel_range  # Save old kernel range
        self.set_kernel_range(kernel_range)  # Set kernel range

        qms_cpy = copy.deepcopy(self.quadratic_models)  # Copy the quadratic models 
        n_models = len(qms_cpy)

        lams_cpy = copy.deepcopy(self.lambda_samples)  # Copy lambdas

        cv_ll = 0.0

        # Add regularization term
        cv_ll += self.kernel_prior.logpdf(kernel_range)

        #  If not enough observations for cross validation, return as is
        if n_models < 2:
            self.kernel_range = old_kernel_range  # Restore old kernel_range
            return cv_ll

        # Iterate through each of the data points in the model
        for i in xrange(len(qms_cpy)):
            X = np.array([qms_cpy[i].get_a()]).T  # Get point
            y = np.array([qms_cpy[i].get_y()])  # Get fun eval

            # Replace old quadratic_models with new version without this point
            self.quadratic_models = qms_cpy[:i]+qms_cpy[(i+1):]

            # Replace old lambdas with sliced lambda
            self.lambda_samples = np.vstack([lams_cpy[:i,:], lams_cpy[(i+1):,:]])
            # Get likelihood
            ll = np.log(self.pdf(X, y)) / n_models
            # print([kernel_range, i, ll])
            cv_ll += ll

        self.quadratic_models = qms_cpy  # Restore old models
        self.lambda_samples = lams_cpy  # Restore old lambda samples
        self.kernel_range = old_kernel_range  # Restore old kernel_range

        return cv_ll

    def loglikelihood(self, kernel_range, regularization=True, skew=True):
        """
        Calculates and returns the log likelihood of the data
        stored in the Bayesian model averaging process
        
        returns:
        --------
        (scalar) log likelihood of the data stored in the Bayesian
        model averaging process
        """
        old_kernel_range = self.kernel_range  # Save old kernel range
        self.set_kernel_range(kernel_range)  # Set new kernel range

        X = self.get_X_stack()
        y = self.get_y_stack()
        loglikelihood = self.loglikelihood_data(X, y)

        # Include kernel prior regularization if requested
        if regularization:
            loglikelihood += self.kernel_prior.logpdf(kernel_range)
            
        self.kernel_range = old_kernel_range  # Restore old kernel_range

        return loglikelihood

    def find_kernel_MAP_estimate(self):
        """
        Determines the MAP estimate of the kernel range using the
        kernel prior and the data. Solved using the Brent bracketing
        algorithm for scalar minimization
 
        CAUTION: Kernel range is assumed to be 
        between 0.01 and 100.

        CAUTION: Max number of iterations is 50. 
        """
        # We work with the log kernel range to get a wider range
        # of kernel rangse
        def minus_loglikelihood_log(log_kernel_range):
            return -self.loglikelihood(np.power(10.0,log_kernel_range))
            #return -self.cv_loglikelihood(np.exp(log_kernel_range))

        # Bracket for the solution space and find
        bracket_center = np.log10(self.kernel_prior.get_mode())
        bracket = [-2.0 + bracket_center, 2.0 + bracket_center]
        maxiter = 100
        result = scipy.optimize.minimize_scalar(minus_loglikelihood_log, 
                                                bracket=bracket,
                                                bounds=bracket,
                                                method='Bounded',
                                                options={'maxiter':maxiter})

        # Extract the new kernel range, and set it
        new_kernel_range = np.power(10.0,result.x)
        self.set_kernel_range(new_kernel_range)

        if self.verbose:
            print("BayesianOracle>> new hyperparameters:")
            print("BayesianOracle>> new ll         : "+str(-result.fun))
            print("BayesianOracle>> kernel range   : "+str(new_kernel_range))

    def find_kernel_min_skew(self):
        """ 
        DEPRICATED
        """
        def abs_skew(log_kernel_range):
            old_kernel_range = self.kernel_range
            self.kernel_range = np.exp(log_kernel_range)
            skew = self.estimate_skewness()
            self.kernel_range = old_kernel_range

            return np.abs(skew)

        # Bracket for the solution space and find
        bracketp = [-3, 4]
        maxiter = 30
        result = scipy.optimize.minimize_scalar(abs_skew,
                                                bracket=bracket,
                                                options={'maxiter':maxiter})

        # Extract the new kernel range, and set it
        new_kernel_range = np.exp(result.x)
        self.set_kernel_range(new_kernel_range)

        if self.verbose:
            print("BayesianOracle>> optimizing hyperparameters")
            print("BayesianOracle>> new hyperparameters:")
            print("BayesianOracle>> new ll         : "+str(-result.fun))
            print("BayesianOracle>> kernel range   : "+str(new_kernel_range))        
            print("")

def der_to_quad(x0, f, g, H):
    """ Converts derivative information to a quadratic form of
    the form 0.5*x'*A*x + b'*x + d
   
    args:
    -----
    x0 : (n dimensional vector) location of observation
    f  : (scalar) of the function evaluation at the location
    g  : (n dimensional vector) of the gradient at the location
    H  : (n x n matrix) of the Hessian at the location

    returns:
    [A, b, d] with
    A  : (n x n) matrix
    b  : (n dimensional vector)
    c  : (scalar)
    such that f(x)= 0.5 x^T A x + b^T x + d is the quadratic model
    corresponding to the derivative observations
    """
    A = H
    b = g - A.dot(x0)
    d = f - (b.T.dot(x0) + 0.5*(A.dot(x0).T.dot(x0)))
    return([A, b, d])

#bo.process_objects.der_to_quad(np.array([[1],[1]]), 10, np.array([[1],[0]]), np.array([[2, 0],[0, 10]]))
""" Returns
[array([[ 2,  0],
       [ 0, 10]]),
 array([[ -1],
       [-10]]),
 array([ 15.])]
"""

def centered_quad_to_quad(A, mean, f_mean):
    A = A
    b = -0.5*(A.dot(mean)+A.T.dot(mean))
    d = f_mean + 0.5*(A.dot(mean)).T.dot(mean)[0]
    return([A, b, d])

class EnrichedQuadraticBMAProcess(object):
    """ Base class for holding an extended QuadraticBMA object """
    def __init__(self, ndim,
                 verbose=True,
                 init_kernel_range=1.0,
                 kernel_type='Gaussian'):

        # Set dimension size, constraints, and maximum derivative order
        self.ndim = ndim
        self.verbose=verbose

        self.data_init = []  # Create initial data list
        self.data = []  # Create the data list

        # Hessian Distances? (Only for convex problems)
        self.hessian_distances = False

        # Initialize BMAProcess
        self.bma = QuadraticBMAProcess(ndim=ndim,
                                       verbose=self.verbose,
                                       hessian_distances=self.hessian_distances,
                                       kernel_type=kernel_type)

        self.set_kernel_range(init_kernel_range)  # Set init kernel_range
        
    def set_gamma_kernel_prior(self, mode, var):
        """ 
        Sets the kernel prior to a gamma function with mode, mode,
        and variance, var. Notice that the mode is used as apposed to
        mean or median, because in the absence of data, the MAP estimate
        will just be the mode.

        args:
        -----
        mode : (scalar) mode value of the gamma distribution to be used
        var  : (scalar) variance value of the gamma distribution to be used
        """
        kernel_prior = priors.GammaPrior()  # Create default GammaPrior
        kernel_prior.set_mode_var(mode, var)  # Set Gamma params using mode and var
        self.bma.set_kernel_prior(kernel_prior)  # Set bma's kernel prior

    def set_invgamma_kernel_prior(self, mode, var):
        """ 
        Sets the kernel prior to an inverse gamma function with mode, mode,
        and variance, var. Notice that the mode is used as apposed to
        mean or median, because in the absence of data, the MAP estimate
        will just be the mode.

        args:
        -----
        mode : (scalar) mode value of the gamma distribution to be used
        var  : (scalar) variance value of the gamma distribution to be used
        """
        kernel_prior = priors.InvGammaPrior()  # Create default GammaPrior
        kernel_prior.set_mode_var(mode, var)  # Set Gamma params using mode and var
        self.bma.set_kernel_prior(kernel_prior)  # Set bma's kernel prior

    def set_lognormal_kernel_prior(self, mode, var):
        """ 
        Sets the kernel prior to an inverse gamma function with mode, mode,
        and variance, var. Notice that the mode is used as apposed to
        mean or median, because in the absence of data, the MAP estimate
        will just be the mode.

        args:
        -----
        mode : (scalar) mode value of the gamma distribution to be used
        var  : (scalar) variance value of the gamma distribution to be used
        """
        kernel_prior = priors.LogNormalPrior()  # Create default GammaPrior
        kernel_prior.set_mode_var(mode, var)  # Set Gamma params using mode and var
        self.bma.set_kernel_prior(kernel_prior)  # Set bma's kernel prior

    def set_precision_prior_params(self, precision_alpha, precision_beta):
        """
        Setter for the precision parameters. The precision prior is a gamma 
        distribution with alpha = precision_alpha, and beta = precision_beta
        
        args:
        -----
        precision_alpha : (scalar) alpha value
        precision_beta  : (scalar) beta value
        """
        self.bma.set_precision_prior_params(precision_alpha, precision_beta)

    def add_observation(self, x, f, g, H):
        """ Add an observation at x with uspecified error

        args:
        -----
        x : (n dimensional vector) location of the observation
        f : (scalar) of the objective at the the observed point
        g : (n dimensional vector) of the gradient at the observed point
        H : (n x n matrix) of the Hessian at the observed point
        """
        # Convert the derivative information to the qudratic model
        [A, b, d] = der_to_quad(x, f, g, H)

        if self.hessian_distances:
            Hchol = np.linalg.inv(np.linalg.cholesky(H))  # Hchol if needed
        else:
            Hchol = None # Else use None type
        
        self.data.append((x, f, g, H, b, d, Hchol))  # Add to data list
        self.bma.add_model(f, A, b, d, x, Hchol)  # Add to bma
        self.bma.generate_lambda_samples()  # Generate new lambda samples
         
    def get_n_models(self):
        """
        returns:
        --------
        (scalar) number of models in the Bayesian model averaging process
        """
        return len(self.bma.quadratic_models)

    def get_X_stack(self):
        """
        returns:
        (n x n_models matrix) of n_models locations that have been previously
        stored in the model
        """
        return self.bma.get_X_stack()

    def predict(self, X, bool_weights=False):
        """
        Basic prediction.

        args:
        -----
        X       : (n x p matrix) of p locations
        
        returns:
        --------
        (2 x p matrix) of predictions. 1st row is means. 2nd row is 
        unexplained standard deviation. 
        """
        return self.bma.predict(X, bool_weights)

    def predict_with_unc(self, X):
        """
        Full prediction

        args:
        -----
        X       : (n x p matrix) of p locations
        
        returns:
        --------
        (3 x p matrix) of predictions. 1st row is means. 2nd row is 
        unexplained standard deviation. 3rd row is explained standard
        deviation.
        """
        return self.bma.predict_with_unc(X)

    def set_kernel_range(self, kernel_range):
        """
        Setter for the kernel range
        
        args:
        -----
        kernel_range : (scalar) new kernel range to be used for the 
                       Bayesian model averaging process
        """
        self.bma.set_kernel_range(kernel_range)

    def calculate_N_eff(self, X):
        """ 
        Calculates the effective sample size at the desired locations
        
        args:
        -----
        X : (n x p matrix) of p proposed locations at which
            sample sizes are desired

        returns:
        --------
        (1 x p matrix) of sample sizes
        """
        model_weights, errors, N_eff = self.bma.estimate_model_weights(X, return_errors=True)
        return N_eff

    def calculate_discounted_mean(self, X, kappa=0.1):
        """ 
        Calculates the discounted_mean = mean - kappa * unexplained_std
        + nu*n_eff
        at the p locations in X

        X     : (n x p matrix) of p proposition points at which discounted_means
                 are desired
        kappa : (scalar) std multiplier used in the discounted means calculation
        """
        # Get the predictions
        predictions = self.bma.predict(X)
        # Get means and unexplained standard deviation
        means = predictions[0, :]
        unexp_std = predictions[1, :]

        # Return the discounted means
        return means - kappa*unexp_std

    def create_validation_bma(self, idx):
        """
        Creates a bma testing object using the indices in idx for training.
        Repeated sampling is permitted
        
        args:
        -----
        idx : (integer tuple) of indices of which observations are to be
        added to the new bma object
        
        returns:
        --------
        QuadraticBMAProcess with the observations as dictated by idx
        """
        bma  = QuadraticBMAProcess(ndim=self.ndim, 
                                   hessian_distances=self.hessian_distances)
        
        # Add all of the data
        for i in idx:
            f = self.data[i][1]
            A = self.data[i][3]
            b = self.data[i][4]
            d = self.data[i][5]
            x = self.data[i][0]
            Hchol = self.data[i][6]

            # Add the data
            bma.add_model(f, A, b, d, x, Hchol)

        # generate lambda samples
        bma.generate_lambda_samples()

        return bma

    def optimize_hyperparameters(self):
        """
        Optimizes the kernel hyperparameter using MAP estimation
        """
        if self.verbose:
            misc.tic()
            print("BayesianOracle>> optimizing hyperparameters")

        self.bma.find_kernel_MAP_estimate()
        if self.verbose:
            print("BayesianOracle>> hyperparameter optimization finished in %.1f seconds" % misc.toc())
            print("")
