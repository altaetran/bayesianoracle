import numpy as np
import gptools
import scipy.stats
import scipy.special
import scipy.misc
import scipy.optimize
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

    def discounted_mean(self, X, kappa=1):
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

class QuadraticBMAProcess(object):
    def __init__(self, 
                 ndim,
                 verbose=True,
                 hessian_distances=False,
                 zero_errors=False):

        self.ndim = ndim
        self.verbose = verbose

        # Initialize models list and weightPriors list 
        self.quadratic_models = []
        self.lambda_priors = []
        
        self.lambda_samples = []
        # Default kernel range
        self.kernel_range = 1.00;

        # Prior on the lambda
        self.lambda_alpha = 3.0
        self.lambda_beta = 1.0

        # Set prior on precision
        self.precision_alpha = 1.0 # Use 1 or 2
        self.precision_beta = 1.0

        # Number of lambda samples
        self.n_samples = 1000

        # HessianDistances
        self.hessian_distances = hessian_distances
        self.zero_errors = zero_errors

    def set_kernel_range(self, kernel_range):
        """ 
        Setter for the kernel range

        kernel_range : desired kernnel_range. Default is 1, but can be changed
        """
        self.kernel_range = kernel_range

    def set_precision_beta(self, precision_beta):
        """
        Setter for the precision beta value. 
        Note, does nto regenerate new lambda samples
        
        precision_beta : desired beta value for the precision gamma distribution
        """
        self.precision_beta = precision_beta
        
    def get_X_stack(self):
        """ 
        Get the X stack for the X data (n x n_models) matrix
        """
        return np.vstack([a for (y, A, b, d, a, Hchol) in self.quadratic_models]).T
        
    def get_y_stack(self):
        """
        returns n dimensional vector 
        """
        return np.hstack([y for (y, A, b, d, a, Hchol) in self.quadratic_models])

    def generate_lambda_samples(self, n_samples=-1):
        """ 
        Generates and stores the lambda samples for estimating the 
        prior distribution. Each sample is stored in self.lambda_samples
        with each column corresponding to one sample
        
        args

        n_samples : Number of samples to be calculated 
        """
        # Look for default number of samples
        if n_samples == -1:
            n_samples = self.n_samples
            
        self.lambda_samples = []

        # Samples of the possible lambdas. Array of nModels x n_samples
        for i in xrange(len(self.quadratic_models)):
            # Get n_samples samples of Lambda values for the i^th model
            self.lambda_samples.append(self.lambda_priors[i].rvs(size=n_samples))

        # Stack Lambdas to get the final lambdas
        self.lambda_samples = np.vstack(self.lambda_samples)
        
    def add_model(self, y, A, b, d, a, Hchol):
        """ 
        Quadratic model of the form 0.5 x'Ax + b'x + d anchored at point a
        Currently assumes that each model has no affect on its likelihood
        Hchol is inverse Hessian
        """

        # Append to the set of models
        self.quadratic_models.append([y, A, b, d, a, Hchol])
        self.lambda_priors.append(priors.GammaPrior(self.lambda_alpha, self.lambda_beta))

    def estimate_model_priors(self, x):
        """
        Estimates the model priors at the positions x using the 
        existing lambda samples
        
        x : n x p matrix of p positions
        """
        
        if self.hessian_distances:
            # Determine the squared distances. nModels x p matrix of distances
            distancesSq = [np.square(np.linalg.norm(Hchol.dot(x-a[:,None]), axis=0)) 
                           for (y, A, b, d, a, Hchol) in self.quadratic_models] 
            distancesSq = np.vstack(distancesSq)
        else:
            # Determine the squared distances. nModels x p matrix of distances
            distancesSq = [np.square(np.linalg.norm(x-a[:,None], axis=0)) 
                           for (y, A, b, d, a, Hchol) in self.quadratic_models] 
            distancesSq = np.vstack(distancesSq)

        # Keep track of the sum of the model priors across the lambda samples
        model_prior_sum = np.zeros((len(self.quadratic_models), x.shape[1]))

        n_samples = self.lambda_samples.shape[1]

        for i in range(n_samples):
            lambda_sample = self.lambda_samples[:, i]
            # Get the energies corresponding to this lambda sample
            energies = np.multiply(lambda_sample[:, np.newaxis], distancesSq)
            # shift the energies to avoid small division (make min energy is zero)
            energies = energies - np.amin(energies, axis=0)[np.newaxis, :]

            # Get the unnormalized probabilities for this lambda
            unnorm_prior_lambda = np.exp(-energies)
            # Get column sums and renormalize the probabilities for this lambda
            col_sums = np.sum(unnorm_prior_lambda, axis=0)
            prior_lambda = np.divide(unnorm_prior_lambda, col_sums[np.newaxis, :])
            
            # Add to the model prior sum for this lambda
            model_prior_sum += prior_lambda

        # Average across samples
        model_priors = model_prior_sum/n_samples

        return model_priors

    def estimate_log_model_priors(self, x):
        """
        Estimates the model priors at the positions x using the 
        existing lambda samples
        
        x : n x p matrix of p positions
        """
        
        if self.hessian_distances:
            # Determine the squared distances. nModels x p matrix of distances
            distancesSq = [np.square(np.linalg.norm(Hchol.dot(x-a[:,None]), axis=0)) 
                           for (y, A, b, d, a, Hchol) in self.quadratic_models] 
            distancesSq = np.vstack(distancesSq)
        else:
            # Determine the squared distances. nModels x p matrix of distances
            distancesSq = [np.square(np.linalg.norm(x-a[:,None], axis=0)) 
                           for (y, A, b, d, a, Hchol) in self.quadratic_models] 
            distancesSq = np.vstack(distancesSq)

        # Keep track of the sum of the model priors across the lambda samples
        model_prior_avg = np.zeros((len(self.quadratic_models), x.shape[1]))

        n_samples = self.lambda_samples.shape[1]

        for i in range(n_samples):
            lambda_sample = self.lambda_samples[:, i]
            # Get the energies corresponding to this lambda sample and normalize them
            energies = -np.multiply(lambda_sample[:, np.newaxis], distancesSq)
            log_normalization = scipy.misc.logsumexp(energies, axis=0)
            normalized_energies = energies - log_normalization[None,:]

            prior_lambda = np.exp(normalized_energies)
            
            # Add to the model prior sum for this lambda
            model_prior_avg += prior_lambda/n_samples

        return np.log(model_prior_avg)

    def calc_kernel_weights(self, x):
        """ 
        Calculates the kernel weights at the position x

        x : n x p matrix containing the location at which the wights are desired.
        returns : relevance weights (n_models x p matrix) of weights
        """
        def kernel_func(z):
            """ Operates on a single distance """
            if (z > self.kernel_range):
                return 0.0
            else:

                return 15.0/16.0*np.square(1.0-(z/self.kernel_range))/np.sqrt(self.kernel_range**self.ndim)
            
        def kernel_func(z):
            #return np.exp(-0.5*np.square(z/(self.kernel_range)))/(np.sqrt(2*np.pi)*(self.kernel_range**self.ndim))
            return np.exp(-0.5*np.square(z/(self.kernel_range)))

        # Vectorize the kernel function
        kernel_func_vec = np.vectorize(kernel_func)

        if self.hessian_distances:
            # Calculate the relevence weights for each model, and then stack them vertically
            kernel_weights = np.vstack([kernel_func_vec(np.linalg.norm(Hchol.dot(x-a[:,None]), axis=0))
                                           for (y, A, b, d, a, Hchol) in self.quadratic_models])
        else:
            """
            print("test")
            for i in range(len(self.quadratic_models)):
                print(i)
                print(self.quadratic_models[i][4])
                print(x-self.quadratic_models[i][4])
                dists = np.linalg.norm(x-self.quadratic_models[i][4], axis=0)
                print(dists)
                print(kernel_func_vec(np.linalg.norm(x-self.quadratic_models[i][4], axis=0)))
                kernel_weights = np.zeros(x.shape[1])
                print(x.shape)
                for j in range(x.shape[1]):
                    kernel_weights[j] = kernel_func(dists[j])
                print(kernel_weights)
            """
            # Calculate the relevence weights for each model, and then stack them vertically
            kernel_weights = np.vstack([kernel_func_vec(np.linalg.norm(x-a[:,None], axis=0))
                                           for (y, A, b, d, a, Hchol) in self.quadratic_models])

        return kernel_weights

    def calc_relevance_weights(self, x):
        """ 
        Calculates the kernel weights at the position x

        x : n x p matrix containing the location at which the wights are desired.
        returns : relevance weights (n_models x p matrix) of weights
        """
        
        # Get the kernel weights of the positions in x
        x_kernel_weights = self.calc_kernel_weights(x)
        
        """
        # Get the kernel weights of the model positions
        x_model = np.hstack([np.array([a]).T for (y, A, b, d, a, Hchol) in self.quadratic_models])
        x_model_kernel_weights = self.calc_kernel_weights(x_model)
        # Calculate the kernel densities at each of the nModel points
        model_N_eff = np.sum(x_model_kernel_weights, axis=0)
        relevance_weights = x_kernel_weights / model_N_eff[:, None]
        """
        # TEST TES TEST. Currently returns the kernel weights
        return x_kernel_weights
        #return relevance_weights

    def estimate_model_weights(self, x, return_errors=False, return_likelihoods=False):
        """
        predicts the model weights

        x : n x p matrix containing p locations at which weights are desired
        returns : (nModels x p) matrix of model weights
        """
        
        # Num models
        n_models = len(self.quadratic_models)
        n_positions = x.shape[1]

        # Get the relevence weights (nModels x p)
        relevance_weights = self.calc_relevance_weights(x)

        # Get the quadratic predictions at each position
        def Q(x):
            return np.vstack([(0.5*(A.dot(x)).T.dot(x) + b.T.dot(x) + d)
                    for (y, A, b, d, a, Hchol) in self.quadratic_models])
        # gets a n_models x n_models matrix with quadratic of model i on observation x_j
        model_means_at_obs = np.hstack([Q(a) for (y, A, b, d, a, Hchol) in self.quadratic_models])

        # Get the observations made at each model
        y_obs = np.hstack([y for (y, A, b, d, a, Hchol) in self.quadratic_models])
        
        # Get the J matrix
        J_mat = model_means_at_obs - y_obs[None,:]
        # Get the J Matrix elementwise square
        J_mat_sq = np.square(J_mat)

        # Calculate the effective number of samples
        N_eff = np.sum(relevance_weights, axis=0)

        # Create the n_models x n_positions error matrix
        # element i,j -> error of model i at j^th position
        errors = np.vstack([np.hstack([np.dot(relevance_weights[:,j],J_mat_sq[i,:])
                        for j in range(n_positions)]) for i in range(n_models)])

        # SET ERRORS TO ZERO
        if self.zero_errors:
            errors = errors*0

        # Calculate marginal likelihoods
        marginal_likelihoods = np.power(1+errors/(2*self.precision_beta), -(self.precision_alpha+N_eff/2.0))

        # Get the log model priors
        log_model_priors = self.estimate_log_model_priors(x)

        # Calculate log marignal likelihoods
        log_marginal_likelihoods = np.multiply(-(self.precision_alpha+N_eff/2.0),
                                                np.log(1+errors/(2*self.precision_beta)))

        log_unnorm_weights = np.add(log_marginal_likelihoods, log_model_priors)

        log_normalization = scipy.misc.logsumexp(log_unnorm_weights, axis=0)
        
        model_weights = np.exp(log_unnorm_weights - log_normalization[None,:])

        if return_errors:
            return model_weights, errors, N_eff
        elif return_likelihoods:
            return model_weights, errors, N_eff, np.exp(log_marginal_likelihoods)
        else:
            return model_weights

    def model_predictions(self, X):
        """
        For each quadratic model, predict for a stack of X (n x p) points
        returns : stack of (m x p) mean predictions. Each row corresponds
                  to a specific model's predictions
        """
        # Number of requested positions
        dim = X.shape[0]
        n_positions = X.shape[1]
        
        # Get model predictions
        def Q(x):
            return np.vstack([(0.5*(A.dot(x)).T.dot(x) + b.T.dot(x) + d)
                              for (y, A, b, d, a, Hchol) in self.quadratic_models])

        # gets a n_models x n_positions matrix with quadratic of model i on observation x_j
        model_means = np.hstack([Q(X[:,j]) for j in range(n_positions)])
        
        return model_means

    def predict_with_unc(self, X):
        """
        predict for a stack of points. Returns mean prediction, and mean disagreement
        and uncertainty
        X       : stack of (n x p) points
        returns : stack of (3 x p) predictions. First row is means. Seccond
                  row is the mean disagreement. Third row is the estimated
                  function uncertainty
        """
        # Minimum N_eff to prevent dividing by zero
        soft_N_eff = 10e-8

        model_weights, errors, N_eff = self.estimate_model_weights(X, return_errors=True)

        model_means = self.model_predictions(X)
        # multiply by model weights to get prediction
        
        # Get prediction means over the models
        bma_mean = np.sum(np.multiply(model_weights, model_means), axis = 0)

        # Get the expected disagreement over the models
        disagreement = np.sum(np.multiply(model_weights, np.square(model_means)), axis = 0) - np.square(bma_mean)
        disagreement[disagreement<0.0] = 0.0
        bma_disagreement = np.sqrt(disagreement)
        
        # Calculate the uncertainty of each model
        prefactor = np.divide(2*(self.precision_alpha+N_eff), 2*(self.precision_alpha+N_eff)-2+soft_N_eff)
        divfactor = (self.precision_alpha+0.5*N_eff)+soft_N_eff
        postfactor = (self.precision_beta+0.5*errors) / divfactor[None,:]
        model_unc = postfactor * prefactor[None,:]
        bma_unc = np.sum(np.multiply(model_weights, model_unc), axis = 0)
        bma_sqrtunc = np.sqrt(bma_unc)
        return np.vstack([bma_mean, bma_disagreement, bma_sqrtunc])

    def pdf(self, X, y):
        """ 
        Computes the pdf of observing the values of y at the locations X

        args
        ------------------------------
        X : (n x p) numpy array of p data points
        y : p dimensional numpy vector of function values at the points p
        returns : p dimensional numpy vector of probabilities of the given
                  observations
        """
        
        # Get model weights, and errors
        model_weights, errors, N_eff = self.estimate_model_weights(X, return_errors=True)

        # Get the model means (nModel x p)
        model_means = self.model_predictions(X)
        
        # Get the t distribution scale parameter (sigma^2)
        divfactor = (self.precision_alpha+0.5*N_eff)
        postfactor = (self.precision_beta+0.5*errors) / divfactor[None,:]
        
        # Sigma  for the student t posterior distributions are postfactor square root
        sigma = np.sqrt(postfactor)
        
        # Get nu (1 x p) vector
        nu = 2*(self.precision_alpha+N_eff)

        # Get z scores  (nModels x p)
        z = (y[None, :] - model_means) / sigma
        
        # Model probabilities (nModels x p)
        #print model_means
        #print z
        #print sigma

        #print nu
        #print(model_weights)
        model_probs = scipy.stats.t.pdf(z, nu[None, :])
        #print(model_probs)
        # Calculate the bma probability
        
        bma_probs = np.sum(np.multiply(model_weights, model_probs), axis = 0)

        return bma_probs

    def loglikelihood_data(self, X, y):
        """
        Calculates the likelihood of observing the values y at the locations X
        
        X : (n x p) array of locations
        y : p dimensional vector witth corresponding observations 

        returns : loglikelihood of the input set
        """
        return np.sum(np.log(self.pdf(X, y)))

    def loglikelihood(self, kernel_range, regularization=True):
        loglikelihood = self.loglikelihood_data(self.get_X_stack(), self.get_y_stack())
        if regularization:
            kernel_alpha = 1
            kernel_scale = 4.0
            loglikelihood = loglikelihood + scipy.stats.invgamma.logpdf(kernel_range, kernel_alpha, scale=kernel_scale)

        #predictions = bma.predict_with_unc(X)
        #cv_error = np.sum(np.square(predictions[2,:])+np.square(predictions[1,:]))
        #predictions = bma.predict_with_unc(X)
        #cv_error = np.sum(np.abs(np.square(predictions[0,:]-y)-np.square(predictions[1,:])))

        return loglikelihood

    def find_kernel_MAP_estimate(self):

        def minus_loglikelihood_log(log_kernel_range):
            return -self.loglikelihood(np.exp(log_kernel_range))

        # Bracket for the solution space and find
        bracket = [-3, 4]
        maxiter = 30
        result = scipy.optimize.minimize_scalar(minus_loglikelihood_log, 
                                                bracket=bracket,
                                                options={'maxiter':maxiter})

        # Extract the new kernel range, and set it
        new_kernel_range = np.exp(result.x)
        self.set_kernel_range(new_kernel_range)

        if self.verbose:
            print("bayesianoracle>> optimizing hyperparameters")
            print("bayesianoracle>> new hyperparameters:")
            print("bayesianoracle>> new ll         : "+str(-result.fun))
            print("bayesianoracle>> kernel range   : "+str(new_kernel_range))

    def predict(self, X, bool_weights=False):
        """
        predict for a stack of points X (n x p)
        returns : stack of (2 x p) predictions. First row is means. Second
                  row is the mean disangreement.
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

def der_to_quad(x0, f, g, H):
    """ Converts derivative information to a quadratic form of
    the form 0.5*x'*A*x + b'*x + d
    x0 : n x 1 location of observation (n dimensional vector)
    f : value of the function at the observation (scalar)
    g : value of the gradient at the observation (n dimensional vector)
    H : value of the Hessian at the observation (n x n) matrix"""
    if (len(x0)==1):
        A = H
        b = np.array(g-A.dot(x0))
        d = f - (b.T.dot(x0) + 0.5*(A.dot(x0).T.dot(x0)))
    else:
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
                 verbose=True):

        # Set dimension size, constraints, and maximum derivative order
        self.ndim = ndim
        self.verbose=verbose

        # Create initial data segment
        self.data_init = []

        # Intialize data segment
        self.data = copy.deepcopy(self.data_init)

        # Hessian Distances? (Only for convex problems)
        self.hessian_distances = False

        # Initialize BMAProcess
        self.bma = QuadraticBMAProcess(ndim=ndim, 
                                       hessian_distances=self.hessian_distances)

    def add_observation(self, x, f, g, H):
        """ Add an observation at x with uspecified error
        x : location of the observation n dimensional vector
        f : value of the objective at the the observed point (value, not array)
        g : value of the gradient at the observed point (n x 1)
        H : value of the Hessian at the observed point (n x n) """
        
        # Find the quadratic model for this observation and add to list of models
        [A, b, d] = der_to_quad(x, f, g, H)
        if self.hessian_distances:
            Hchol = np.linalg.inv(np.linalg.cholesky(H))
        else:
            Hchol = 0*A
        
        # Add to the data list
        self.data.append((x, f, g, H, b, d, Hchol))

        # Add the observation to the bma
        self.bma.add_model(f, A, b, d, x, Hchol)

        # Regenerate lambda samples
        self.bma.generate_lambda_samples()
        
    def get_n_models(self):
        return len(self.bma.quadratic_models)

    def get_X_stack(self):
        """
        returns an (n x nModels) matrix of the nModels observation locations 
        """
        return self.bma.get_X_stack()

    def predict(self, X, bool_weights=False):
        return self.bma.predict(X, bool_weights)

    def predict_with_unc(self, X):
        return self.bma.predict_with_unc(X)

    def set_kernel_range(self, kernel_range):
        self.bma.set_kernel_range(kernel_range)

    def calculate_discounted_mean(self, X, kappa=0.1):
        """ Calculates the discounted mean = mean - kappa * std at the values 
        in X
        X : (n x p) matrix of p proposition points
        kappa : std multiplier """
        
        # Get the predictions
        predictions = self.bma.predict_with_unc(X)
        # Get means and unexplained standard deviation
        means = predictions[0, :]
        unexp_std = predictions[2, :]

        # Protect the -Inf
        unexp_std[unexp_std == np.inf] = 10e40

        # Return the discounted means
        return means - kappa*unexp_std

    def create_validation_bma(self, idx):
        """
        Creates a bma testing object using the indices in idx for training.
        Repeated sampling is permitted
        
        idx : tuple of indices to be added to the bma object
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
        self.bma.find_kernel_MAP_estimate()
