import numpy as np
import gptools
import scipy.stats
# For deep copy
import copy
# Distribution priors

from . import priors
from . import misc

class GaussianProcess(object):
    """ Base class for holding an extended GaussianProcess object """
    def __init__(self, ndim,
                 max_der_order=2,
                 constraints=None,
                 amplitude_prior=None,
                 scale_priors=None,
                 noisy=False,
                 kernel_type='SquaredExponential',
                 verbose=True):
        # Set dimension size, constraints, and maximum derivative order
        self.ndim = ndim
        self.constraints = constraints
        self.max_der_order = max_der_order

        self.verbose=verbose

        # Check that the constraints dimensions is consistent
        if (not (constraints is None)) and (len(constraints) != ndim):
            raise Exception("GaussianProcess: constraints inconsistent with input dimension")

        # Determine the default priors
        if amplitude_prior is None:
            amplitude_prior = priors.UniformPrior([0, 1e2])
        if scale_priors is None:
            scale_priors = [priors.LogNormalPrior(0, 1) for _ in range(self.ndim)]
        
        # Set priors
        self.amplitude_prior = amplitude_prior
        self.scale_priors = scale_priors

        # Set kernel type 
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
            #noise_pX = priors.LogNormalPrior(0, 1)
            #noise_pG = priors.LogNormalPrior(0, 1)
            #noise_pH = priors.LogNormalPrior(0, 1)

            noise_pX = priors.ExponentialPrior(1)
            noise_pG = priors.ExponentialPrior(1)
            noise_pH = priors.ExponentialPrior(1)

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
        y_err : list of errors, the element types of y_err must match that of y """

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
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ll = self.gp_.optimize_hyperparameters(random_starts=n_starts)
            return ll

    def fit(self, n_starts):
        """Fit the estimator with the stored data
        """
        
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
                
        # Update fit status
        self.is_fit = True
        return self.gp_.ll

    def predict(self, X, der_order=0):
         """ Performs a standard prediciton using the underlying Gaussian Process
         
         Parameters
         ----------
         X : array-like, shape = (n_points, ndim])
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

         for i in range(1,int(np.ceil(n_points/batch_sz))):
             # Calculate the underlying prediction
             batch_mean, batch_std = self.gp_.predict(X[(i-1)*batch_sz:np.min([n_points, i*batch_sz]),:], n=0, return_mean=True, return_std=True)
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
        
