import numpy as np
import scipy
# For deep copy
import copy
# Distribution priors
from . import priors
from . import misc
# Optimizer objects

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

    def predict_batch(self, X):
        return np.array([self.predict(X[:,i]) for i in xrange(X.shape[1])])
        
    def get_y(self):
        return self.y
    
    def get_a(self):
        return self.a

    def get_min_location(self):
        return np.linalg.solve(self.A, -self.b)

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
    d  : (scalar)
    such that f(x)= 0.5 x^T A x + b^T x + d is the quadratic model
    corresponding to the derivative observations
    """
    A = H
    b = g - A.dot(x0)
    d = f - (b.T.dot(x0) + 0.5*(A.dot(x0).T.dot(x0)))
    return([A, b, d])


class QuadraticBMAProcess(object):
    def __init__(self, 
                 ndim,
                 verbose=True,
                 n_int=50,
                 hessian_distances=False,
                 zero_errors=False,
                 kernel_type='Gaussian',
                 bool_variable_kernel=False,
                 bool_sample_lambdas=False):

        self.ndim = ndim
        self.verbose = verbose

        self.quadratic_models = []  # Init models list
        self.kernel_type = kernel_type
        self.kernel_range = 1.00;  # Default kernel range

        self.bool_variable_kernel = bool_variable_kernel

        self.__init_priors()  # Initialize priors
        self.__init_kernel_func()  # Initialize kernel function

        self.n_int = n_int
        self.__init_quadrature_samples()  # Initialize samples for integration
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

        self.bias_lambda = 0.1

        # Set the kernel prior to a default gamma
        self.kernel_prior = priors.GammaPrior()

    def __init_kernel_func(self):
        """
        Initializes the default kernel function to be used. 
        """
        if self.kernel_type == 'Gaussian':
            def gaussian_kernel(dist, kernel_range):
                #return np.exp(-0.5*dist/kernel_range)
                return np.exp(-0.5*np.square(dist/kernel_range))

            self.set_kernel_func(gaussian_kernel)            
        elif self.kernel_type == 'triweight':
            def triweight_kernel(dist, kernel_range):
                if (dist > kernel_range):
                    return 0.0
                else:
                    #return np.exp(-0.5*dist/kernel_range)
                    return np.power(1.0-np.square(dist/kernel_range), 3.0)
            
            self.set_kernel_func(triweight_kernel)            
        else:
            assert False,'kernel__type must be a \'Gaussian\', or \'Triweight\''

    def __init_quadrature_samples(self):
        ### Only works for gamma prior on the kernel range
        n_int = self.n_int  # Number of roots for laguerre
        min_weight = 1.0e-20
        # Get the kernel ranges from generalized lageurre 
        # First get beta parma
        beta = self.kernel_prior.get_beta()
        alpha = self.kernel_prior.get_alpha()
 
        if n_int <= 1:
            roots, weights = scipy.special.la_roots(n_int, alpha-1)

            # Make the roots real
            roots = np.real(roots)

        else:
            roots = np.logspace(-3, 3, num=n_int)
            e = (alpha)*np.log(roots) - beta*roots
            weights = np.exp(e - np.amax(e))
            weights /= np.sum(weights)

        # pare off small weights
        ind = np.array(range(n_int))
        ind_rem = ind[weights<min_weight]
        lg_roots = np.delete(roots, ind_rem)
        lg_weights = np.delete(weights, ind_rem)

        print(lg_weights)

        self.lg_roots = lg_roots
        self.lg_weights = lg_weights
    
        kernel_ranges = self.lg_roots / beta

        if self.verbose:
            min_kernel_range = np.min(kernel_ranges)
            max_kernel_range = np.max(kernel_ranges)
            print("BayesianOracle>> Minimum kernel range considered: "+str(min_kernel_range))
            print("BayesianOracle>> Maximum kernel range considered: "+str(max_kernel_range))
            print("BayesianOracle>> Number of kernel ranges considered: "+str(len(lg_weights)))

        """
        self.roots = np.linspace(0.01, 10, 50) * beta
        self.weights = np.ones(50) / 50.0
        """

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

    def set_bias_prior_params(self, bias_lambda):
        """
        Setter for the bias parameters. The bias prior is a normal gamma
        related to the precision prior
        
        args:
        -----
        bias_lambda  : (scalar) lambda value
        """
        self.bias_lambda = bias_lambda

    def set_kernel_range(self, kernel_range):
        """ 
        Setter for the kernel range. The kernel range is the effective
        width of the kernel used in the bayesian model averaging process

        args:
        -----
        kernel_range : (scalar) desired kernnel_range.
        """
        self.kernel_range = kernel_range
        if self.get_n_models() > 0:
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
        self.__init_quadrature_samples()

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

    def get_n_models(self):
        return len(self.quadratic_models)

    def generate_lambda_samples(self):
        """ 
        Generates and stores the lambda samples for estimating the 
        prior distribution. Each sample is stored in self.lambda_samples
        with each column corresponding to one sample        
        """
        n_samples = self.n_samples
        n_models = self.get_n_models()

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

    def estimate_model_priors(self, X, kernel_range):
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
        # Get energies corresponding to this lambda vec and normalize
        distances_sq = self.__get_model_distances(X)  # Model distances
        energies = -distances_sq / (2. * kernel_range)
        log_normalization = scipy.misc.logsumexp(energies, axis=0)
        normalized_energies = energies - log_normalization[None,:]

        model_priors = np.exp(normalized_energies)
             
        """
        p = X.shape[1]
        n_models = self.get_n_models()
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
        """
        return model_priors

    def estimate_log_model_priors(self, X, kernel_range):
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
        soft_log = 1e-32
        return np.log(self.estimate_model_priors(X, kernel_range)+soft_log)

    def calc_variable_kernel_weights(self, X, renorm, kernel_range):
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
                                               kernel_range) 
                              for (y, A, b, d, a, Hchol) in self.quadratic_models]
            kernel_weights = np.vstack(kernel_weights)  # Reformat
        else:
            # Calculate the relevence weights for each model, and then stack them vertically
            kernel_weights = [self.kernel_func(np.linalg.norm(X-self.quadratic_models[i].get_a()[:,None], axis=0),
                                               kernel_range*renorm[i])
                              for i in xrange(self.get_n_models())]
            kernel_weights = np.vstack(kernel_weights)  # Reformat
            kernel_weights / np.sum(np.power(1/renorm,self.ndim))
        return kernel_weights

    def calc_kernel_weights(self, X, kernel_range=-1.0):
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
        # Check for default
        if kernel_range == -1.0:
            kernel_range = self.kernel_range

        if self.hessian_distances:
            # Calculate the relevence weights for each model, and then stack them vertically
            kernel_weights = [self.kernel_func(np.linalg.norm(Hchol.dot(X-a[:,None]), axis=0), 
                                               kernel_range) 
                              for (y, A, b, d, a, Hchol) in self.quadratic_models]
            kernel_weights = np.vstack(kernel_weights) # Reformat
        else:
            # Calculate the relevence weights for each model, and then stack them vertically
            kernel_weights = [self.kernel_func(np.linalg.norm(X-qm.get_a()[:,None], axis=0), 
                                               kernel_range)
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
        
        n_models = self.get_n_models()
        X_obs = self.get_X_stack()
        init_kernel_weights = self.calc_kernel_weights(X_obs, self.kernel_range)
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

    def calc_relevance_weights(self, X, kernel_range=-1.0):
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
        # Check for default kernel_range
        if kernel_range == -1.0:
            kernel_range = self.kernel_range

        if self.bool_variable_kernel:
            return self.calc_variable_kernel_weights(X, self.renorm_factors, kernel_range)
        else:
            return self.calc_kernel_weights(X, kernel_range)

    def estimate_model_weights(self, X, return_errors=False, return_likelihoods=False, kernel_range=-1.0):
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
        n_models = self.get_n_models()
        p = X.shape[1]  # Number of positions to check

        # Get the relevence weights (nModels x p)
        relevance_weights = self.calc_relevance_weights(X, kernel_range)

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

        soft_N_eff = 1e-8
        # calculate the residual mean
        resi_mean = np.dot(J_mat, relevance_weights) / (N_eff[np.newaxis,:] + soft_N_eff)

        # Create the n_models x p error matrix
        # element i,j -> error of model i at j^th position
        errors = np.vstack([np.hstack([np.dot(relevance_weights[:,j],J_mat_sq[i,:])
                                       for j in xrange(p)]) for i in xrange(n_models)])

        lam = self.bias_lambda
        lam_factor = - np.square(N_eff) / (lam + N_eff)
        eps = errors + lam_factor*np.square(resi_mean)

        # SET ERRORS TO ZERO
        if self.zero_errors:
            errors = errors*0

        # Calculate marginal likelihoods
        #marginal_likelihoods = np.power(1+errors/(2*self.precision_beta), -(self.precision_alpha+N_eff/2.0))

        # Get the log model priors
        log_model_priors = self.estimate_log_model_priors(X, kernel_range)

        # Calculate log marignal likelihoods
        log_marginal_likelihoods = np.multiply(-(self.precision_alpha+N_eff/2.0),
                                                np.log(1+eps/(2*self.precision_beta)))

        log_unnorm_weights = np.add(log_marginal_likelihoods, log_model_priors)

        log_normalization = scipy.misc.logsumexp(log_unnorm_weights, axis=0)
        
        model_weights = np.exp(log_unnorm_weights - log_normalization[None,:])

        # Conditional return values
        if return_errors:
            return model_weights, eps, N_eff, resi_mean
        if return_likelihoods:
            return model_weights, eps, N_eff, resi_mean, np.exp(log_marginal_likelihoods)
        # Default return values
        return model_weights

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

    def model_biased_predictions(self, X, kernel_range=-1):
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
        

        # Check for default kernel range
        if kernel_range == -1:
            kernel_range = self.kernel_range

        # Get model weights, and errors
        model_weights, eps, N_eff, resi_mean = self.estimate_model_weights(X, return_errors=True, kernel_range=kernel_range)

        # Get model predictions
        def Q(x):
            return np.vstack([qm.predict(x) for qm in self.quadratic_models])

        # Gets a n_models x p matrix with quadratic of model i on observation x_j
        model_means = np.hstack([Q(X[:,j]) for j in xrange(p)])

        # Correct for bias
        a = (N_eff / (self.bias_lambda + N_eff))[np.newaxis,:] * resi_mean
        model_means -= a
        
        return model_means

    def predict(self, X, bool_weights=False, kernel_range=-1.0):
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
        
        if kernel_range == -1.0:
            kernel_range = self.kernel_range

        # Get model weights
        model_weights, eps, N_eff, resi_mean = self.estimate_model_weights(X, kernel_range=kernel_range, return_errors=True)

        model_means = self.model_predictions(X) 

        # Correct for biases
        a = (N_eff / (self.bias_lambda + N_eff))[np.newaxis,:] * resi_mean
        model_means -= a        

        # multiply by model weights to get prediction
        bma_mean = np.sum(np.multiply(model_weights, model_means), axis = 0)

        disagreement = np.sum(np.multiply(model_weights, np.square(model_means)), axis = 0) - np.square(bma_mean)
        disagreement[disagreement<0.0] = 0.0
        bma_disagreement = np.sqrt(disagreement)

        if bool_weights:
            print("BayesianOracle>> model weights")
            print(model_weights)
        return np.vstack([bma_mean, bma_disagreement])                

    def predict_with_unc(self, X, kernel_range=-1.0):
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
        if kernel_range == -1.0:
            kernel_range = self.kernel_range

        # Minimum N_eff to prevent dividing by zero
        soft_N_eff = 10e-8

        model_weights, eps, N_eff, resi_mean = self.estimate_model_weights(X, return_errors=True, kernel_range=kernel_range)

        N_eff = N_eff+soft_N_eff  # Update soft minimum for sample size
        model_means = self.model_predictions(X)  # Individual model predictions

        # Correct for bias
        a = (N_eff / (self.bias_lambda + N_eff))[np.newaxis,:] * resi_mean
        model_means -= a        
        
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
        lamfactor = 1. + 1./(self.bias_lambda + N_eff)
        postfactor = (self.precision_beta+0.5*eps) / divfactor[None,:]
        model_unc = postfactor * lamfactor * prefactor[None,:]
        bma_unc = np.sum(np.multiply(model_weights, model_unc), axis = 0)
        exp_std = np.sqrt(bma_unc)

        exp_std[exp_std == np.inf] = 10e40  # Protect against inf
        unexp_std[unexp_std == np.inf] = 10e40  # Protect against inf

        return np.vstack([bma_mean, unexp_std, exp_std, N_eff])

    def pdf(self, X, y, kernel_range=-1, bool_return_model_pdfs=False, bool_dataless=False):
        """ 
        Computes the pdf of observing the values of y at the locations X

        args:
        -----
        X : (n x p matrix) of p data points
        y : (p dimsional vector) of function values at each of the p points

        bool_dataless : returns the prior pdfs if True
        returns:
        --------
        p dimensional numpy vector of probabilities of the given observations
        """        
        # Check for default kernel range
        if kernel_range == -1:
            kernel_range = self.kernel_range

        # Get model weights, and errors
        model_weights, eps, N_eff, resi_mean = self.estimate_model_weights(X, return_errors=True, kernel_range=kernel_range)

        if bool_dataless:
            n = len(self.quadratic_models)
            p = X.shape[1]
            eps = np.zeros([n, p])
            N_eff = np.zeros([p])
            resi_mean = np.zeros([n, p])

        # Get the model means (n_model x p)
        model_means = self.model_predictions(X)
        
        # Calculate bias correction
        a = (N_eff / (self.bias_lambda + N_eff))[np.newaxis,:] * resi_mean
        model_means -= a

        # Get the t distribution scale parameter (sigma^2)
        divfactor = self.precision_alpha+0.5*N_eff
        lamfactor = 1. + 1./(self.bias_lambda + N_eff)
        postfactor = (self.precision_beta+0.5*eps) * lamfactor[None,:] / divfactor[None,:]
        
        # Sigma for the student t posterior distributions are postfactor square root
        sigma = np.sqrt(postfactor)

        # Get nu (1 x p) vector
        nu = 2*self.precision_alpha+N_eff

        # Get z scores  (nModels x p)
        z = (y[None, :] - model_means) / sigma
        
        model_probs = scipy.stats.t.pdf(z, nu[None, :])

        # Calculate the bma probability        
        bma_probs = np.sum(np.multiply(model_weights, model_probs), axis = 0)

        if bool_return_model_pdfs:
            return bma_probs, model_probs, model_weights
        else:
            return bma_probs

    def get_KL(self, X, kernel_range=-1.):
        if kernel_range == -1.0:
            kernel_range = self.kernel_range

        # Minimum N_eff to prevent dividing by zero
        soft_p = 1e-8

        model_weights = self.estimate_model_weights(X, return_errors=False, kernel_range=kernel_range)

        H = -np.sum(np.multiply(model_weights, np.log(model_weights+soft_p)), axis=0)

        model_priors = self.estimate_model_priors(X, kernel_range=kernel_range)

        H_prior = -np.sum(np.multiply(model_weights, np.log(model_priors+soft_p)), axis=0)

        return H_prior - H

    def predict_bayesian(self, X, return_likelihoods=False, return_H=False):
        ### Only works for gamma prior on the kernel range

        p = X.shape[1]
        
        # Get the kernel ranges from generalized lageurre 
        # First get beta parma
        beta = self.kernel_prior.get_beta()
        alpha = self.kernel_prior.get_alpha()
                
        # rescale roots
        kernel_ranges = self.lg_roots / beta

        # (n_int x p matrix) of the different kernel ranges for each location
        naive_logs = []
        predictions = []
        means = []
        unexp_std = []
        exp_std = []
        n_eff = []
        if return_H:
            H = []
        log_priors = []

        for kernel_range in kernel_ranges:
            naive_logs.append(self.log_naive_local_pdf(X, kernel_range))
            # Get the prediction at this kernel range
            m, u, e, n = self.predict_with_unc(X, kernel_range=kernel_range)            
            means.append(m)
            unexp_std.append(u)
            exp_std.append(e)
            n_eff.append(n)
            if return_H:
                h = self.get_KL(X, kernel_range=kernel_range)
                H.append(h)
            # Get the prior probabilties of this kernel_range value
            log_priors.append(self.kernel_prior.logpdf(kernel_range))

        # Stack all the predictions and probabilities. Each is (n_int x p matrix)
        naive_logs = np.vstack(naive_logs)
        means = np.vstack(means)
        unexp_std = np.vstack(unexp_std)
        exp_std = np.vstack(exp_std)
        n_eff = np.vstack(n_eff)
        if return_H:
            H = np.vstack(H)
        log_priors = np.array(log_priors)

        # Note, prior values can be kept as a vector

        # Calculate the normalization integrals obtained by discretizing
        # the continuous probability distribution along kernel_range
        naive_likelihoods = np.exp(naive_logs)
        # unnormalized posterior (n_int x p matrix)
        #posterior_unnorm_log = naive_logs +  log_priors[:, None]
        #posterior_unnorm = np.exp(posterior_unnorm_log)

        # Get the normalization through integration. Remember other 
        # constant factors in the integral cancel out when doing prediction
        #normalization = self.weights.dot(naive_likelihoods)

        # Renormalized weights
        log_posterior = np.log(self.lg_weights[:, None]) + naive_logs # + log_priors[:, None]
        log_normalization = scipy.misc.logsumexp(log_posterior, axis=0)
        log_posterior -= log_normalization[None, :]
        
        def integrate(f_arr):
            # Edit f_arr
            f_arr_min = np.min(np.min(f_arr))-1.0
            if len(f_arr.shape) == 1:
                # Vector case
                logsum = scipy.misc.logsumexp(log_posterior, axis=0, b=f_arr[:, None]-f_arr_min)
                val = np.exp(logsum) + f_arr_min
                return val
            else:
                #matrix case
                logsum = scipy.misc.logsumexp(log_posterior, axis=0, b=f_arr-f_arr_min)
                val = np.exp(logsum) + f_arr_min
                return val

        
        bayesian_means = integrate(means)
        bayesian_unexp_std = integrate(unexp_std)
        bayesian_exp_std = integrate(exp_std)
        bayesian_n_eff = integrate(n_eff)
        if return_H:
            bayesian_H = integrate(H)
        avg_kernel_ranges = integrate(kernel_ranges)

        if return_H:
            return_val = np.vstack([bayesian_means, bayesian_unexp_std, bayesian_exp_std, bayesian_n_eff, bayesian_H])
        else:
            return_val = np.vstack([bayesian_means, bayesian_unexp_std, bayesian_exp_std, bayesian_n_eff])

        #normalized_likelihoods = np.divide(naive_likelihoods, normalization[None, :])
        #full_likelihoods = np.exp(np.log(normalized_likelihoods)+log_priors[:, None])
        #full_likelihoods = np.exp(log_posterior - np.log(self.weights[:, None]))
        #full_likelihoods = np.exp(log_posterior)
        #full_likelihoods = np.exp((np.log(self.weights[:, None]) + naive_logs) - naive_logs)
        full_likelihoods = np.exp(log_posterior)

        normalized_priors = self.lg_weights/np.sum(self.lg_weights)

        if return_likelihoods == True:
            
            # Calcualte expected relevance weights
            all_relevance_weights = []
            kernel_range = kernel_ranges[0]
            relevance_weights = self.calc_relevance_weights(X, kernel_range)
            for k in xrange(relevance_weights.shape[0]):
                all_relevance_weights.append(relevance_weights[k,:])
            
            for i in xrange(len(kernel_ranges)-1):
                kernel_range = kernel_ranges[i+1]
                relevance_weights = self.calc_relevance_weights(X, kernel_range)
                # Now iterate through relevance weights
                for k in xrange(relevance_weights.shape[0]):
                    all_relevance_weights[k] = np.vstack([all_relevance_weights[k], relevance_weights[k,:]])
            # Now integrate them
            integrated_relevance_weights = []
            for k in xrange(relevance_weights.shape[0]):
                integrated_relevance_weights.append(integrate(all_relevance_weights[k]))
                
            # Then stack them
            integrated_relevance_weights = np.vstack(integrated_relevance_weights)


            return return_val, kernel_ranges, full_likelihoods, avg_kernel_ranges, normalized_priors, integrated_relevance_weights
        else:
            return return_val

    def log_naive_local_pdf(self, X_eval, kernel_range):
        X_data = self.get_X_stack()
        y_data = self.get_y_stack()
        q = X_data.shape[1]
        p = X_eval.shape[1]

        model_weights, errors, N_eff, resi_mean, marginal_likelihoods = self.estimate_model_weights(X_eval, return_likelihoods=True, kernel_range=kernel_range)
        log_marginal_likelihoods = np.log(marginal_likelihoods+1e-16)
        log_prior = self.estimate_log_model_priors(X_eval, kernel_range)
        # Add logs
        logsum = log_prior + log_marginal_likelihoods
        # Then sum exp log over the models (axis=0)
        log_naive_likelihoods = scipy.misc.logsumexp(logsum, axis=0)
        return log_naive_likelihoods
        
    def log_dataless_pdf(self, X, y, kernel_range=-1.0, bool_return_model_pdfs=False):
        """
        Computes the log local pdf of observing the values of y at the locations X
        WITHOUT any training data, given the current location x. 

        args:
        -----
        X : (n x q matrix) of q data point locations
        y : (q dimsional vector) of function values at each of the q data point locations
        X_eval : (n x p matrix) of locations

        returns:
        --------
        p dimensional numpy vector of probabilities of the given observations
        """
        p = X.shape[1]

        if kernel_range == -1:
            kernel_range = self.kernel_range

        def Q(x):
            """
            Array of each model's prediction on x
            """
            return np.vstack([qm.predict(x) for qm in self.quadratic_models])

        # Get the n_models x p matrix of log model priors
        log_prior = self.estimate_log_model_priors(X, kernel_range)

        # gets a n_models x q matrix with quadratic of model i on observation x_j
        model_means_at_X = np.hstack([Q(X[:,i]) for i in xrange(p)])
        log_prob = np.log(1 + np.power(y[None, :] - model_means_at_X, 2)
                          / (2*self.precision_beta))
        log_prob *= -self.precision_alpha

        # Get the t distribution scale parameter (sigma^2)
        divfactor = (self.precision_alpha+0.5*0)
        postfactor = (self.precision_beta+0.5*0) / divfactor[None,:]
        
        # Sigma for the student t posterior distributions are postfactor square root
        sigma = np.sqrt(postfactor)

        # Get nu (1 x p) vector
        nu = 2*self.precision_alpha

        # Get z scores  (nModels x p)
        z = (y[None, :] - model_means_at_X) / sigma

        log_prob = scipy.stats.t.logpdf(z, nu[None, :])



        # Calculate probabiltiy dot product using log sum
        comb_log_prob = log_prob + log_prior
        
        # Get the log sum exp across the models (likelihood for each 
        # individual data point)
        log_indv_probs = scipy.misc.logsumexp(comb_log_prob, axis=0)
    
        # Get the relevance weights of the position x
        relevance_weights = self.calc_relevance_weights(X, kernel_range)
        
        # Get the likelihood across all the data = weighted avg
        # of log likelihoods with relevance weights

        # Currently not working
        total_log_probs = 0
        # log_probs = np.sum(np.multiply(log_indv_probs[:, None], relevance_weights), axis=0)

        if bool_return_model_pdfs:
            return total_log_probs, log_prob, relevance_weights
        else:
            return log_probs

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

        sample_size = self.get_n_models()
        
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
                 n_int=50,
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
                                       n_int=n_int,
                                       hessian_distances=self.hessian_distances,
                                       kernel_type=kernel_type)

        self.set_kernel_range(init_kernel_range)  # Set init kernel_range
        
    def set_gamma_kernel_prior_mode(self, mode, var):
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

    def set_gamma_kernel_prior(self, mean, var):
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
        kernel_prior.set_mean_var(mean, var)  # Set Gamma params using MEAN and var
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

    def set_bias_prior_params(self, bias_lambda):
        """
        Setter for the bias parameters. The bias prior is a normal gamma
        related to the precision prior
        
        args:
        -----
        bias_lambda  : (scalar) lambda value
        """
        self.bma.set_bias_prior_params(bias_lambda)

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
        return self.bma.get_n_models()

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

    def predict_bayesian(self, X, return_H=False):
        """
        Full Bayesian prediction, integrating over the kernel range
        """
        return self.bma.predict_bayesian(X, return_H=return_H)

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
        model_weights, errors, N_eff = self.bma.estimate_model_weights(X, return_errors=True, kernel_range=self.kernel_range)
        return N_eff

    def calculate_N_eff_bayesian(self, X):
        return self.bma.predict_bayesian(X)[3,:]

    def calculate_KL_bayesian(self, X):
        return self.bma.predict_bayesian(X, return_H=True)[4,:]

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
        predictions = self.bma.predict_bayesian(X)
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
        """
        if self.verbose:
            misc.tic()
            print("BayesianOracle>> optimizing hyperparameters")

        self.bma.find_kernel_MAP_estimate()
        if self.verbose:
            print("BayesianOracle>> hyperparameter optimization finished in %.1f seconds" % misc.toc())
            print("")
        """
