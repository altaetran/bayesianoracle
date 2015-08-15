import numpy as np
import scipy.stats

class LogNormalPrior(object):
    """ Single dimensional Log normal prior """
    def __init__(self, mean=1.0, sigma=1.0):
        self.mean = mean
        self.sigma = sigma

    def set_mode_var(self, mode, var):
        # Determine beta and alpha for Gamma using the mean and var
        import scipy.optimize
        
        if var / mode**2 > 1e6:
            def func(sigma):
                # Large sigma approximation
                return (sigma**2)*(3*sigma**2) - np.log(var / mode**2)
        else:
            def func(sigma):
                return (np.exp(sigma**2)-1)*np.exp(3*sigma**2) - var / mode**2

        sigma = scipy.optimize.fsolve(func, 1.0)[0]
        mu = np.log(mode) + sigma**2

        # Set params
        self.mean = mu
        self.sigma = sigma

        # Set the shape param to sigma
        self.a = sigma
        # Set the scale to exp(mu)
        self.scale = np.exp(mu)

    def get_mode(self):
        return np.exp(self.mean-self.sigma**2)

    def get_scale(self):
        return self.scale

    def rvs(self, size=None):
        return scipy.stats.lognorm.rvs(self.sigma, loc=0, scale=np.exp(self.mean),
                                       size=size)

    def logpdf(self, x):
        return scipy.stats.lognorm.logpdf(x, self.sigma, loc=0, scale=np.exp(self.mean))

    def interval(self, alpha):
        if alpha == 1:
            return (0, np.inf)
        else:
            raise ValueError("Unsupported interval!")

class UniformPrior(object):
    """ Single dimensional uniform prior over bounds, of the form: [lower,upper] """
    def __init__(self, bounds):
        self.bounds = bounds

    def rvs(self, size=None):
        return scipy.stats.uniform.rvs(loc=self.bounds[0], scale=self.bounds[1]-self.bounds[0],
                                       size=size)

    def logpdf(self, x):
        return scipy.stats.uniform.logpdf(x, loc=self.bounds[0], scale=self.bounds[1]-self.bounds[0])

    def interval(self, alpha):
        if alpha == 1:
            return (0, np.inf)
        else:
            raise ValueError("Unsupported interval!")

class ExponentialPrior(object):
    """ Single dimensional Log normal prior """
    def __init__(self, lam):
        self.scale = 1.0/lam

    def rvs(self, size=None):
        return scipy.stats.expon.rvs(scale=self.scale,
                                     size=size)

    def get_scale(self):
        return self.scale

    def logpdf(self, x):
        return scipy.stats.expon.logpdf(x, scale=self.scale)

    def interval(self, alpha):
        if alpha == 1:
            return (0, np.inf)
        else:
            raise ValueError("Unsupported interval!")

class GammaPrior(object):
    """ Single dimensional gamma prior """
    def __init__(self, shape=1.0, scale=1.0):
        self.a = shape
        self.scale = scale

    def set_mode_var(self, mode, var):
        # Determine beta and alpha for Gamma using the mean and var
        beta = (mode + np.sqrt(np.square(mode)+4.0*var)) / (2.0*var)
        alpha = np.square(beta)*var
        
        # Set the shape param to alpha
        self.a = alpha
        # Set the scale to the reciprocal of beta
        self.scale = 1.0 / beta

    def get_mode(self):
        return (self.a-1.0) * self.scale

    def get_beta(self):
        return 1.0 / self.scale

    def get_alpha(self):
        return self.a

    def rvs(self, size=None):
        return scipy.stats.gamma.rvs(self.a, scale = self.scale,
                                     size=size)

    def logpdf(self, x):
        return scipy.stats.gamma.logpdf(x, self.a, scale=self.scale)

class InvGammaPrior(object):
    """ Single dimensional gamma prior """
    def __init__(self, shape=1.0, scale=1.0):
        self.a = shape
        self.scale = scale

    def set_mode_var(self, mode, var):
        # Determine beta and alpha for Gamma using the mean and var
        import scipy.optimize

        def func(alpha):
            return (alpha+1)**2/((alpha-1)**2*(alpha-2)) - var / mode**2

        alpha = scipy.optimize.fsolve(func, 3.0)[0]
        beta = mode*(alpha+1)

        # Set the shape param to alpha
        self.a = alpha
        # Set the scale to beta
        self.scale = beta

    def get_mode(self):
        return self.scale / (self.a + 1)

    def get_scale(self):
        return self.scale

    def rvs(self, size=None):
        return scipy.stats.invgamma.rvs(self.a, scale = self.scale,
                                     size=size)

    def logpdf(self, x):
        return scipy.stats.invgamma.logpdf(x, self.a, scale=self.scale)
