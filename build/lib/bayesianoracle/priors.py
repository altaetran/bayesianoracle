import numpy as np
import scipy.stats

class LogNormalPrior(object):
    """ Single dimensional Log normal prior """
    def __init__(self, mean, sigma):
        self.mean = mean
        self.sigma = sigma

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

    def rvs(self, size=None):
        return scipy.stats.gamma.rvs(self.a, scale = self.scale,
                                     size=size)

    def logpdf(self, x):
        return scipy.stats.gamma.logpdf(x, self.a, scale=self.scale)
