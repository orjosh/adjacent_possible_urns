import scipy as sp
import numpy as np
from math import sqrt, pi, log, exp

def pdf_power(x, x_min, gamma):
    """
    Probability distribution function for a simple 1-parameter power law.

    Params:         - `x` is the value to evaluate at
                    - `x_min` is the lower-bound for the scaling behaviour
                    - `gamma` is the (positive) scaling exponent x^(-gamma) which must be > 1
    """
    if x <= 0:
        return ValueError("x must be > 0")
    elif gamma <= 1:
        return ValueError("gamma must be > 1")
    
    return (x**(-gamma))/sp.special.zeta(gamma, q=x_min)

def cdf_power(x, x_min, gamma):
    """
    Cumulative distribution function for a simple 1-parameter power law.  See `pdf_power()` for details. 
    """
    if x <= 0:
        return ValueError("x must be > 0")
    elif gamma <= 1:
        return ValueError("gamma must be > 1")

    return sp.special.zeta(gamma, q=x) / sp.special.zeta(gamma, q=x_min)

def pdf_lognormal(x, x_min, mu, sigma):
    """
    Probability distribution function for a log-normal. NOTE: this PDF assumes continuous data.

    Params:         - `x` is the value to evaluate at
                    - `x_min` is the lower-bound for the scaling behaviour
                    - `mu` is the mean
                    - `sigma` is the standard deviation
    """
    norm_const = sqrt(2/(pi*sigma**2))/sp.special.erfc((log(x_min)-mu)/(sqrt(2)*sigma))
    func = (1/x)*exp(-((log(x)-mu)**2)/(2*sigma**2))

    return norm_const*func

def cdf_lognormal(x, mu, sigma):
    return (1/2)*sp.special.erfc(-(log(x)-mu)/(sqrt(2)*sigma))

def get_empirical_cdf(data):
    """
    Calculates the empirical cumulative distribution function for the given data, and returns it as an array
    of points.

    Params:         - `data` is a 1-dimensional list of points representing the observations 
    """
    n = len(data)
    return [(1/n)*x for x in range(1, len(data)+1)]