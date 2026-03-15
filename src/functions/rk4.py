import numpy as np

def rk4(function, h, x, *args):
    """
    RK4 integration.

    Parameters
    ----------
    function : callable
        Dynamics function: xdot = function(x, *args)
    h : float
        Sampling time (seconds)
    x : numpy.ndarray
        Current state vector x[k]
    *args :
        Additional parameters passed to the dynamics function

    Returns
    -------
    numpy.ndarray
        Updated state vector x[k+1]
    """

    # k1
    k1 = function(x, *args)

    # k2
    k2 = function(x + 0.5 * h * k1, *args)

    # k3
    k3 = function(x + 0.5 * h * k2, *args)

    # k4
    k4 = function(x + h * k3, *args)

    # RK4 update
    x_next = x + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return x_next