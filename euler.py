"""
@author: Rowan Parker
@date: 02/21/2022
"""
def euler(f, x0, y0, xn, n):
    """
    Forward Euler's Method for a single linear equation.

    Parameters
    ----------
    f : function
        linear equation.
    x0 : float
        initial x value.
    y0 : float
        initial f(x) value.
    xn : float
        x-value at which to estimate f.
    n : integer
        number of iterations.

    Returns
    -------
    x_values : array
        list of x values.
    y_estimate : array
        list of f(x) values.
    """
    y_estimate = [y0]
    x_values = [x0]
    h = abs((xn - x0)/n)

    for i in range(n):
        y_estimate.append(y_estimate[i] + h*f(x_values[i], y_estimate[i]))
        x_values.append(x_values[i] + h)

    return x_values, y_estimate

def euler(f0, f1, t0, x0, t1, y0, n, xn, yn, debug=False):
    """
    Forward Euler's Method for a system of two linear equations.

    Parameters
    ----------
    f0 : function
        f1 equation.
    f1 : function
        f2 equation.
    t0 : float
        initial t-value for f1.
    x0 : float
        initial f1(t) value.
    t1 : float
        initial t-value for f2.
    y0 : float
        initial f2'(t) value.
    n : integer
        number of iterations.
    xn : float
        t-value at which to estimate f1.
    yn : float
        t-value at which to estimate f2.
    debug : TYPE, optional
        enable debugging print statements. The default is False.

    Returns
    -------
    x_estimate : array
        list of f1 values.
    y_estimate : array
        list of f2 values.
    """
    x_estimate = [x0]
    y_estimate = [y0]

    h = abs((xn - t0)/n)
    h0 = t0

    for i in range(n + 1):
        if debug:
            print("h = %.3f \t x = %.16f y = %.16f" %
                  (h0, x_estimate[i], y_estimate[i]))
        x_estimate.append(x_estimate[i] + h*f0(x_estimate[i], y_estimate[i]))
        y_estimate.append(y_estimate[i] + h*f1(x_estimate[i], y_estimate[i]))
        h0 += h

    return x_estimate, y_estimate
