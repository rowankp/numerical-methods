"""
@author: Rowan Parker
@date: 10/24/2021
"""
import numpy as np

def incremental_search(f, a, b, dx):
    """
    Searches the interval (a,b) in increments dx for
    the bounds (x1,x2) of the smallest root of f(x)

    Parameters
    ----------
        f  - function
        a  - left bound of the interval, (a, b)
        b  - right bound of the interval, (a, b)
        dx - value by which to increment

    Returns
    -------
    x1 = x2 = None if no roots were detected
    """
    x1 = a
    x2 = x1 + dx

    f1 = f(x1)
    f2 = f(x2)

    while (f1*f2) > 0:
        if x1 >= b:
            return None, None

        x1 = x2
        x2 = x1 + dx

        f1 = f2
        f2 = f(x2)

    return x1, x2


def newton(f, f_prime, x0, tol):
    """
    Find the root of f(x) = 0 using the Newton-Raphson method recursively.

    Parameters
    ----------
        f       - function
        f_prime - derivative of f
        x0      - starting value
        tol     - tolerance

    Returns
    -------
    root of f(x)
    """
    if abs(f(x0)) < tol:
        return x0
    else:
        x1 = x0 - (f(x0)/f_prime(x0))
        print("x0 = ", x0, "f(x0) = ", f(x0), "f'(x0) =", f_prime(x0), "x1 = ", x1)
        return newton(f, f_prime, x1, tol)


def rbisection(f, a, b, tol):
    """
    Find the root of f(x) = 0 by bisection recursively.
    The root must be in the interval (a, b).

    Parameters
    ----------
        f   - function
        a   - left bound of the interval, (a, b)
        b   - right bound of the interval, (a, b)
        tol - tolerance

    Returns
    -------
    root of f(x)
    """
    if np.sign(f(a)) == np.sign(f(b)):
        raise Exception("There is either no root or multiple roots between "
                        "{} and {}".format(a, b))

    # midpoint
    m = (a + b) / 2

    if np.abs(f(m)) < tol:
        return m
    elif np.sign(f(a)) == np.sign(f(m)):
        return bisection(f, m, b, tol)
    elif np.sign(f(b)) == np.sign(f(m)):
        return bisection(f, a, m, tol)


def bisection(f, a, b, tol):
    """
    Find the root of f(x) = 0 by bisection without recursion.
    The root must be in the interval (a, b).

    Parameters
    ----------
        f   - function
        a   - left bound of the interval, (a, b)
        b   - right bound of the interval, (a, b)
        tol - tolerance

    Returns
    -------
    root of f(x)
    """
    x1 = a
    x2 = b
    delta_x = np.abs(x2 - x1)
    steps = int(np.ceil(np.log(delta_x / tol) / np.log(2.0)))

    if np.sign(f(a)) == np.sign(f(b)) or f(a)*f(b) >= 0:
        print("There is either no root or multiple roots between "
              "{} and {}".format(a, b))

    for i in range(steps):
        m = (x1 + x2) / 2

        if f(x1)*f(m) < 0:
            x2 = m
        elif f(x2)*f(m) < 0:
            x1 = m
        elif f(m) == 0:
            return m

    return (x1 + x2) / 2
