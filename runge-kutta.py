# -*- coding: utf-8 -*-
"""
@author: Rowan Parker
@date: 03/05/2022
"""
import math

def rk2_two_eq(t0, tn, n, x0, y0, f1, f2, debug=False):
    """
    Runge-Kutta method of order 2 for a system of two linear equations.

    Parameters
    ----------
    t0 : float
        initial t-value.
    tn : float
        t-value at which to estimate f1 and f2.
    n : integer
        number of iterations.
    x0 : float
        initial f1(t0) value.
    y0 : float
        initial f2(t0) value.
    f1 : function
        equation 1.
    f2 : function
        equation 2.
    debug : boolean, optional
        enable debugging print statements. The default is False.

    Returns
    -------
    t : array
        list of t values.
    x : array
        list of x(t) values.
    y : array
        list of y(t) values.
    """
    t = [t0]
    x = [x0]
    y = [y0]
    h = abs((tn-t0)/n)

    for i in range(n+1):
        K11 = f1(t[i], x[i], y[i])
        K12 = f2(t[i], x[i], y[i])

        K21 = f1(t[i] + (h/2),
                 x[i] + (h/2)*K11,
                 y[i] + (h/2)*K12)
        K22 = f2(t[i] + (h/2),
                 x[i] + (h/2)*K11,
                 y[i] + (h/2)*K12)

        t.append(t[i] + h)
        x.append(x[i] + h*K21)
        y.append(y[i] + h*K22)

        if debug:
            print("STEP %d\nt = %.16f\t x = %.16f\t y = %.16f\n"
                  "K11 = %.16f\t K12 = %.16f\n"
                  "K21 = %.16f\t K22 = %.16f\n"
                  "x(%.1f) = %.16f\ny(%.1f) = %.16f\n"
                  % (i, t[i], x[i], y[i],
                     K11, K12, K21, K22,
                     t[-1], x[-1], t[-1], y[-1]))

    # last step is extra to support debug print messages, so remove
    # before returning (example: 3 steps are desired, 4 steps are performed)
    return t[:n+1], x[:n+1], y[:n+1]


def rk4_two_eq(t0, tn, n, x0, y0, f1, f2, debug=False):
    """
    Runge-Kutta method of order 4 for a system of two linear equations.

    Parameters
    ----------
    t0 : float
        initial t-value.
    tn : float
        t-value at which to estimate f1 and f2.
    n : integer
        number of iterations.
    x0 : float
        initial f1(t0) value.
    y0 : float
        initial f2(t0) value.
    f1 : function
        equation 1.
    f2 : function
        equation 2.
    debug : boolean, optional
        enable debugging print statements. The default is False.

    Returns
    -------
    t : array
        list of t values.
    x : array
        list of x(t) values.
    y : array
        list of y(t) values.
    """
    t = [t0]
    x = [x0]
    y = [y0]
    h = abs((tn-t0)/n)

    for i in range(n+1):
        K11 = f1(t[i], x[i], y[i])
        K12 = f2(t[i], x[i], y[i])

        K21 = f1(t[i] + (h/2),
                 x[i] + (h/2)*K11,
                 y[i] + (h/2)*K12)
        K22 = f2(t[i] + (h/2),
                 x[i] + (h/2)*K11,
                 y[i] + (h/2)*K12)

        K31 = f1(t[i] + (h/2),
                 x[i] + (h/2)*K21,
                 y[i] + (h/2)*K22)
        K32 = f2(t[i] + (h/2),
                 x[i] + (h/2)*K21,
                 y[i] + (h/2)*K22)

        K41 = f1(t[i] + h,
                 x[i] + h*K31,
                 y[i] + h*K32)
        K42 = f2(t[i] + h,
                 x[i] + h*K31,
                 y[i] + h*K32)

        t.append(t[i] + h)
        x.append(x[i] + (h/6)*(K11 + 2*K21 + 2*K31 + K41))
        y.append(y[i] + (h/6)*(K12 + 2*K22 + 2*K32 + K42))

        if debug:
            print("STEP %d\nt = %.16f\t x = %.16f\t y = %.16f\n"
                  "K11 = %.16f\t K12 = %.16f\n"
                  "K21 = %.16f\t K22 = %.16f\n"
                  "K31 = %.16f\t K32 = %.16f\n"
                  "K41 = %.16f\t K42 = %.16f\n"
                  "x(%.1f) = %.16f\ny(%.1f) = %.16f\n"
                  % (i, t[i], x[i], y[i],
                     K11, K12, K21, K22, K31, K32, K41, K42,
                     t[-1], x[-1], t[-1], y[-1]))

    # last step is extra to support debug print messages, so remove
    # before returning (example: 3 steps are desired, 4 steps are performed)
    return t[:n+1], x[:n+1], y[:n+1]
