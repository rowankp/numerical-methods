"""
@author: Rowan Parker
@date: 03/05/2022
"""
import numpy as np

def power_method(A, x, k, scale=True, debug=False):
    """
    Power method for finding dominant eigenvectors and eigenvalues.

    Parameters
    ----------
    A : n × n matrix
        Must have a dominant eigenvalue.
    x : n × 1 non-zero vector
        An initial approximation for the dominant eigenvector.
    k : number of iterations to perform
    scale : boolean, optional
        Use scaling. The default is True.
    debug : boolean, optional
        Enable debugging print statements. The default is False.

    Returns
    -------
    (dominant eigenvalue, dominant eigenvector)
    """
    for i in range(k):
        if debug:
            print("step_{}: x = \n{}".format(i, x))

        x = A*x
        if scale:
            x = np.divide(x, np.max(x))

    eigenvalue = np.dot(np.squeeze(np.asarray(A*x)),
                        np.squeeze(np.asarray(x)))
    eigenvalue /= np.dot(np.squeeze(np.asarray(x)),
                         np.squeeze(np.asarray(x)))

    return eigenvalue, x
