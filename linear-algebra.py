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

def swaprows(i, j, Ab):
    (n, m) = Ab.shape
    temp_row = np.zeros(m)

    temp_row[:] = Ab[i, :]
    Ab[i, :] = Ab[j, :]
    Ab[j, :] = temp_row[:]

    return(Ab)

def gauss_pp(Ab):
    (n, m) = Ab.shape
    # Row reduce
    for k in range(0, n-1):
        # Find the best pivot using partial pivoting
        MAX = abs(Ab[k, k])
        I = k 
        for i in range(k+1, n):
            m = abs(Ab[i, k])
            if m > MAX:
                print("Row %d switched with Row %d:" % ((i+1), (k+1)))
                MAX = m
                I = i

        swaprows(I, k, Ab)

        # Continue with row reduction after finding the best pivot
        for j in range(k+1, n):
            c = Ab[j, k]/Ab[k, k]
            Ab[j, :] = Ab[j, :] - c*Ab[k, :]

        print(Ab, "\n")

    # back-substitute [A:b]
    (n, m) = Ab.shape
    for i in range(n-1, -1, -1):
        # subtract constants from solution
        for j in range(m-1):
            if j != i:
                Ab[i, m-1] -= Ab[i, j]
                Ab[i, j] = 0

        # divide both sides by coefficent
        Ab[i, m-1] /= Ab[i, i]
        Ab[i, i] /= Ab[i, i]

        # substitute value back into other rows
        for k in range(i-1, -1, -1):
            Ab[k, i] *= Ab[i, m-1]

        # print augmented matrix
        print("After solving for x%d:\n" % (i+1), Ab, "\n")

    # print solution
    print("Solution:", (Ab[:, -1]).reshape(-1))

    return Ab

def gauss_npp(Ab):
    (n,m) = Ab.shape
    #Row Reduce [A:b]
    for k in range(0, n-1): #produce k-th column of zeros
        for j in range(k+1, n): #j-th row operation
            c = Ab[j,k]/Ab[k,k]
            Ab[j,:] = Ab[j,:] - c*Ab[k,:]
    
    # print augmented matrix after row reduction
    print("After row reduction:\n", Ab, "\n")
    
    # back-substitute [A:b]
    for i in range(n-1, -1, -1):
        # subtract constants from solution
        for j in range(m-1):
            if j != i:
                Ab[i, m-1] -= Ab[i, j]
                Ab[i, j] = 0
        
        # divide both sides by coefficent
        Ab[i, m-1] /= Ab[i, i]
        Ab[i, i] /= Ab[i, i]
        
        # substitute value back into other rows
        for k in range(i-1, -1, -1):
            Ab[k, i] *= Ab[i, m-1]
        
        # print augmented matrix
        print("After solving for x%d:\n" % (i+1), Ab, "\n")
        
    # print solution
    print("Solution:", (Ab[:, -1]).reshape(-1))
    
    return Ab
