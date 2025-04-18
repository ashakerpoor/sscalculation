import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve



S,Q = u0.shape()

# Initialize the reaction matrix
# def reaction_term(u_i, i):


def residual(u):
    res = np.zeros_like(u)

    for i in range(Q):
        for s in range(S):
            diff_term = 0.0

            for j in neighbors[i]:
                index = (i,j) if (i,j) in A else (j,i)
                Aij = A[index]
                dij = d[index]
                diff_term += D[s] * Aij / dij * (u[(s,j)] - u[(s,i)])
            
            react_term = 0.0 # TODO: add the correct fucntion call later
            res[s, i] = diff_term + react_term

    return res



# Calculate the Jacobian


# Implement the Newton-Raphson algorithm