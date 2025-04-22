import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve



S,Q = u0.shape()


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
            
            react_term = V[i] * reaction_term(u, s, i)
            res[s, i] = diff_term + react_term

    return res


def reaction_term(u, s, i):
    ds = 0.0

    for reactants, products, stoich, powers, rates in reactions[i]:
        # stoich = [r1, r2, ..., p1, p2, ...]
        # powers = [pr1, pr2, ..., pp1, pp2, ...]
        # rates = [kfwd, krev]

        len_r = len(reactants)
        sr = 1 if s in reactants else 0
        sp = 1 if s in products else 0

        if not (sr or sp):
            continue

        fwd = rates[0]
        for idx, specie in enumerate(reactants):
            conc = u[specie, i]
            fwd *= pow(conc, powers[idx])

        rev = 0.0
        if rates[1] > 0:
            rev = rates[1]
            for idx, specie in enumerate(products):
                conc = u[specie, i]
                rev *= pow(conc, powers[len_r+idx])

        delta = 0.0
        if sr:
            idx = reactants.index(s)
            delta -= stoich[idx]*(fwd - rev)
        if sp:
            idx = products.index(s)
            delta += stoich[len_r+idx]*(fwd - rev)

        ds += delta

    return ds