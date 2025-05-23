import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


# Newton-Raphson solver
class NewtonRaphson:
    def __init__(self, neighbors, A, d, D, V, reactions):
        self.neighbors = neighbors
        self.A = A
        self.d = d
        self.D = D
        self.V = V
        self.reactions = reactions
        self.S = None
        self.Q = None


    def newton(self, u0, tol=1e-6, max_iter=20):
        self.S, self.Q = u0.shape
        u = u0.copy()

        for k in range(max_iter):
            F = self.residual(u)
            norm_F = np.linalg.norm(F)

            print(f"Iter {k}: ||F|| = {norm_F:.3e}")
            if norm_F < tol:
                print("Converged.")
                return u

            J = self.jacobian(u)
            delta_u = spsolve(J, -F.flatten())
            u += delta_u.reshape(self.S, self.Q)

        raise RuntimeError("Solver failed to converge.")


    def residual(self, u):
        res = np.zeros_like(u)

        for i in range(self.Q):
            for s in range(self.S):
                diff_term = 0.0

                for j in self.neighbors[i]:
                    index = (i,j) if (i,j) in self.A else (j,i)
                    Aij = self.A[index]
                    dij = self.d[index]
                    diff_term += self.D[s] * Aij / dij * (u[(s,j)] - u[(s,i)])
                
                react_term = self.V[i] * self.reaction_term(u, s, i)
                res[s, i] = diff_term + react_term

        return res


    def reaction_term(self, u, s, i):
        ds = 0.0

        for reactants, products, stoich, powers, rates in self.reactions[i]:
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


    def jacobian(self, u, h=1e-6):
        N = self.S * self.Q
        J = lil_matrix((N, N))
        u_flat = u.flatten()
        F0 = self.residual(u).flatten()

        for k in range(N):
            u_perturbed = u_flat.copy()
            u_perturbed[k] += h
            F1 = self.residual(u_perturbed.reshape(self.S,self.Q)).flatten()
            J[:, k] = (F1 - F0) / h

        return J.tocsr()