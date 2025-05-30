import numpy as np
from scipy.sparse import lil_matrix, vstack
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import lsqr
from numpy.linalg import cond


# Newton-Raphson solver
class NewtonRaphson:
    def __init__(self, neighbors, A, d, D, V, reactions):
        self.neighbors = neighbors
        self.A = A
        self.d = d
        self.D = D
        self.V = V
        self.reactions = reactions
        self.total_mass = None
        self.S = None
        self.Q = None

    """
    # Implement without mass conservation constraints
    def newton(self, u0, tol=1e-6, max_iter=200):
        self.S, self.Q = u0.shape
        u = u0.copy()

        for k in range(max_iter):
            F = self.residual(u)
            norm_F = np.linalg.norm(F)

            print(f"Iter {k}: ||F|| = {norm_F:.3e}")
            if norm_F < tol:
                print("Converged.")
                print("u:\n", u)
                return u

            J = self.jacobian(u)
            condJ = cond(J.toarray())
            print("Condition number of Jacobian:", condJ)
            # print("Jacobian:", J)

            # delta_u = spsolve(J, -F.flatten())
            delta_u = lsqr(J, -F.flatten())[0]
            # u += delta_u.reshape(self.S, self.Q)
            u += delta_u.reshape(self.S, self.Q)

        raise RuntimeError("Solver failed to converge.")


    def residual(self, u):
        res = np.zeros_like(u)
        # self.S, self.Q = u.shape
        # print(f"Type of self.Q: {type(self.Q)}, Value: {self.Q}")
        # print(f"Type of self.S: {type(self.S)}, Value: {self.S}")

        for i in range(self.Q):
            for s in range(self.S):
                diff_term = 0.0

                for j in self.neighbors[i]:
                    index = (i,j) if (i,j) in self.A else (j,i)
                    Aij = self.A[index]
                    dij = self.d[index]
                    diff_term += self.D[s] * Aij / dij * (u[(s,j)] - u[(s,i)])

                    # n_si = u[(s,i)] * self.V[i]
                    # n_sj = u[(s,j)] * self.V[j]
                    # diff_term += self.D[s] * Aij / dij * (n_sj / self.V[i] - n_si / self.V[j])
                    # diff_term += self.D[s] * Aij / dij * (self.V[j] * u[(s,j)] - self.V[i] * u[(s,i)])
                    # diff_term += self.D[s] * Aij / (dij*self.V[i]) * (u[(s,j)] - u[(s,i)])

                
                react_term = self.V[i] * self.reaction_term(u, s, i)
                # res[s, i] = diff_term / self.V[i] + react_term
                res[s, i] = diff_term + react_term

        return res
    """

    # Implement with the mass conservation constraints
    def newton(self, u0, tol=1e-6, max_iter=1000):
        self.S, self.Q = u0.shape
        self.total_mass = np.dot(u0, self.V)  # shape (S,)
        u = u0.copy()

        for k in range(max_iter):
            F_phys = self.residual(u)  # shape: (S*Q, 1)

            # Build mass conservation residual (shape: (S, 1))
            mass_res = np.zeros((self.S, 1))
            for s in range(self.S):
                mass_res[s, 0] = np.dot(self.V, u[s, :]) - self.total_mass[s]

            F = np.concatenate((F_phys, mass_res), axis=0)

            norm_F = np.linalg.norm(F)
            if k%1 == 0:
                print(f"Iter {k}: ||F|| = {norm_F:.3e}")
            if norm_F < tol:
                print("Converged.")
                print("u:\n", u)
                return u

            # Jacobian for physical part
            J_phys = self.jacobian(u)  # shape: (S*Q, S*Q)

            constraint_rows = lil_matrix((self.S, self.S * self.Q))
            for s in range(self.S):
                for i in range(self.Q):
                    constraint_rows[s, s * self.Q + i] = self.V[i]

            J_aug = vstack([J_phys, constraint_rows]).tocsr()  # shape: (S*Q + S, S*Q)

            # delta_u = spsolve(J_aug, -F.flatten())
            delta_u = lsqr(J_aug, -F.flatten())[0]  # shape: (S*Q,)
            u += delta_u.reshape(self.S, self.Q)

        print(f'Did not converge! u:\n {u}')
        raise RuntimeError("Solver failed to converge.")


    def residual(self, u):
        res = np.zeros_like(u)  # shape (S, Q)

        for i in range(self.Q):
            for s in range(self.S):
                diff_term = 0.0
                for j in self.neighbors[i]:
                    index = (i, j) if (i, j) in self.A else (j, i)
                    Aij = self.A[index]
                    dij = self.d[index]
                    diff_term += self.D[s] * Aij / dij * (u[(s, j)] - u[(s, i)])
                react_term = self.V[i] * self.reaction_term(u, s, i)
                res[s, i] = diff_term + react_term

        return res.reshape(-1, 1)  # shape: (S*Q, 1)







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

    # # Finite difference approximation
    # def jacobian(self, u, h=1e-6):
    #     N = self.S * self.Q
    #     J = lil_matrix((N, N))
    #     u_flat = u.flatten()
    #     F0 = self.residual(u).flatten()

    #     for k in range(N):
    #         u_perturbed = u_flat.copy()
    #         uk = u_flat[k]
    #         h_k = h * max(abs(uk), 1.0) # Avoid underflow problem
    #         u_perturbed[k] += h_k
    #         F1 = self.residual(u_perturbed.reshape(self.S,self.Q)).flatten()
    #         J[:, k] = (F1 - F0) / h_k

    #     return J.tocsr()


    # Central difference approx.
    def jacobian(self, u, h=1e-6):
        N = self.S * self.Q
        J = lil_matrix((N, N))
        u_flat = u.flatten()

        for k in range(N):
            u_forward = u_flat.copy()
            u_backward = u_flat.copy()
            uk = u_flat[k]
            h_k = h * max(abs(uk), 1.0)
            u_forward[k] += h_k
            u_backward[k] -= h_k

            F_forward = self.residual(u_forward.reshape(self.S, self.Q)).flatten()
            F_backward = self.residual(u_backward.reshape(self.S, self.Q)).flatten()

            J[:, k] = (F_forward - F_backward) / (2 * h_k)

        return J.tocsr()