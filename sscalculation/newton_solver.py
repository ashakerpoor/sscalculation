import numpy as np
from scipy.sparse import lil_matrix, vstack
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import lsqr
from scipy.linalg import svd
from scipy.linalg import qr
from numpy.linalg import cond
import itertools



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


    # Implement without mass conservation constraints
    def newton(self, u0, tol=1e-6, max_iter=200):
        self.S, self.Q = u0.shape
        u = u0.copy()
        gWr, M_tot = self.get_static_params(u0)

        for k in range(max_iter):
            F_unconstrained = self.residual(u)
            F = self.constrained_res(u, F_unconstrained, gWr, M_tot)

            norm_F = np.linalg.norm(F)

            print(f"Iter {k}: ||F|| = {norm_F:.3e}")
            if norm_F < tol:
                print("Converged.")
                print("u:\n", u)
                return u

            J_unconstrained = self.jacobian(u)    # shape: (S*Q, S*Q)
            J_aug = vstack([J_unconstrained, gWr]).tocsr()  # J_aug shape: (S*Q + K, S*Q), gWr shape: (K, S*Q)
            condJ = cond(J_aug.toarray())
            print("Condition number of Jacobian:", condJ)

            # delta_u = spsolve(J, -F.flatten())
            delta_u = lsqr(J_aug, -F.flatten())[0]
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
    """






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
    

    def get_stoichmatrix(self, i):
        """
        Constructs the stoichiometric matrix N for compartment i.
        """
        reaction_list = self.reactions[i]
        if not reaction_list:
            return np.zeros((0, 0))

        # Determine local number of species by maximum index in reactants/products
        # TODO needs to be made more scalable for other formats of input data.
        max_idx = 0
        for reaction in reaction_list:
            reactants, products, _, _, _ = reaction
            all_species = reactants + products
            if all_species:
                max_idx = max(max_idx, max(all_species))
        S_local = max_idx + 1
        R = len(reaction_list)
        N = np.zeros((S_local, R))

        for r_idx, reaction in enumerate(reaction_list):
            reactants, products, stoichs, powers, rate_constants = reaction

            for species_idx, stoich in zip(reactants, stoichs[:len(reactants)]):
                N[species_idx, r_idx] -= stoich

            for species_idx, stoich in zip(products, stoichs[len(reactants):]):
                N[species_idx, r_idx] += stoich

        return N
    

    @staticmethod
    def calc_nullspace(N, tol=1e-12):
        """
        Computes the nullspace of the stoichiometric matrix N.
        Returns an array (S, k) with k being the nullity.
        """
        U, s, Vh = svd(N.T)
        nullmask = s < tol
        nullspace = U[:, nullmask]

        return nullspace  # shape: (S_local, num_null_vectors)
    # def calc_nullspace(N, tol=1e-12):
    #     """ Computes the nullspace of the stoichiometric matrix N. """
    #     U, s, Vh = svd(N.T)
    #     nullmask = s < tol
    #     nullspace = Vh[nullmask, :].T  # Use right singular vectors (Vh)
    #     return nullspace  # shape: (S_local, num_null_vectors)
    

    def calc_gW(self):
        """
        Constructs the global conservation matrix gW.
        Each row of gW corresponds to a global conservation law vector w \in R^{S*Q}.
        While the rows are zero padded to match the global dimension of problem (R^{S*Q}),
        each row corresponds to local conservation laws.
        The conservation law is of the form: w^T u = M_total.
        """
        k_list = []
        gW_list = []

        for i in range(self.Q):
            N_i = self.get_stoichmatrix(i)
            w_i = self.calc_nullspace(N_i)
            k_i = w_i.shape[1]
            k_list.append(k_i)

            if w_i.size == 0:
                continue

            species_involved = set()
            for reaction in self.reactions[i]:
                reactants, products, *_ = reaction
                species_involved.update(reactants + products)
            species_involved = sorted(species_involved)
            # local2global = {s_local: s_global for s_local, s_global in enumerate(species_involved)}

            # w_i has shape (S_local, k_i), so iterate over each nullspace vector
            for k in range(k_i):
                w_local = w_i[:, k]
                w_global = np.zeros(self.S * self.Q)

                for s_local, w_val in enumerate(w_local):
                    # s_global = local2global[s_local]
                    s_global = species_involved[s_local]
                    global_idx = s_global * self.Q + i
                    w_global[global_idx] = self.V[i] * w_val

                gW_list.append(w_global)

        if not gW_list:
            return np.zeros((0, self.S * self.Q)), k_list

        gW = np.vstack(gW_list)  # shape: (K, S*Q), K = total raw conservation laws
        return gW, k_list
    

    def calc_gWg(self, gW, k_list):
        """
        Constructs global conservation law candidates by linear combination of local laws.
        Each compartment contributes exacly 1 local law  (if available) into each linear combo.

        Parameters:
        gW: (K_total, S*Q), rows are local volume-weighted conservation vectors
        k_list: list of ints [k_0, ..., k_{Q-1}], number of local constraints per compartment
        """
        if not (k_list or sum(k_list)):
            return np.zeros((0, self.S * self.Q))

        gW_blocks = []
        start = 0
        for k in k_list:
            end = start + k
            gW_blocks.append(gW[start:end])
            start = end

        combinations = list(itertools.product(*gW_blocks))
        gWg = np.array([np.sum(combo, axis=0) for combo in combinations])  # shape: (K_comb, S*Q)

        return gWg
    

    @staticmethod
    def reduce_gWg(gWg, tol=1e-12):
        if gWg.shape[0] <= 1:
            return gWg
        
        Q, R, P = qr(gWg.T, mode='economic', pivoting=True)
        rank = np.sum(np.abs(np.diag(R)) > tol)
        
        # P determines which columns (in gWg.T) are independent
        independent_rows = np.sort(P[:rank])
        return gWg[independent_rows]


    @staticmethod
    def calc_Mtotal(u0, gWr):
        """
        Computes the vector of total conserved mass M_total.
        Each entry is w_k^T u0.flatten() per global conservation law w_k.
        """
        u0_flat = u0.flatten()  # shape: (S*Q,)
        M_total = gWr @ u0_flat  # shape: (K,)
        return M_total