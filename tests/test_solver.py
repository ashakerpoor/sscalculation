import numpy as np
from sscalculation.newton_solver import NewtonRaphson



def test_newton(solver, u0):
    solver.newton(u0)


def test_residual():
    # Parameter setup for a 3x2 voxel grid
    neighbors = {
        0: [1, 3],
        1: [0, 2, 4],
        2: [1, 5],
        3: [0, 4],
        4: [1, 3, 5],
        5: [2, 4],
    }

    Q = len(neighbors) # number of compartments
    S = 2 # number of species
    D = [1.0, 2.0]
    V = [1.0] * Q
    A = {}
    d = {}
    for i in range(Q):
        for j in neighbors[i]:
            if (i, j) not in A and (j, i) not in A:
                A[(i, j)] = 1.0
                d[(i, j)] = 1.0

    
    reactions = [[] for _ in range(Q)]
    u = np.ones((S, Q))

    solver = NewtonRaphson(neighbors, A, d, D, V, reactions)
    res = solver.residual(u)

    expected = np.zeros_like(u)
    np.testing.assert_allclose(res, expected, atol=1e-12)

    
def test_reaction_term(solver, u):
    """
    Test case of A + B <-> 2C
    No diffusion
    Species: A:0, B:1, C:2
    powers = [1,1,2], rates = [2.,1.]
    """

    # Reaction terms associated w/ voxel 0
    ds_A = solver.reaction_term(u, 0, 0)
    ds_B = solver.reaction_term(u, 1, 0)
    ds_C = solver.reaction_term(u, 2, 0)

    r_, p_, s_, pw_, rates = solver.reactions[0][0]
    kfwd, krev = rates[0], rates[1]

    # Expected values
    fwd = kfwd * (1.0 ** 1) * (2.0 ** 1)      # 5.0
    rev = krev * (3.0 ** 2)                  # 5.0
    net = fwd - rev                          # -10.0

    np.testing.assert_allclose(ds_A, 5.0, atol=1e-12)
    np.testing.assert_allclose(ds_B, 5.0, atol=1e-12)
    np.testing.assert_allclose(ds_C, -10.0, atol=1e-12)



def main():
    neighbors = {
        0: []
    }
    Q = len(neighbors)
    S = 3  # Species: A, B, C

    # Diffusion params (not used in this test)
    A, d = {}, {}
    for i in neighbors:
        for j in neighbors[i]:
            if (i, j) not in A and (j, i) not in A:
                A[(i, j)] = 1.0
                d[(i, j)] = 1.0
    
    reactions = [[] for _ in range(Q)]

    #####################################################################
    ##### 1st reaction test case (1 reaction):
    ##### Reaction: A + B <-> 2C in voxel 0
    ##### expected values: A = 1.3, B = 2.3, C = 2.4
    D = [0.0] * S
    V = [1.0] * Q
    kfwd = 2.0
    krev = 1.0
    reactions[0].append((   # the first and only reaction in voxel 0
        [0, 1],     # reactants: A, B
        [2],        # products: C
        [1, 1, 2],  # stoichiometries
        [1, 1, 2],  # powers
        [kfwd, krev]
    ))

    u0 = np.array([
        [1.0],  # A
        [2.0],  # B
        [3.0],  # C
    ])

    solver = NewtonRaphson(neighbors, A, d, D, V, reactions)
    test_newton(solver,u0)
    # test_reaction_term(solver, u0)
    # test_residual()

if __name__ == "__main__":
    main()