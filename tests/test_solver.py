import numpy as np
from sscalculation.newton_solver import NewtonRaphson


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
    for i in range(6):
        for j in neighbors[i]:
            if (i, j) not in A and (j, i) not in A:
                A[(i, j)] = 1.0
                d[(i, j)] = 1.0

    
    reactions = [[] for _ in range(Q)]
    u = np.ones((S, Q))

    solver = NewtonRaphson(neighbors, A, d, D, V, reactions)
    res = solver.residual(u)

    expected = np.zeros_like(u)
    print(res)
    np.testing.assert_allclose(res, expected, atol=1e-12)


def main():
    test_residual()

if __name__ == "__main__":
    main()