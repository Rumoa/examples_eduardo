import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.linalg import expm, sqrtm
from functools import partial

from dask import compute, delayed


def norm_laplacian_matrix(dim, p):
    # Initialize flag for loop control
    flag = 0
    # Loop until a valid matrix is generated
    while flag == 0:
        # Number of nodes in the hypercube
        n = 2**dim

        nodes = np.arange(n)
        binary_repr = np.array(
            [list(format(i, f"0{dim}b")) for i in nodes], dtype=int
        )
        adjacency_matrix = (
            (
                np.sum(
                    np.abs(binary_repr[:, None, :] - binary_repr[None, :, :]),
                    axis=2,
                )
                == 1
            )
        ).astype(int)
        degree_matrix = np.zeros((n, n), dtype=int)
        # Generate edges with probability (1-p)
        for i in range(n - 1):
            for j in range(i + 1, n):
                if random.uniform(0, 1) <= (1 - p):
                    adjacency_matrix[i][j] = 0
                    adjacency_matrix[j][i] = 0

        # Calculate degree matrix
        for i in range(n):
            degree_sum = np.sum(adjacency_matrix[i])
            degree_matrix[i][i] = degree_sum

            # Check for isolated nodes
            if degree_matrix[i][i] == 0:
                flag -= 1

        # If no isolated nodes, calculate the Laplacian matrix
        if flag == 0:
            degree_matrix = np.linalg.inv(degree_matrix)
            degree_matrix = sqrtm(degree_matrix)

        flag += 1

    # Return Laplacian matrix
    return np.eye(n) - degree_matrix @ adjacency_matrix @ degree_matrix


def get_gamma(w, l, n):
    eigenvalues, eigenvectors = np.linalg.eigh(l)
    gamma = 0

    for i in range(1, n):
        if not np.isclose(eigenvalues[i], 0, atol=1e-3):
            gamma += (np.abs(np.vdot(w, eigenvectors[:, i])) ** 2) / (
                eigenvalues[i]
            )

    return gamma


def change_matrix(dim, p, w):
    l = norm_laplacian_matrix(dim, p)
    gamma = get_gamma(w, l, n)
    h = gamma * l - w @ w.transpose()
    return h


def ctqw_search(tau, dim, p, w, t_max, t_steps):
    n = 2**dim
    t = t_max / t_steps
    phi = (1 / np.sqrt(n)) * np.ones((n, 1), dtype=int)
    success_probabilities = []
    sum_time = 0
    h = change_matrix(dim, p, w)

    for _ in range(t_steps):
        # Prepare exp(-1j*H_tot*t)
        h_tot = h
        # Each Hamiltonian can only evolve tau time.
        if sum_time + t >= tau:
            x = tau - sum_time
            h_tot *= x
            n_hamiltonians = int(np.ceil((t - x) / tau))

            for n in range(n_hamiltonians):
                if n == n_hamiltonians - 1:
                    sum_time = t - x - n * tau
                    h = change_matrix(dim, p, w)
                    h_tot += h * sum_time
                else:
                    h = change_matrix(dim, p, w)
                    h_tot += h * tau
            # Evolve and get success probability
            evolution = expm(-1j * h_tot)
            phi_tplus1 = evolution @ phi
            success_probabilities.append(abs(np.vdot(w, phi_tplus1)) ** 2)
            # print(success_probabilities[-1])
            phi = phi_tplus1
        else:
            sum_time += t
            evolution = expm(-1j * h_tot * t)
            phi_tplus1 = evolution @ phi
            success_probabilities.append(abs(np.vdot(w, phi_tplus1)) ** 2)
            # print(success_probabilities[-1])
            phi = phi_tplus1

    # t_values = np.linspace(t, t_max, t_steps)
    runtimes = [i * t / success_probabilities[i] for i in range(1, t_steps)]
    min_runt = np.min(runtimes)
    return success_probabilities, min_runt

    plt.figure()
    plt.plot(t_values, success_probabilities)
    plt.xlabel("Time")
    plt.ylabel("Success Probability")
    plt.title("Success Probability vs. Time")
    plt.grid(True)
    plt.show()

    print(min_runt)
    return min_runt


if __name__ == "__main__":
    dim = 8
    p = 0.9
    n = 2**dim
    t_max = np.pi * np.sqrt(n)
    t_steps = 100
    tau = 1

    tau_list = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
    w = np.zeros((n, 1), dtype=int)
    w[0][0] = 1
    f = partial(ctqw_search, dim=dim, p=p, w=w, t_max=t_max, t_steps=t_steps)

    delayed_results = [delayed(f)(tau_i) for tau_i in tau_list]
    results = compute(*delayed_results, scheduler="processes")

    # print(f(tau_array[0]))
    # print(results)


# print(ctqw_search(tau, dim, p, w, t_max, t_steps))

# f = ctqw_search()
# [ f(i) for i in tau_array]
