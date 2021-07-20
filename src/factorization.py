import networkx as nx
import numpy as np
from scipy.sparse.linalg import svds, eigsh
import utils

def adjacency_matrix(G):
    """
    Adjacency matrix for the input graph.

    :param G: Input graph.
    :return: Adjacency matrix.
    """
    return nx.adjacency_matrix(G, sorted(G.nodes)).toarray()


def degree_vector(G):
    """
    Degree vector for the input graph.

    :param G: Input graph.
    :return: Degree vector.
    """
    return np.array([a[1] for a in sorted(G.degree(weight='weight'), key=lambda a: a[0])])


def out_degree_vector(G):
    """
    Out-degree vector for the input graph.

    :param G: Input graph.
    :return: Out-degree vector.
    """
    return np.array([a[1] for a in sorted(G.out_degree(weight='weight'), key=lambda a: a[0])])


def standard_random_walk_transition_matrix(G):
    """
    Transition matrix for the standard random-walk given the input graph.

    :param G: Input graph.
    :return: Standard random-walk transition matrix.
    """

    D_1 = np.diag(1 / degree_vector(G))
    A = adjacency_matrix(G)
    return np.asarray(np.matmul(D_1, A))


def pagerank_transition_matrix(G, mu=0.15):
    """
    Transition matrix for the PageRank given the input graph.

    :param G: Input graph.
    :param mu: teleportation probability.
    :return: PageRank transition matrix.
    """

    d = out_degree_vector(G).astype(float)
    d_1 = np.divide(np.ones_like(d), d, out=np.zeros_like(d), where=d!=0)  # Set inverse of zero degree to zero.
    M_std = np.diag(d_1) @ adjacency_matrix(G)  # Standard random walk transition matrix.

    a = (d == 0).astype(int)  # Dangling node (out degree = 0) indicator.
    J = np.ones_like(M_std)
    n = G.number_of_nodes()
    M_pr = (1 - mu) * M_std + mu * J / n + (1 - mu) * np.diag(a) @ J / n

    return M_pr


def stationary_distribution(M):
    """
    Stationary distribution given the transition matrix.

    :param M: Transition matrix.
    :return: Stationary distribution.
    """

    # We solve (M^T - I) pi = 0 and 1 pi = 1. Combine them and let A = [M^T - I; 1], b = [0; 1]. We have A pi = b.
    n = M.shape[0]
    A = np.concatenate([M.T - np.identity(n), np.ones(shape=(1,n))], axis=0)
    b = np.concatenate([np.zeros(n), [1]], axis=0)

    # Solve A^T A pi = A^T pi instead (since A is not square).
    pi = np.linalg.solve(A.T @ A, A.T @ b)

    return pi


def autocovariance_matrix(M, tau, b=1):
    """
    Autocovariance matrix given a transition matrix. Pi M^tau/b -pi pi^T

    :param M: Transition matrix.
    :param tau: Markov time.
    :param b: Number of negative samples used in the sampling algorithm.
    :return: Autocovariance matrix.
    """

    pi = stationary_distribution(M)
    Pi = np.diag(pi)
    M_tau = np.linalg.matrix_power(M, tau)

    return Pi @ M_tau/b - np.outer(pi, pi)


def average_autocovariance_matrix(M, tau, b=1):
    """
    Average autocovariance matrix given a transition matrix. Pi sum(M^tau)/b -pi pi^T

    :param M: Transition matrix.
    :param tau: Markov time.
    :param b: Number of negative samples used in the sampling algorithm.
    :return: Average autocovariance matrix.
    """
    pi = stationary_distribution(M)
    Pi = np.diag(pi)

    M_tau = np.identity(M.shape[0])
    M_tau_sum = np.zeros(M.shape)
    for tau_ in range(1, tau+1):
        M_tau = M_tau @ M
        M_tau_sum = M_tau_sum + M_tau

    return Pi @ (M_tau_sum/tau)/b - np.outer(pi, pi)


def PMI_matrix(M, tau, b=1):
    """
    PMI matrix given a transition matrix. log(Pi sum(M^tau)/b) - log(pi pi^T)

    :param M: transition matrix
    :param tau: Markov time
    :param b: Number of negative samples used in the sampling algorithm.
    :return: PMI matrix.
    """

    pi = stationary_distribution(M)
    Pi = np.diag(pi)
    M_tau = np.linalg.matrix_power(M, tau)

    return np.log(Pi @ M_tau/b) - np.log(np.outer(pi, pi))


def average_PMI_matrix(M, tau, b=1):
    """
    Log-mean-exp PMI matrix given a transition matrix.. log (Pi sum(M^tau)/b) - log(pi pi^T)

    :param M: transition matrix
    :param tau: Markov time
    :param b: Number of negative samples used in the sampling algorithm.
    :return: Log-mean-exp PMI matrix.
    """

    pi = stationary_distribution(M)
    Pi = np.diag(pi)

    M_tau = np.identity(M.shape[0])
    M_tau_sum = np.zeros(M.shape)
    for tau_ in range(1, tau+1):
        M_tau = M_tau @ M
        M_tau_sum = M_tau_sum + M_tau

    return np.log(Pi @ (M_tau_sum/tau)/b) - np.log(np.outer(pi, pi))


def preprocess_similarity_matrix(R):
    """
    Preprocess the similarity matrix.

    :param R: Similarity matrix.
    :return: Preprocessed similarity matrix.
    """

    R = R.copy()

    # Replace nan with 0 and negative infinity with min value in the matrix.
    R[np.isnan(R)] = 0
    R[np.isinf(R)] = np.inf
    R[np.isinf(R)] = R.min()

    return R


def postprocess_decomposition(u, s, v=None):
    """
    Postprocess the decomposed vectors and values into final embeddings.

    :param u: Eigenvectors (or left singular vectors)
    :param s: Eigenvalues (or singular values)
    :param v: Right singular vectors.
    :return: Embeddings.
    """

    dim = len(s)

    # Weight the vectors with square root of values.
    for i in range(dim):
        u[:, i] *= np.sqrt(s[i])
        if v is not None:
            v[:, i] *= np.sqrt(s[i])

    # Unify the sign of vectors for reproducible results.
    for i in range(dim):
        if u[0, i] < 0:
            u[:, i] *= -1
            if v is not None:
                v[:, i] *= -1

    # Rescale the embedding matrix.
    if v is not None:
        return utils.rescale_embeddings(u), utils.rescale_embeddings(v)
    else:
        return utils.rescale_embeddings(u)


def embed(G, dim, tau, directed, similarity, average_similarity):
    """
    Embed the graph with the matrix factorization algorithm.

    :param G: Input graph.
    :param dim: Dimensions of embedding.
    :param tau: Markov time.
    :param directed: Whether the graph is directed.
    :param similarity: Similarity metric.
    :param average_similarity: Whether to use the average version of similarity metric.
    :return: Embeddings of shape (num_nodes, dim)
    """

    # Infer random walk process from the graph type.
    if directed:
        M = pagerank_transition_matrix(G)
    else:
        M = standard_random_walk_transition_matrix(G)

    # Select the similarity metric.
    if similarity == 'autocovariance':
        if average_similarity:
            R = average_autocovariance_matrix(M, tau)
        else:
            R = autocovariance_matrix(M, tau)
    elif similarity == 'PMI':
        if average_similarity:
            R = average_PMI_matrix(M, tau)
        else:
            R = PMI_matrix(M, tau)
    else:
        raise NotImplementedError(f'Similarity metric {similarity} not implemented. ')

    # Compute the embedding by decomposition.
    R = preprocess_similarity_matrix(R)
    if directed:  # Singular value decomposition.
        u, s, vt = svds(A=R, k=dim)
        v = vt.T
        u, v = postprocess_decomposition(u, s, v)
        return u, v
    else:  # Eigenvalue decomposition.
        s, u = eigsh(A=R, k=dim, which='LA', maxiter=R.shape[0] * 20)
        u = postprocess_decomposition(u, s)
        return u

