import numpy as np
import networkx as nx
from sklearn.preprocessing import scale


def read_graph(path, directed, weighted):
    """
    Read the graph in the networkx format.

    :param directed: Whether the graph is directed.
    :param weighted: Whether the graph is weighted.
    :return: networkx graph
    """
    create_using = nx.DiGraph if directed else nx.Graph
    data = (("weight", float),) if weighted else False
    G = nx.read_edgelist(path, nodetype=int, data=data, create_using=create_using)

    return G


def rescale_embeddings(u):
    """
    Rescale the embedding matrix by mean removal and variance scaling.

    :param u: Embeddings.
    :return: Rescaled embeddings.
    """
    shape = u.shape
    scaled = scale(u.flatten())
    return np.reshape(scaled, shape)
