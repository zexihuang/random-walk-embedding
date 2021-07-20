import argparse
from ast import literal_eval
import numpy as np
import random
import networkx as nx
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import linear_kernel

import utils


def parse_args():
    """
    Parse the evaluating arguments.

    :return: Parsed arguments
    """

    parser = argparse.ArgumentParser(description="Evaluating")

    parser.add_argument('--task', default='node-classification',
                        help='Type of downstream task. Default is "node-classification". '
                             'Other options: "link-prediction", "community-detection".')

    parser.add_argument('--embeddings', default='emb/blogcatalog.embeddings',
                        help='Input node embeddings.'
                             'Default is "emb/blogcatalog.embeddings". ')

    # Node classification parameters.
    parser.add_argument('--labels', default='graph/blogcatalog.labels',
                        help='Input ground-truth labels for node classification. Default is "graph/blogcatalog.labels".')

    parser.add_argument('--multi-label', type=literal_eval, default=True,
                        help='Whether predict multi-label (True) or multi-class (False) in node classification. '
                             'Default is True. ')

    parser.add_argument('--training-ratio', type=float, default=0.5,
                        help='Training ratio in node classification. Default is 0.5. ')

    parser.add_argument('--repetitions', type=int, default=10,
                        help='Number of repetitions in node classification. Default is 10. ')

    # Community detection parameters.
    parser.add_argument('--communities', default='graph/airport.country-labels',
                        help='Input ground-truth communities for community detection. '
                             'Default is "graph/airport.country-labels". ')

    # link prediction parameters.
    parser.add_argument('--mode', default='prepare',
                        help='Mode of link prediction evaluation. '
                             'Options: "prepare" (remove edges in the graph and generate residual graph for embedding), '
                             '"evaluate" (evaluate link prediction performance based on embeddings). '
                             'Default is "prepare". ')

    parser.add_argument('--graph', default='graph/blogcatalog.edges',
                        help='Input graph edgelist for edge removal in link prediction preparation. '
                             'Default is "graph/blogcatalog.edges". ')

    parser.add_argument('--weighted', type=literal_eval, default=False,
                        help='Whether the graph is weighted. Default is False. ')

    parser.add_argument('--remove-ratio', type=float, default=0.2,
                        help='Ratio of removed edges in link prediction preparation. Default is 0.2. ')

    parser.add_argument('--remaining-edges', default='graph/blogcatalog.remaining-edges',
                        help='Remaining edges after removal in link prediction preparation. '
                             'Serve as output when mode is "prepare" and input when mode is "evaluate". '
                             'Default is "graph/blogcatalog.remaining-edges". ')

    parser.add_argument('--removed-edges', default='graph/blogcatalog.removed-edges',
                        help='Removed edges in link prediction preparation. '
                             'Serve as output when mode is "prepare" and input when mode is "evaluate". '
                             'Default is "graph/blogcatalog.removed-edges". ')

    parser.add_argument('--k', type=float, default=1.0,
                        help='Ratio of top ranking pairs over removed edges in link prediction evaluation. '
                             'Default is 1.0. ')

    return parser.parse_args()


def convert_label_format(num_nodes, labels, multi_label=False):
    """
    Convert the node labels into multi-label/multi-class format.

    :param num_nodes: Number of nodes.
    :param labels: Input labels.
    :param multi_label: Whether convert to multi-label (True) or multi-class (False) format.
    :return: Converted labels.
    """

    if multi_label:  # Convert labels to multi-label format.
        label_list = [[] for _ in range(num_nodes)]
        for node, label in labels:
            label_list[node-1].append(label)
        mlb = MultiLabelBinarizer()
        labels = mlb.fit_transform(label_list)
    else:  # Convert labels to multi-class format.
        label_list = [0] * num_nodes
        for node, label in labels:
            label_list[node-1] = label
        labels = np.asarray(label_list)

    return labels


def node_classification(emb, labels, multi_label, training_ratio, repetitions, cv_random_state=0):
    """
    Multi-class prediction with the embedding and the labels.

    :param emb: Node embeddings.
    :param labels: Node labels.
    :param multi_label: Whether predict multi-label (True) or multi-class (False).
    :param training_ratio: Proportion of training data.
    :param repetitions: Number of repetitions.
    :param cv_random_state: Random state for reproducible cross validation.
    :return: (micro_f1, macro_f1) score.
    """

    labels = convert_label_format(len(emb), labels, multi_label)

    ovr_classifier = OneVsRestClassifier(LogisticRegression(solver='liblinear'))

    micro_f1s = []
    macro_f1s = []
    for repetition in range(repetitions):
        X_train, X_test, y_train, y_test = train_test_split(emb, labels, train_size=training_ratio, random_state=cv_random_state+repetition)
        ovr_classifier.fit(X_train, y_train)
        if multi_label:  # Predict the same number of top-score labels as true labels.
            y_pred = np.zeros_like(y_test, dtype=int)
            y_number = y_test.sum(axis=1)
            y_proba = ovr_classifier.predict_proba(X_test)
            for i in range(len(y_proba)):
                top_labels = np.argpartition(y_proba[i], -y_number[i])[-y_number[i]:]
                y_pred[i, top_labels] = 1
        else:  # Predict the label with the highest score.
            y_pred = ovr_classifier.predict(X_test)
        micro_f1 = f1_score(y_test, y_pred, average='micro', zero_division=0)
        macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        micro_f1s.append(micro_f1)
        macro_f1s.append(macro_f1)

    return np.asarray(micro_f1s).mean(), np.asarray(macro_f1s).mean()


def community_detection(emb, comms, cluster_random_state=0):
    """
    Community detection with the embedding and the ground-truth community labels.

    :param emb: Node embeddings.
    :param comms: Node community labels.
    :param cluster_random_state: Random state for reproducible clustering initialization.
    :return: Normalized mutual information score.
    """

    comms = convert_label_format(len(emb), comms)
    n_clusters = len(set(comms))
    cluster = KMeans(n_clusters=n_clusters, random_state=cluster_random_state)
    pred = cluster.fit_predict(emb)
    return normalized_mutual_info_score(comms, pred)


def link_prediction_preparation(G, p, weighted, shuffle_random_state=0):
    """
    Remove edges in the graph and generate residual graph for embedding.

    :param G: Input graph.
    :param p: Ratio of removed edges.
    :param weighted: Whether the graph is weighted.
    :param shuffle_random_state: Random state for reproducible edge removal.
    :return: (remaining_edges, removed_edges)
    :raise ValueError: if the given ratio of edges can't be removed while maintaining connectivity.
    """

    # Reformat the set of edges as list of tuples with edge[0] <= edge[1]
    all_edges = list(map(tuple, np.sort(np.asarray(G.edges()))))

    remaining_edges = all_edges
    removed_edges = []
    added_back_edges = []
    num_edges = len(all_edges)
    random.Random(shuffle_random_state).shuffle(remaining_edges)
    residual_G = G.copy()

    while len(removed_edges) < num_edges * p:
        if len(remaining_edges) == 0:
            raise ValueError(f'The given ratio of edges {p} cannot be removed while maintaining connectivity. ')
        edge = remaining_edges.pop()
        edge_data = residual_G.edges[edge]
        residual_G.remove_edge(edge[0], edge[1])
        if nx.is_connected(residual_G):  # Remove the edge if the graph is still connected.
            removed_edges.append(edge)
        else:  # Otherwise, add that edge back.
            residual_G.add_edge(edge[0], edge[1], **edge_data)
            added_back_edges.append(edge)

    remaining_edges.extend(added_back_edges)

    if weighted:  # Append the edge weight.
        remaining_edges = [(edge[0], edge[1], G.edges[edge]['weight']) for edge in remaining_edges]
        removed_edges = [(edge[0], edge[1], G.edges[edge]['weight']) for edge in removed_edges]

    return np.asarray(remaining_edges), np.asarray(removed_edges)


def link_prediction(emb, removed_edges, remaining_edges, k):
    """
    Link prediction with the embedding based on the residual graph.

    :param emb: Node embeddings.
    :param removed_edges: Edges that have been removed from the graph.
    :param remaining_edges: Edges that are still present in the graph.
    :param k: Ratio of top ranking pairs over removed edges.
    :return: precision@k score.
    """

    # Reformat the edges.
    removed_edges = removed_edges[:, :2].astype(int) - 1
    remaining_edges = remaining_edges[:, :2].astype(int) - 1

    # Compute the actual k in terms of number of edges.
    k = int(np.ceil(len(removed_edges) * k))

    # Reconstruct similarity by inner product of embeddings.
    similarity = linear_kernel(emb, emb)

    # Set the similarity of existing edges and self-loops to be negative infinity to remove them from ranking.
    similarity[np.tril_indices_from(similarity)] = -np.inf
    similarity[tuple(remaining_edges.T)] = -np.inf

    # Rank the top pairs.
    index = np.argpartition(similarity.ravel(), -k)[-k:]  # Find the top candidates.
    index = index[np.argsort(-similarity[np.unravel_index(index, similarity.shape)])]  # Sort the top candidates
    predicted_edges = np.asarray(np.unravel_index(index, similarity.shape)).T

    # Compute the precision@k score.
    return len(set(map(tuple, removed_edges)).intersection(set(map(tuple, predicted_edges)))) / k


def main():
    """
    Pipeline for evaluating embeddings.
    """
    args = parse_args()

    if args.task == 'node-classification':
        emb = np.loadtxt(args.embeddings)
        labels = np.loadtxt(args.labels, dtype=int)
        micro_f1, macro_f1 = node_classification(emb, labels, args.multi_label, args.training_ratio, args.repetitions)
        print(f'Micro-F1: {micro_f1:.4f}, Macro-F1: {macro_f1:.4f}')

    elif args.task == 'link-prediction':
        if args.mode == 'prepare':  # Remove a proportion of edges in the original graph for embedding.
            G = utils.read_graph(args.graph, False, args.weighted)
            remaining_edges, removed_edges = link_prediction_preparation(G, args.remove_ratio, args.weighted)
            fmt = ['%d', '%d', '%f'] if args.weighted else '%d'
            np.savetxt(args.remaining_edges, remaining_edges, fmt=fmt)
            np.savetxt(args.removed_edges, removed_edges, fmt=fmt)
        elif args.mode == 'evaluate':  # Evaluate link prediction performance based on embeddings of the residual graph.
            emb = np.loadtxt(args.embeddings)
            removed_edges = np.loadtxt(args.removed_edges)
            remaining_edges = np.loadtxt(args.remaining_edges)
            prec = link_prediction(emb, removed_edges, remaining_edges, args.k)
            print(f'Precision @ {args.k:.0%}: {prec:.4f}')
        else:
            raise NotImplementedError(f'Link prediction mode {args.mode} not implemented. ')

    elif args.task == 'community-detection':
        emb = np.loadtxt(args.embeddings)
        comms = np.loadtxt(args.communities, dtype=int)
        nmi = community_detection(emb, comms)
        print(f'Normalized Mutual Information: {nmi:.4f}.')

    else:
        raise NotImplementedError(f'Task {args.task} not implemented. ')


if __name__ == "__main__":
    main()