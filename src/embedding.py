import argparse
from ast import literal_eval
import networkx as nx
import numpy as np
import factorization
import sampling
import utils


def parse_args():
    """
    Parse the embedding arguments.

    :return: Parsed arguments
    """

    parser = argparse.ArgumentParser(description="Embedding")

    parser.add_argument('--graph', default='graph/blogcatalog.edges',
                        help='Input graph edgelist. Default is "graph/blogcatalog.edges". ')

    parser.add_argument('--embeddings', default='emb/blogcatalog.embeddings',
                        help='Output embeddings for undirected graphs or source embeddings for directed graphs. '
                             'Default is "emb/blogcatalog.embeddings". ')

    parser.add_argument('--target-embeddings', default='emb/cora.target-embeddings',
                        help='Output target embeddings for directed graphs. '
                             'Default is "emb/cora.target-embeddings".')

    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--markov-time', type=int, default=3,
                        help='Markov time scale. Default is 3.')

    parser.add_argument('--weighted', type=literal_eval, default=False,
                        help='Whether the graph is weighted. Default is False. ')

    #Sampling parameters (gradient descent and random walks)

    #Gradient descent
    parser.add_argument('--lr', type=float, default=6e-3,
                        help='Learning rate for gradient descent. Default is 6e-3.')
    
    parser.add_argument('--iter', type=int, default=50,
                        help='Max iterations of gradient descent. Default is 200.')
    
    parser.add_argument('--early-stop', type=int, default=5,
                        help='Iterations before early stop of gradient descent. Default is 5.')
    
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Random walk length. Default is 100.')
    
    #Random walks
    parser.add_argument('--neg', type=int, default=1,
                        help='Number of negative samples. Default is 1.')
    
    parser.add_argument('--walks', type=int, default=10,
                        help='Random walks per node. Default is 10.')
    
    parser.add_argument('--walk-length', type=int, default=80,
                        help='Random walk length. Default is 80.')
    
    parser.add_argument('--damp', type=float, default=0.85,
                        help='Pagerank damping factor. Default is 0.85')
    
    parser.add_argument('--workers', type=int, default=32,
                        help='Number of workers for node2vec (sampling PMI). Default is 4.')
    
    # Process parameters.
    parser.add_argument('--directed', type=literal_eval, default=False,
                        help='Whether the graph is directed. Determines the random-walk process. '
                             'Standard random-walk is used for undirected graphs, while PageRank is used for directed graphs. '
                             'Default is False.')

    # Similarity parameters.
    parser.add_argument('--similarity', default='autocovariance',
                        help='Type of similarity metric. Default is "autocovariance". Other option: "PMI". ')

    parser.add_argument('--average-similarity', type=literal_eval, default=False,
                        help='Whether to use the average version of the similarity metrics. Default is False. ')

    # Algorithm parameters.
    parser.add_argument('--algorithm', default='factorization',
                        help='Type of embedding algorithm. Default is "factorization". Other option: "sampling".')

    return parser.parse_args()


def main():
    """
    Pipeline for random-walk embedding.
    """
    args = parse_args()
    G = utils.read_graph(args.graph, args.directed, args.weighted)
    if args.algorithm == 'factorization':
        emb = factorization.embed(G, args.dimensions, args.markov_time, args.directed, args.similarity, args.average_similarity)
    elif args.algorithm == 'sampling':
        emb = sampling.embed(G, args.dimensions, args.markov_time, None, args.directed, args.similarity, args.average_similarity,
                args.lr, args.iter, args.early_stop, args.batch_size, args.neg, args.walks, args.walk_length, args.damp, args.workers)
    else:
        raise NotImplementedError(f'Embedding algorithm {args.algorithm} not implemented. ')

    if args.directed or (args.algorithm == 'sampling' and args.similarity == 'autocovariance'):
        source_emb, target_emb = emb
        np.savetxt(args.embeddings, source_emb, fmt='%.16f')
        np.savetxt(args.target_embeddings, target_emb, fmt='%.16f')
    else:
        np.savetxt(args.embeddings, emb, fmt='%.16f')

    print(f'Embedding done.')

if __name__ == "__main__":
    main()
