import networkx as nx
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import time
from node2vec import Node2Vec
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sys
import multiprocessing
from multiprocessing import Process, Queue
import utils

os.environ["CUDA_VISIBLE_DEVICES"]="1"

EPS = 1e-16

class ProcSampler(object):
    '''
        Generic sampler for a random walk process

        Usage:
            start_prob = stat_dist_undirected(G)
            sampler = RWSampler(G, start_prob, 2, 2)
            walks = sampler.sample_walks(1)
            sampler.get_pos_samples(walks)
    '''
    def __init__(self, G, window, workers=1, avg=False, weight=None):
        '''
        '''
        self.G = G
        self.window = window
        self.avg = avg
        self.weight = weight
        self.workers = workers

    def sample_walks(self, n):
        '''
        '''
        raise NotImplementedError

class RWSampler(ProcSampler):
    '''
        Generic class for random walk sampling.
    '''
    def __init__(self, G, start_prob, length, window, workers=1, avg=False, weight=None):
        '''
            :param G: input graph
            :param start_prob: vertex starting probabilities
            :param length: number of steps taken in the process
            :param window: window for generating positive samples
            :param workers: number of workers for parallel processing
            :param avg: whether all pairs within window are considered
            :param weight: edge weights
        '''
        super().__init__(G, window, workers, avg, weight)
        self.start_prob = start_prob
        self.length = length

    def sample_walk(self, length, start, weight=None):
        '''
        '''
        raise NotImplementedError

    def sample_walks_node_list(self, res, nodes, n):
        '''
            Samples walks starting from a particular node list

            :param res: results
            :param nodes: node list
            :param n: number of walks
        '''
        walks = []    
        for i in range(n):
            for v in nodes:
                w = self.sample_walk(self.length, v, self.weight)
                walks.append(w)

        res.extend(walks)

    def sample_walks(self, n):
        '''
            Samples walks of lenght t with starting node probability.

            :param n: number of walks
        '''
        walks = []

        
        if self.workers <= 1:
            for i in range(n):
                for v in self.G.nodes():
                    w = self.sample_walk(self.length, v, self.weight)

                    walks.append(w)
        else:
            nodes_per_thread = []
            results = []
            for t in range(self.workers):
                nodes_per_thread.append([])
                results.append(multiprocessing.Manager().list())
            
            i = 0
            for v in self.G.nodes():
                nodes_per_thread[i % self.workers].append(v)
                i = i + 1

            processes = [Process(target=self.sample_walks_node_list, args=(results[t], nodes_per_thread[t], n)) for t in range(self.workers)]
            
            for p in processes:
                p.start()

            
            for p in processes:
                p.join()
        
            for t in range(self.workers):
                walks.extend(results[t])

        return walks

    def get_pos_samples_seq(self, walks, sym=False):
        '''
            Generates positive samples as two lists [u] [v]
            
            :param walks: random walks
            :param sym: whether samples are symmetric or not
        '''
        pos_u = []
        pos_v = []

        for w in walks:
            if self.avg is True:
                for j in range(len(w)-self.window+1):
                    for r in range(1, self.window+1):
                        if j+r < len(w):
                            u = w[j]-1
                            v = w[j+r]-1
                            pos_u.append(u)
                            pos_v.append(v)

                            if sym:
                                pos_u.append(v)
                                pos_v.append(u)
            else:
                for j in range(len(w)-self.window):
                    u = w[j]-1
                    v = w[j+self.window]-1
                    pos_u.append(u)
                    pos_v.append(v)

                    if sym:
                        pos_u.append(v)
                        pos_v.append(u)

        return pos_u, pos_v
            
    def get_pos_samples_worker(self, res, walks, sym):
        '''
            Generates positive samples from walks (for parallel processing).

            :param res: results
            :param walks: random walks
            :param sym: whether samples are symmetric or not
        '''
        pos_u, pos_v = self.get_pos_samples_seq(walks, sym)

        for u in pos_u:
            res[0].append(u)

        for v in pos_v:
            res[1].append(v)
    
    def get_pos_samples(self, walks, sym=False):
        '''
            Generates positive samples from walks.
            
            :param walks: set of walks as a list of lists
            :param sym: whether samples are symmetric or not
        '''
        
        if self.workers <=1:
            #sequential
            return self.get_pos_samples_seq(walks, sym)
        else:
            #parallel
            nodes_per_thread = []
            pos_u = []
            pos_v = []
            
            results = []
            walks_per_worker = []
            for t in range(self.workers):
                results.append([multiprocessing.Manager().list(), multiprocessing.Manager().list()])
                walks_per_worker.append([])
            
            i = 0
            for w in walks:
                walks_per_worker[i % self.workers].append(w)
                i = i + 1

            processes = [Process(target=self.get_pos_samples_worker, args=(results[t], walks_per_worker[t], sym)) for t in range(self.workers)]
            
            for p in processes:
                p.start()

            for p in processes:
                p.join()
        
            for t in range(self.workers):
                pos_u.extend(results[t][0])
                pos_v.extend(results[t][1])

            return pos_u, pos_v

    def get_neg_samples(self, pos, k=1):
        '''
            Generates n*k negative samples as an (n,k) matrix

            :param pos: list of positive samples
            :param k: number of negative samples per positive sample
        '''
        nodes = list(np.arange(self.G.number_of_nodes()))
        prob = np.array(self.start_prob)
        
        return np.random.choice(nodes, size=(len(pos), k), p=prob)

class StdRWSampler(RWSampler):
    '''
        Standard random-walk sampler.
    '''
    def __init__(self, G, start_prob, length, window, workers=1, avg=False, weight=None):
        '''
            :param G: input graph
            :param start_prob: vertex starting probabilities
            :param length: number of steps taken in the process
            :param window: window for generating positive samples
            :param workers: number of workers for parallel processing
            :param avg: whether all pairs within window are considered
            :param weight: edge weights
        '''
        super().__init__(G, start_prob, length, window, workers, avg, weight)

    def get_pos_samples(self, walks):
        '''
            Generates positive samples from walks.

            :param walks: set of walks as a list of lists
        '''
        return super().get_pos_samples(walks, sym=True)

    def sample_walk(self, length, start, weight=None):
        '''
            Samples a single walk of length t from start.

            :param length: random-walk length
            :param start: starting node
            :param weight: edge weights
        '''
        walk = [start]
        v = start

        if weight is None:
            for i in range(length):
                neighbs = list(self.G.neighbors(v))

                v = np.random.choice(neighbs)
                walk.append(v)

        else:
            for i in range(length):
                neighbs = list(self.G.neighbors(v))
                prob = np.zeros(len(neighbs))

                for u in range(len(neighbs)):
                    prob[u] = self.G.edges[(v,u)]['weight']

                v = np.random.choice(neighbs, p=prob)
                walk.append(v)

        return walk

class PRSampler(RWSampler):
    '''
        Pagerank sampler
    '''
    def __init__(self, G, start_prob, length, window, workers, avg=False, weight=None, alpha=0.85):
        '''
            :param G: input graph
            :param start_prob: vertex starting probabilities
            :param length: number of steps taken in the process
            :param window: window for generating positive samples
            :param workers: number of workers for parallel processing
            :param avg: whether all pairs within window are considered
            :param weight: edge weights
            :param alpha: teleportation probability
        '''
        super().__init__(G, start_prob, length, window, workers, avg, weight)
        self.alpha = alpha

    def get_pos_samples(self, walks):
        '''
            Generates positive samples from walks.

            :param walks: set of walks as a list of lists
        '''
        return super().get_pos_samples(walks,sym=False)

    def sample_walk(self, length, start, weight=None):
        '''
            Samples a single walk of length t from start.

            :param length: random-walk length
            :param start: starting vertex
        '''
        walk = [start]
        v = start

        for i in range(length):
            r = np.random.random()
            neighbs = list(self.G.neighbors(v))

            if r < self.alpha and len(neighbs) > 0:
                neighbs = list(self.G.neighbors(v))

                if weight is None:
                    v = np.random.choice(neighbs)
                else:
                    prob = np.zeros(len(neighbs))

                    for u in range(len(neighbs)):
                        prob[u] = self.G.edges[(v,u)]['weight']

                    v = np.random.choice(neighbs, p=prob)

            else:
                v = np.random.choice(neighbs)

            walk.append(v)

        return walk

def stat_dist_undirected(G, weight=None):
    '''
        Stationary distribution for undirected graph.
        
        :param G: Input graph
        :param weight: edge weights
    '''
    deg = np.array([a[1] for a in sorted(G.degree(weight='weight'), key=lambda a: a[0])])
    
    return deg / deg.sum()

def pagerank(G, alpha=0.85, weight=None):
    '''
        Computes pagerank for all vertices in G.

        :param G: Input graph
        :param alpha: teleportation probability
        :param weight: edge weights
    '''
    stat_dist = np.zeros(G.number_of_nodes())
    pr = nx.pagerank(G, alpha=alpha, weight=weight)

    i = 0
    for v in sorted(G.nodes()):
        stat_dist[i] = pr[v]
        i = i + 1

    return stat_dist

class SampleEmbedding(nn.Module):
    '''
        Generic class for sampling based embedding
    '''
    def __init__(self, G, sampler, n_samples=100, n_dim=100, learning_rate=0.025,
                 batch_size=50, n_iter=10, n_neg_samples=1, early_stop=10, momentum=0.99, patience=1):
        '''
            :param G: Input graph
            :param sampler: process sampler
            :param walks: number of random walks per node
            :param dim: Dimensions of embedding
            :param lr: learning rate for SGD
            :param batch_size: size of batches for SGD
            :param n_iter: max number of iterations for SGD
            :param neg: number of negative samples
            :param early_stop: number of iterations before early stop
            :param penalty: penalizes embeddings out of probability range
            :param momentum: Pytorch optimizer parameters
            :param patience: Pytorch optimizer parameters
        '''
        super(SampleEmbedding, self).__init__()
        self.sampler = sampler
        self.G = G
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.n_neg_samples = n_neg_samples
        self.n_samples = n_samples
        self.n_dim = n_dim
        self.emb = self.init_emb(n_dim)
        self.early_stop = early_stop
        self.momentum = momentum
        self.patience = patience

        self.tmp_model_file = "samp_emb"+str(int(time.time()))+".pt"

        self.use_cuda = torch.cuda.is_available()

        if self.use_cuda:
            self.cuda()

        self.optimizer = optim.SparseAdam(list(self.parameters()), lr=self.learning_rate, betas=(self.momentum, 0.999))
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=self.patience, min_lr=1e-10, verbose=True)

    def init_emb(self):
        '''
            Initializes embedding.

            :param dim: Dimensions of embedding
        '''
        n = self.G.number_of_nodes()
        self.u_embed = nn.Embedding(n, self.n_dim, sparse=True)
        self.v_embed = nn.Embedding(n, self.n_dim, sparse=True)

    def train(self, verbose=False):
        '''
            Learns embedding using batched SGD.
        '''
        batches = []

        walks = self.sampler.sample_walks(self.n_samples)

        pos_u, pos_v = self.sampler.get_pos_samples(walks)
        neg_v = self.sampler.get_neg_samples(pos_u, self.n_neg_samples)

        n_batches = int(np.floor(len(pos_u) / self.batch_size))
        
        for b in range(n_batches):
            i = b * self.batch_size
            j = (b+1) * self.batch_size
            batches.append([pos_u[i:j], pos_v[i:j], neg_v[i:j]])

        losses = []
        best_loss = sys.float_info.max

        for i in range(self.n_iter):
            sum_loss = 0
            for b in batches:
                pos_u, pos_v, neg_v = b

                pos_u = Variable(torch.LongTensor(pos_u))
                pos_v = Variable(torch.LongTensor(pos_v))
                neg_v = Variable(torch.LongTensor(neg_v))

                if self.use_cuda:
                    pos_u = pos_u.cuda()
                    pos_v = pos_v.cuda()
                    neg_v = neg_v.cuda()

                self.optimizer.zero_grad()
                loss = self.forward(pos_u, pos_v, neg_v)
                loss.backward()

                self.optimizer.step()
                sum_loss = sum_loss + loss.item()
            
            sum_loss = sum_loss / len(batches)
            self.scheduler.step(sum_loss)

            if verbose is True:
                print("iteration: ", i, " loss = ", sum_loss)

            if sum_loss < best_loss:
                best_loss = sum_loss
                torch.save(self.state_dict(), self.tmp_model_file)

            losses.append(sum_loss)

            if i > self.early_stop and losses[-1] > np.mean(losses[-(self.early_stop+1):-1]):
                break
            
        self.load_state_dict(torch.load(self.tmp_model_file))
        os.remove(self.tmp_model_file)

    def forward(self, pos_u, pos_v, neg_v):
      '''
      '''
      raise NotImplementedError

class SamplePMI(SampleEmbedding):
    '''
        Sampling based embedding using the pointwise mutual
        information (as word2vec, node2vec).

        Not currently being used.
    '''
    def __init__(self, G, sampler, n_samples, n_dim=100, learning_rate=0.025,
                 batch_size=50, n_iter=10, n_neg_samples=1, early_stop=10):
        '''
        '''
        super().__init__(G, sampler, n_samples, n_dim, learning_rate, batch_size, n_iter, n_neg_samples, early_stop)

    def init_emb(self, n_dim):
        '''
        '''
        super().init_emb()

        initrange = 0.5 / self.n_dim
        self.u_embed.weight.data.uniform_(-initrange, initrange)
        self.v_embed.weight.data.uniform_(0, 0)

    def train(self, verbose=False):
        '''
        '''
        super().train(verbose)

    def forward(self, pos_u, pos_v, neg_v):
        '''
        '''
        emb_u = self.u_embed(pos_u)
        emb_v = self.v_embed(pos_v)

        score = torch.mul(emb_u, emb_v).squeeze()
        score = torch.sum(score, dim=1)
        score = F.logsigmoid(score)
        neg_emb_v = self.v_embed(neg_v)
        neg_score = torch.bmm(neg_emb_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = F.logsigmoid(-neg_score)

        return -1 * (torch.sum(score)+torch.sum(neg_score))

class ClampGradient(torch.autograd.Function):
    '''
        Activation function for autocovariance
    '''
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.clamp(min=EPS,max=1.)

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        grad_input = g.clone()
        grad_input[x < EPS] = 0.
        grad_input[x > 1] = 0.

        return grad_input

def log_stab_pos(p_uv, dot_prod):
    '''
        Computes part of score for positive samples using
        autocovariance similarity.

        :param p_uv: product of values for stat. dist. pi_u*pi_v
        :param dot_prod: dot product of embeddings
    '''
    c = ClampGradient.apply
    p = c(p_uv + dot_prod)
    
    return torch.log(torch.div(p, p + p_uv))

def log_stab_neg(p_uv, dot_prod):
    '''
        Computes part of score for negative samples using
        autocovariance similarity. 
        
        :param p_uv: product of values for stat. dist. pi_u*pi_v
        :param dot_prod: dot product of embeddings
    '''
    c = ClampGradient.apply
    p = c(p_uv + dot_prod)
    return torch.log(torch.div(p_uv, p + p_uv))

def penalty_neg_stab(p_uv, dot_prod, penalty):
    '''
        Penalizes score when embeddings produce
        negative probabilies in autocovariance
        formulation.
        
        :param p_uv: product of values for stat. dist. pi_u*pi_v
        :param dot_prod: dot product of embeddings
        :param penalty: amount of penalty
    '''
    p = p_uv + dot_prod
    m = nn.ReLU()
    pen = -penalty * m(-p)

    return pen

def penalty_pos_stab(p_uv, dot_prod, penalty):
    '''
        Penalizes score when embeddings produce
        probabilities greater than 1 in autocovariance
        formulation.
        
        :param p_uv: product of values for stat. dist. pi_u*pi_v
        :param dot_prod: dot product of embeddings
        :param penalty: amount of penalty
    '''
    p = p_uv + dot_prod
    m = nn.ReLU()
    pen = -penalty * m(p-1)

    return pen

class SampleAutoCov(SampleEmbedding):
    '''
        Implements sampling-based autocovariance embedding
    '''
    def __init__(self, G, sampler, walks, dim=128, lr=0.025, batch_size=50,
        n_iter=10, neg=5, early_stop=10, penalty=0, momentum=0.99, patience=1):
        '''
            :param G: Input graph
            :param sampler: process sampler
            :param walks: number of random walks per node
            :param dim: Dimensions of embedding
            :param lr: learning rate for SGD
            :param batch_size: size of batches for SGD
            :param n_iter: max number of iterations for SGD
            :param neg: number of negative samples
            :param early_stop: number of iterations before early stop
            :param penalty: penalizes embeddings out of probability range
            :param momentum: Pytorch optimizer parameters
            :param patience: Pytorch optimizer parameters
        '''
        self.penalty = penalty
        super().__init__(G, sampler, walks, dim, lr, batch_size, n_iter, neg, early_stop)

    def init_emb(self, dim):
        '''
            Initializes embedding.

            :param dim: Dimensions of embedding
        '''
        n = self.G.number_of_nodes()
        self.u_embed = nn.Embedding(n, self.n_dim, sparse=True)
        self.v_embed = nn.Embedding(n, self.n_dim, sparse=True)

        self.prob = nn.Embedding.from_pretrained(torch.FloatTensor(self.sampler.start_prob[np.newaxis].T))

        initrange = 0.5 / self.n_dim
        self.u_embed.weight.data.uniform_(-initrange, initrange)
        self.v_embed.weight.data.uniform_(-initrange, initrange)

        w = self.u_embed.weight.data
        w.div_(torch.div( torch.norm(w, 2, 1, keepdim=True), self.prob.weight.data).expand_as(w))
        w = self.v_embed.weight.data
        w.div_(torch.div( torch.norm(w, 2, 1, keepdim=True), self.prob.weight.data).expand_as(w))

    def forward(self, pos_u, pos_v, neg_v):
        '''
            Performs one batch iteration for training/embedding using autocovariance.

            :param pos_u: first vertex in positive sample
            :param pos_v: second vertex in positive sample
            :param neg_v: negative samples for pos_u
        '''
        emb_u = self.u_embed(pos_u)
        emb_v = self.v_embed(pos_v)

        prob_u = self.prob(pos_u)
        prob_v = self.prob(pos_v)

        score = torch.mul(emb_u, emb_v).squeeze()
        score = torch.sum(score, dim=1)

        prob_uv = torch.mul(prob_u,prob_v).squeeze()

        #Penalizing embeddings that do not produce probabilities
        pen_pos = penalty_pos_stab(prob_uv, score, self.penalty) + penalty_neg_stab(prob_uv, score, self.penalty)

        score = log_stab_pos(prob_uv, score)

        neg_emb_v = self.v_embed(neg_v)
        prob_neg_v = self.prob(neg_v)

        neg_score = torch.bmm(neg_emb_v, emb_u.unsqueeze(2)).squeeze()

        prob_neg_uv = torch.bmm(prob_neg_v, prob_u.unsqueeze(2)).squeeze()

        #Penalizing embeddings that do not produce probabilities
        pen_neg = penalty_neg_stab(prob_neg_uv, neg_score, self.penalty) + penalty_pos_stab(prob_neg_uv, neg_score, self.penalty)

        neg_score = log_stab_neg(prob_neg_uv, neg_score)

        return -1 * (torch.sum(score)/score.shape[0]+torch.sum(neg_score)/neg_score.shape[0]+torch.sum(pen_pos)/pen_pos.shape[0]+torch.sum(pen_neg)/pen_neg.shape[0])

def embed(G, dim, tau, weight, directed, similarity, average_similarity, lr, n_iter, early_stop, batch_size, 
    neg, walks, walk_length, damp, workers):
    """
    Embed the graph with the sampling algorithm.

    :param G: Input graph.
    :param dim: Dimensions of embedding.
    :param tau: Markov time.
    :param weight: Edge weights.
    :param directed: Whether the graph is directed.
    :param similarity: Similarity metric.
    :param average_similarity: Whether to use the average version of similarity metric.
    :param lr: learning rate for SGD.
    :param n_iter: max number of iterations for SGD
    :param early_stop: number of iterations before early stop
    :param batch_size: size of batches for SGD
    :param neg: number of negative samples
    :param walks: number of random walks per node
    :param walk_length: length of random walks sampled
    :param damp: damping parameter for pagerank 
    :param workers: number of workers (for node2vec)
    :return: Embeddings of shape (num_nodes, dim)
    """
    if directed and similarity == 'PMI':
        raise NotImplementedError(f'PMI embedding not implemented for directed graphs. ')

    use_cuda = torch.cuda.is_available()
    
    # Select the similarity metric.
    if similarity == 'autocovariance':
        if directed:
            start_prob = pagerank(G, alpha=damp, weight=weight)
            sampler = PRSampler(G, start_prob, walk_length, tau, workers=workers, avg=average_similarity)
        else:
            start_prob = stat_dist_undirected(G, weight=weight)
            #Uniform selection: start_prob = np.ones(G.number_of_nodes()) / G.number_of_nodes()
            sampler = StdRWSampler(G, start_prob, walk_length, tau, workers=workers, avg=average_similarity)

        samp_emb = SampleAutoCov(G, sampler, walks=walks, dim=dim, lr=lr, 
            batch_size=batch_size, n_iter=n_iter, neg=neg, penalty=0.)
        
        samp_emb.train(verbose=True)
        
        if use_cuda:
            u = samp_emb.u_embed(torch.cuda.LongTensor([list(sorted(G.nodes()))])-1).cpu().detach().numpy()[0]
            v = samp_emb.v_embed(torch.cuda.LongTensor([list(sorted(G.nodes()))])-1).cpu().detach().numpy()[0]
        else:
            u = samp_emb.u_embed(torch.LongTensor([list(sorted(G.nodes()))])-1).detach().numpy()[0]
            v = samp_emb.v_embed(torch.LongTensor([list(sorted(G.nodes()))])-1).detach().numpy()[0]

        if directed:
            return utils.rescale_embeddings(u), utils.rescale_embeddings(v) 
        else:
            return utils.rescale_embeddings(u), utils.rescale_embeddings(v)

    elif similarity == 'PMI':
        if directed:
            raise NotImplementedError(f'Directed PMI not implemented. ')
        else:
            node2vec = Node2Vec(G, dimensions=dim, walk_length=walk_length, num_walks=walks, workers=workers, temp_folder='./tmp/')
            model = node2vec.fit(window=tau, min_count=1, batch_words=batch_size, negative=neg)

            u = np.zeros((G.number_of_nodes(), dim))

            i = 0
            for v in sorted(G.nodes()):
                u[i] = model.wv[str(v)]
                i = i + 1
        
            return utils.rescale_embeddings(u)
    else:
        raise NotImplementedError(f'Similarity metric {similarity} not implemented. ')

