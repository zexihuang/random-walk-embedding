# Random-walk Embedding Framework

This repository is a reference implementation of the random-walk embedding framework as described in the paper:
<br/>
> A Broader Picture of Random-walk Based Graph Embedding.<br>
> Zexi Huang, Arlei Silva, Ambuj Singh.<br>
> ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2021.
> <Insert paper link>

The framework decomposes random-walk based graph embedding into three major components: random-walk process, similarity function, and embedding algorithm. 
By tuning the components, it not only covers many existing approaches such as DeepWalk but naturally motivates novel ones that have shown superior performance on certain downstream tasks.

## Usage

### Example
To use the framework with default settings to embed the BlogCatalog network:<br/>
``python src/embedding.py --graph graph/blogcatalog.edges --embeddings emb/blogcatalog.embeddings``
<br/> 
where `graph/blogcatalog.edges` stores the input graph and `emb/blogcatalog.embeddings` is the target file for output embeddings. 
### Options
You can check out all the available options (framework components, Markov time parameters, graph types, etc.) with:<br/>
	``python src/embedding.py --help``

### Input Graph
The supported input graph format is a list of edges:

	node1_id_int node2_id_int <weight_float, optional>
		
where node ids are should be consecutive integers starting from 1. The graph is by default undirected and unweighted, which can be changed by setting appropriate flags. 

### Output Embeddings
The output embedding file has *n* lines where *n* is the number of nodes in the graph. Each line stores the learned embedding of the node with its id equal to the line number: 

	emb_dim1 emb_dim2 ... emb_dimd

## Evaluating

Here, we show by examples how to evaluate and compare different settings of our framework on node classification, link prediction, and community detection tasks. 
Full evaluation options are can be found with:<br/>
                                              ``python src/evaluating.py --help``

Note that the results shown below may not be identical to those in the paper due to different random seeds, but the conclusions are the same.  

### Node Classification

Once we generate the embedding with the script in previous section, we can call<br/>
	``python src/evaluating.py --task node-classification --embeddings emb/blogcatalog.embeddings --training-ratio 0.5`` 
	<br/>
to compute the *Micro-F1* and *Macro-F1* scores of the node classification. 

The results for comparing Pointwise Mutual Information (PMI) and Autocovariance (AC) similarity metrics with the best Markov times and varying training ratios are as follows:

|          | Training Ratio |   10%  |   20%  |   30%  |   40%  |   50%  |   60%  |   70%  |   80%  |   90%  |
|:--------:|:--------------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
|    PMI   |    Micro-F1    | 0.3503 | 0.3814 | 0.3993 | 0.4106 | 0.4179 | 0.4227 | 0.4255 | 0.4222 | 0.4228 |
| (time=4) |    Macro-F1    | 0.2212 | 0.2451 | 0.2575 | 0.2669 | 0.2713 | 0.2772 | 0.2768 | 0.2689 | 0.2678 |
|    AC    |    Micro-F1    | 0.3547 | 0.3697 | 0.3785 | 0.3837 | 0.3872 | 0.3906 | 0.3912 | 0.3927 | 0.3930 |
| (time=5) |    Macro-F1    | 0.2137 | 0.2299 | 0.2371 | 0.2406 | 0.2405 | 0.2413 | 0.2385 | 0.2356 | 0.2352 |

### Link Prediction

#### Prepare

To evaluate the embedding method on link prediction, we first have to remove a ratio of edges in the original graph:<br/>
     ``python src/evaluating.py --task link-prediction --mode prepare --graph graph/blogcatalog.edges --remaining-edges graph/blogcatalog.remaining-edges --removed-edges graph/blogcatalog.removed-edges``

This takes the original graph `graph/blogcatalog.edges` as input and output the removed and remaining edges to `graph/blogcatalog.removed-edges` and `graph/blogcatalog.remaining-edges`.

#### Embed

Then, we embed based on the remaining edges of the network with the embedding script. For example:<br/>
``python src/embedding.py --graph graph/blogcatalog.remaining-edges --embeddings emb/blogcatalog.residual-embeddings``

#### Evaluate

Finally, we evaluate the performance of link prediction in terms of *precision@k* based on the embeddings of the residual graph and the removed edges:<br/>
``python src/evaluating.py --task link-prediction --mode evaluate --embeddings emb/blogcatalog.residual-embeddings --remaining-edges graph/blogcatalog.remaining-edges --removed-edges graph/blogcatalog.removed-edges --k 1.0``

The results for comparing PMI and autocovariance similarity metrics with the best Markov times and varying *k* are as follows:

|       k      |   10%  |   20%  |   30%  |   40%  |   50%  |   60%  |   70%  |   80%  |   90%  |  100%  |
|:------------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| PMI (time=1) | 0.2958 | 0.2380 | 0.2068 | 0.1847 | 0.1678 | 0.1560 | 0.1464 | 0.1382 | 0.1315 | 0.1260 |
|  AC (time=3) | 0.4213 | 0.3420 | 0.2982 | 0.2667 | 0.2434 | 0.2253 | 0.2112 | 0.2000 | 0.1893 | 0.1802 |

### Community Detection

Assume the embeddings for the Airport network `emb/airport.embeddings` have been generated. The following computes the Normalized Mutual Information (*NMI*) between the ground-truth country communities and the k-means clustering of embeddings:<br/>
``python src/evaluating.py --task community-detection --embeddings emb/airport.embeddings --communities graph/airport.country-labels``


## Citing
If you find our framework useful, please consider citing the following paper:

	@inproceedings{random-walk-embedding,
	author = {Huang, Zexi and Silva, Arlei and Singh, Ambuj},
	 title = {A Broader Picture of Random-walk Based Graph Embedding},
	 booktitle = {SIGKDD},
	 year = {2021}
	}