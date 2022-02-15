import json
import logging
import os
import pickle
import re
import sys
from collections import defaultdict
from typing import Optional, Union, List, Tuple

import fire
import numpy as np
import torch
from gensim import matutils
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from smart_open import open
from tqdm.auto import tqdm
from transformers import set_seed

from cli_s2orc import write_citations
from gdt.utils import get_workers, get_graph_embeddings

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def read_citations_from_tsv(input_path: str) -> List[Tuple[str, str]]:
    """
    Read citation tuples from TSV file (ignore all comments = lines with #)

    :param input_path: Path to TSV file
    :return:
    """
    cits = []
    with open(input_path) as f:
        for l in f:
            l = l.strip()
            if not l.startswith('#'):
                from_id, to_id = l.split('\t')
                cits.append((from_id, to_id))

    return cits


def get_largest_connected_subgraph(input_path, output_path, largest_n: int = 0, override: bool = False):
    """
    Get citations from largest connected subgraph of given citation data.

    Example:

    python cli_graph.py get_largest_connected_subgraph \
        --input_path data/biggraph/specter_train/citations.tsv \
        --output_path data/biggraph/specter_train/largest_connected_subgraph/citations.tsv
    :return:
    """
    col_sep = '\t'
    line_sep = '\n'

    if not override and os.path.exists(output_path):
        raise FileExistsError(f'Output exist already and --override not set')

    cits = read_citations_from_tsv(input_path)

    logger.info(f'Total citations: {len(cits)}')

    import networkx as nx

    # Init full graph
    g = nx.Graph()
    g.add_edges_from(cits)

    # Generate a sorted list of connected components, largest first.
    connected_sub_graphs = sorted(nx.connected_components(g), key=len, reverse=True)

    logger.info(f'Subgraphs found: {len(connected_sub_graphs)}')
    logger.info(f'Select subgraph #{largest_n}')

    sub_graph_nodes = connected_sub_graphs[largest_n]
    sg_cits = [(_from, _to) for _from, _to in cits if
                _from in sub_graph_nodes and _to in sub_graph_nodes]

    # init new graph
    sg = nx.Graph()
    sg.add_edges_from(sg_cits)

    # sub graph must be connected for graph embedding training
    if not nx.is_connected(sg):
        raise ValueError(f'Sub-graph #{largest_n} is not connected!')

    # save to TSV
    write_citations(sg_cits, output_path, len(sub_graph_nodes), col_sep, line_sep, f'Largest connect sub graph of {input_path}')


def train_poincare(
        input_path: str,
        output_path: str,
        include_path: Optional[str] = None,
        size: int = 768,
        epochs: int = 50,
        burn_in: int = 10
        ):
    """
    Train poincare embeddings based on citation graph (by default uses hyperparameters from legaldocsim paper)

    TODO poincare is too slow !!! other graph embedding methods needed

    python cli_graph.py train_poincare ./data/biggraph/s2orc_full/citations.tsv ./data/poincare/model \
        --include_path ./data/sci/gridsearch/s2orc_paper_ids.json

    python cli_graph.py train_poincare ./data/biggraph/s2orc_full/citations.tsv ./data/poincare/model \
        --include_path ./data/sci/gridsearch/query_s2orc_paper_ids.sample_0.01.json --epochs 1 --burn_in 2

    # Training on limited citation graph (635_248 edges, 2_671_009 nodes => approx. 5 hrs for 1 epoch)
    python cli_graph.py train_poincare data/biggraph/specter_train/largest_connected_subgraph/citations.tsv ./data/poincare/specter_train --epochs 1 --burn_in 2

    python cli_graph.py train_poincare data/biggraph/specter_train/largest_connected_subgraph/citations.tsv ./data/poincare/specter_train --epochs 20 --size 300

    :param burn_in:
    :param include_path: JSON file with IDs to be included
    :param input_path: Path to citations (TSV)
    :param output_path: Path to model (+ hdf5 + paper ids json)
    :param size: Vector size
    :param epochs: Training epochs
    :return:
    """
    from gensim.models.poincare import PoincareModel
    import h5py

    if os.path.exists(output_path):
        logger.info('Model exist already.. loading existing model')
        poincare_model = PoincareModel.load(output_path)
    else:

        logger.info(f'Loading citations from {input_path}')
        citations = read_citations_from_tsv(input_path)  # (citing_id, cited_id)

        logger.info(f'Citation loaded: {len(citations):,}')

        if include_path:
            include_ids_set = set(json.load(open(include_path)))
            logger.info(f'Filter with IDs: {len(include_ids_set):,}')

            # TODO and/or?
            citations = [(from_id, to_id) for from_id, to_id in citations if from_id in include_ids_set or to_id in include_ids_set]

            logger.info(f'Filtered citations: {len(citations)}')

        # unique_paper_ids = set(paper_ids)

        # logger.info(f'Paper IDs: {len(paper_ids):,} (unique: {len(unique_paper_ids):,})')

        # if include_ids_set and (len(unique_paper_ids) != len(include_ids_set)):
        #     logger.error('Filter set but unique paper count does not match actual paper count')
        #     logger.error(f'unique_paper_ids = {len(unique_paper_ids)} != include_ids_set = {len(include_ids_set):,}')
        #     return

        poincare_model = PoincareModel(
            # Same as in
            # https://github.com/malteos/legal-document-similarity/blob/master/commands/compute_doc_vecs.py#L299-L312
            citations,
            size=size,
            alpha=0.1,
            negative=10,
            workers=1,  # gensim implementation does not support multi threading
            epsilon=1e-05,
            regularization_coeff=1.0,
            burn_in=burn_in,
            burn_in_alpha=0.01,
            init_range=(-0.001, 0.001),
        )
        logger.info('Training ...')
        poincare_model.train(
            epochs=epochs,
        )
        logger.info(f'Save at {output_path}')
        poincare_model.save(output_path)

    # Save JSON: entity_names_paper_id_0.json
    with open(output_path + '.paper_ids.json', 'w') as f:
        json.dump(list(poincare_model.kv.key_to_index.keys()), f)

    # Embeddings
    embeds = poincare_model.kv.vectors

    # Save h5: embeddings_paper_id_0.v200.h5
    with h5py.File(output_path + '.embeddings.hdf5', "w") as hf:
        hf_ds = hf.create_dataset("embeddings", data=embeds)

    logger.info('done')


def build_annoy(
        graph_embeddings_path: str,
        index_path: str,
        paper_ids: Optional[str] = None,
        include_paper_ids: Optional[str] = None,
        do_normalize: bool = False,
        ann_metric: str = 'euclidean',
        ann_trees: int = 100,
        ann_on_disk_build: bool = False,
        workers: int = 10,
        seed: int = 0,
        ):
    """
    Build approximate nearest neighbors index

    Usage:

    python cli_s2orc.py build_annoy ${S2ORC_EMBEDDINGS} \
        ${BASE_DIR}/data/biggraph/models/s2orc/epoch_20/index__dot__1000-trees.ann \
        --do_normalize --ann_metric dot --ann_trees 1000 --workers ${WORKERS} --ann_on_disk_build

    :param include_paper_ids: See get_graph_embeddings
    :param paper_ids: See get_graph_embeddings
    :param ann_on_disk_build:
    :param seed:
    :param graph_embeddings_path:
    :param index_path:
    :param do_normalize:
    :param ann_metric:
    :param ann_trees:
    :param workers:
    :return:
    """

    graph_embeddings = get_graph_embeddings(graph_embeddings_path, do_normalize, workers, paper_ids, include_paper_ids)
    graph_embedding_size = len(graph_embeddings[0])

    ann_index = AnnoyIndex(graph_embedding_size, ann_metric)
    ann_index.set_seed(seed)

    for idx, embed in tqdm(enumerate(graph_embeddings), desc='Adding to index', total=len(graph_embeddings)):
        ann_index.add_item(idx, embed)

    # delete to free up memory
    del graph_embeddings

    if ann_on_disk_build:
        logger.info('Building on disk')
        ann_index.on_disk_build(index_path + '.build')
    else:
        logger.info(f'Building ANN index with trees={ann_trees} and workers={workers}')
        ann_index.build(ann_trees, n_jobs=workers)

    logger.info(f'Saving ANN index to {index_path}')

    ann_index.save(index_path)


def build_faiss(
        graph_embeddings_or_path: Union[str, np.ndarray],
        index_path: str,
        string_factory: str = 'PCA64,IVF16384_HNSW32,Flat',
        metric_type: int = 0,
        do_normalize: bool = False,
        paper_ids: Optional[str] = None,
        include_paper_ids: Optional[str] = None,
        workers: int = 10,
        batch_size: int = 1000,
        train_size: int = 0,
        seed: int = 0,
        device: Union[int, str] = -1,):
    """
    Build FAISS ANN index

    # on full S2ORC
    python cli_s2orc.py build_faiss ${S2ORC_EMBEDDINGS} \
        ${BASE_DIR}/data/ann_benchmark/IVF16384_HNSW32,Flat.faiss \
        --string_factory IVF16384_HNSW32,Flat \
        --do_normalize \
        --batch_size 512 --workers ${WORKERS} --device 0

    # on SPECTER papers only
    python cli_s2orc.py build_faiss ${S2ORC_EMBEDDINGS} \
        ${OLD_DIR}/IVF16384_HNSW32,Flat.faiss \
        --string_factory IVF16384_HNSW32,Flat \
        --paper_ids ${S2ORC_PAPER_IDS} --include_paper_ids ${OLD_DIR}/s2orc_paper_ids.json \
        --do_normalize \
        --batch_size 512 --workers ${WORKERS} --device 0

    # TODO maybe limit train_size if GPU memory is not enough
    # TODO use PCA -> PCA128, PCA256?

    IndexFlat: the vectors are stored without compression
    IndexIVF:IVF16384 (2^14): The feature space is partitioned into nlist cells.
    HNSW32: M is the number of neighbors used in the graph. A larger M is more accurate but uses more memory


    :param seed:
    :param include_paper_ids: See get_graph_embeddings
    :param paper_ids: See get_graph_embeddings
    :param batch_size:
    :param train_size:
    :param device: GPU index or "all"
    :param string_factory:  https://github.com/facebookresearch/faiss/wiki/The-index-factory
    :param metric_type: METRIC_INNER_PRODUCT = 0
    :param graph_embeddings_or_path: Path to graph embeddings or embeddings as numpy array
    :param index_path: Output path
    :param do_normalize: Normalize input graph embeddings (for cosine similarity)
    :param workers:
    :return:
    """
    import faiss

    set_seed(seed)  # make reproducible

    if isinstance(string_factory, tuple) or isinstance(string_factory, list):
        string_factory = ','.join(string_factory)  # force to be string

    workers = get_workers(workers)

    if isinstance(graph_embeddings_or_path, str):
        # Load from disk
        graph_embeddings = get_graph_embeddings(graph_embeddings_path=graph_embeddings_or_path,
                                                do_normalize=False,  # normalize with faiss
                                                workers=workers,
                                                paper_ids=paper_ids,
                                                include_paper_ids=include_paper_ids)
    else:
        graph_embeddings = graph_embeddings_or_path

    graph_embedding_size = len(graph_embeddings[0])

    if do_normalize:
        logger.info(f'Normalize {graph_embeddings.shape} with faiss')
        faiss.normalize_L2(graph_embeddings)

    index = faiss.index_factory(graph_embedding_size, string_factory, metric_type)

    if isinstance(device, int) and device > -1:
        logger.info(f'Use GPU device: {device}')
        faiss_res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(faiss_res, device, index)
    elif device == 'all':
        # use all available GPUs
        logger.info(f'Use all available GPUs: {faiss.get_num_gpus()}')

        index = faiss.index_cpu_to_all_gpus(  # build the index
            index
        )

    verbose = True
    index.verbose = verbose

    if hasattr(index, "index") and index.index is not None:
        index.index.verbose = verbose
    if hasattr(index, "quantizer") and index.quantizer is not None:
        index.quantizer.verbose = verbose
    if hasattr(index, "clustering_index") and index.clustering_index is not None:
        index.clustering_index.verbose = verbose

    if train_size > 0:
        train_vecs = graph_embeddings[:train_size]
    else:
        train_vecs = graph_embeddings

    logger.info(f'Training ... train_size = {train_size:,}')

    index.train(train_vecs)

    # write to disk
    if (isinstance(device, int) and device > -1) or device == 'all':
        logger.info('Index back to CPU')

        index = faiss.index_gpu_to_cpu(index)

    for i in tqdm(range(0, len(graph_embeddings), batch_size), desc=f'Adding (batch_size={batch_size:,})'):
        vecs = graph_embeddings[i: i + batch_size]
        index.add(vecs)

        # See https://github.com/facebookresearch/faiss/issues/1517
        # index.reclaimMemory()

    faiss.write_index(index, index_path)

    logger.info(f'Index saved at {index_path}')

    # Save config
    config_path = index_path + '.faiss_config.json'
    config = {
        'faiss_version': faiss.__version__,
        'seed': seed,
        'string_factory': string_factory,
        'device': device,
        'batch_size': batch_size,
        'train_size': train_size,
        'workers': workers,
        'train_vecs_shape': list(train_vecs.shape),
    }
    with open(config_path, 'w') as f:
        json.dump(config, f)

    logger.info(f'Config saved to {config_path}')


def get_knn(
        graph_embeddings_path: str,
        query_idxs: Union[str, List],
        output_path: Optional[str] = None,
        do_normalize: bool = False,
        batch_size: int = 100,
        k: int = 1000,
        workers: int = 10):
    """
    Get exact k nearest neighbors from graph embeddings

    python cli_s2orc.py get_knn ${S2ORC_EMBEDDINGS} --query_idxs 0,1,2,3,4 \
        --output_path ${BASE_DIR}/data/ann_benchmark/query_knns.json \
        --batch_size 3 --workers ${WORKERS} --k 1000 --do_normalize

    :param graph_embeddings_path:
    :param query_idxs:
    :param output_path:
    :param do_normalize:
    :param batch_size:
    :param k:
    :param workers:
    :return:
    """
    graph_embeddings = get_graph_embeddings(graph_embeddings_path, do_normalize, workers)

    if isinstance(query_idxs, str):
        logger.info(f'Load query indices from {query_idxs}')
        with open(query_idxs) as f:
            query_idxs = json.load(f)

    query_vecs = graph_embeddings[query_idxs, :]

    logger.info(query_vecs.shape)

    batch_out = []

    for i in tqdm(range(0, len(query_vecs), batch_size), desc='Search NN'):
        batch_query_vecs = query_vecs[i: i + batch_size]

        # cosine similarity or cosine distance?
        cosine_similarities = linear_kernel(batch_query_vecs, graph_embeddings)

        sorted_idxs = matutils.argsort(cosine_similarities, reverse=True, topn=k+1)  # in reverse order

        sorted_idxs = sorted_idxs[:, 1:k+1]  # k nearest neighbors (exclude self)

        # stack
        batch_out.append(sorted_idxs)

        del cosine_similarities

    if len(batch_out) == 1:
        query_knns = batch_out[0]
    else:
        query_knns = np.concatenate(list(batch_out), axis=0)

    if output_path:
        # write as JSON
        logger.info(f'Writing to {output_path}')
        with open(output_path, 'w') as f:
            json.dump(query_knns.tolist(), f)
    else:
        return query_knns


def build_k_means(
        graph_embeddings_path: str,
        paper_ids: Union[str, List[str]],
        output_dir: Optional[str] = None,
        seed: int = 0,
        k: int = 10,
        max_points_per_centroid: int = 1_000_000,
        min_points_per_centroid: int = 1,
        niter: int = 20,
        nredo: int = 1,
    ):
    """
    Run k-means clustering on graph embeddings (approx. 25min on two GPUs with k=10k; little GPU memory needed)

    Usage:

    export CUDA_VISIBLE_DEVICES=...
    python cli_s2orc.py build_k_means ${S2ORC_EMBEDDINGS} ${S2ORC_PAPER_IDS} ${K_MEANS_DIR} \
        -k ${K_MEANS_CENTROIDS} -niter ${K_MEANS_N_ITER} -nredo 3

    :param paper_ids: Path to JSON with paper IDs or list of paper IDs
    :param nredo: Repeat k-means n times
    :param niter: K-means iterations
    :param max_points_per_centroid:
    :param graph_embeddings_path: Path to graph embeddings
    :param output_dir: Store output files in this directory (paper_idx_to_centroid.json, centroids.npy, sorted_centroids.npy, similarities.npy)
    :param seed: Random seed
    :return: paper idx to centroid mapping, centroid positions, cosine similarities, centroids sorted by cosine similarity.
    """

    import faiss

    set_seed(seed)

    if isinstance(paper_ids, str):
        with open(paper_ids) as f:
            paper_ids = json.load(f)

    if not os.path.exists(output_dir):
        logger.info(f'Creating output dir: {output_dir}')
        os.makedirs(output_dir)

    ngpu = torch.cuda.device_count()

    if ngpu < 1:
        raise ValueError(f'No GPU available')

    logger.info(f'GPUs: {ngpu}')

    graph_embeddings = get_graph_embeddings(graph_embeddings_path, do_normalize=False, workers=10)
    graph_embedding_size = graph_embeddings.shape[1]

    clustering = faiss.Clustering(graph_embedding_size, k)

    clustering.verbose = True
    clustering.niter = niter
    clustering.nredo = nredo
    clustering.seed = seed
    clustering.max_points_per_centroid = max_points_per_centroid  # otherwise the kmeans implementation sub-samples the training set
    clustering.min_points_per_centroid = min_points_per_centroid

    # GPU setup
    res = [faiss.StandardGpuResources() for i in range(ngpu)]

    flat_config = []
    for i in range(ngpu):
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = i
        flat_config.append(cfg)

    if ngpu == 1:
        index = faiss.GpuIndexFlatL2(res[0], graph_embedding_size, flat_config[0])
    else:
        indexes = [faiss.GpuIndexFlatL2(res[i], graph_embedding_size, flat_config[i])
                   for i in range(ngpu)]
        index = faiss.IndexProxy()
        for sub_index in indexes:
            index.addIndex(sub_index)

    logger.info('start train')
    clustering.train(graph_embeddings, index)
    logger.info(f'end train')

    # quantization_error = clustering.train(graph_embeddings, index)
    # logger.info(f'end train. quantization error  = {quantization_error}')

    # Return the assignment and centroid positions
    logger.info('start search')
    _, ids = index.search(graph_embeddings, 1)

    # centroid pairwise distance
    centroids = faiss.vector_float_to_array(clustering.centroids).reshape(k, graph_embedding_size)
    # objective = faiss.vector_float_to_array(clustering.obj)

    logger.info('Computing centroid similarity')
    similarities = cosine_similarity(centroids)

    sorted_centroids = np.argsort(-1 * similarities, axis=1)  # descending order

    idx_to_paper_id = {idx: paper_id for idx, paper_id in enumerate(paper_ids)}
    paper_id_to_centroid = {}
    centroid_to_paper_ids = defaultdict(list)

    for paper_idx, centroid in enumerate(ids[:, 0]):
        paper_id = idx_to_paper_id[paper_idx]
        paper_id_to_centroid[paper_id] = centroid
        centroid_to_paper_ids[centroid].append(paper_id)

    if output_dir:
        logger.info(f'Writing output into: {output_dir}')

        with open(os.path.join(output_dir, 'centroids.npy'), 'wb') as f:
            np.save(f, centroids)

        with open(os.path.join(output_dir, 'sorted_centroids.npy'), 'wb') as f:
            np.save(f, sorted_centroids)

        with open(os.path.join(output_dir, 'similarities.npy'), 'wb') as f:
            np.save(f, similarities)

        with open(os.path.join(output_dir, 'centroid_to_paper_ids.pickle'), 'wb') as f:
            pickle.dump(centroid_to_paper_ids, f,
                        pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(output_dir, 'paper_id_to_centroid.pickle'), 'wb') as f:
            pickle.dump(paper_id_to_centroid, f,
                        pickle.HIGHEST_PROTOCOL)

    else:
        return ids, clustering.centroids, similarities


def extract_ids_in_graph(input_paths: Union[str, List[str]], graph_ids_path: str):
    """
    python cli_graph.py extract_ids_in_graph \
        --graph_ids_path ${S2ORC_PAPER_IDS} \
        --input_paths ${QUERY_DIR}/query_s2orc_paper_ids.specter.json,${QUERY_DIR}/s2orc_paper_ids.json

    :param input_paths: Comma separated list of paths (.json files with list of IDs)
    :param graph_ids_path: JSON list with IDs
    :return:
    """

    logger.info(f'Loading {graph_ids_path} ...')

    with open(graph_ids_path) as f:
        graph_paper_ids = json.load(f)

    graph_paper_ids_set = set(graph_paper_ids)
    logger.info(f'Graph IDs: {len(graph_paper_ids_set):,}')

    if isinstance(input_paths, str):  # to list if only single path is given
        input_paths = input_paths.split(',')

    for input_path in input_paths:
        if not input_path.endswith('.json') and not input_path.endswith('.json.gz'):
            raise ValueError(f'Invald input path (not .json or .json.gz): {input_path}')

        logger.info(f'Loading {input_path} ...')

        with open(input_path) as f:
            input_ids = json.load(f)

        input_ids_set = set(input_ids)

        input_ids_in_graph = input_ids_set & graph_paper_ids_set

        logger.info(f'Input IDs in graph: {len(input_ids_in_graph):,} / {len(input_ids_set):,}')

        output_path = input_path.replace('.json', '.in_graph.json')

        logger.info(f'Saving to {output_path}...')

        with open(output_path, 'w') as f:
            json.dump(list(input_ids_in_graph), f)

    logger.info('done')


def make_edges_bidirectional(input_path, output_path, override: bool = False):
    """
    Convert edges bidirectional (a->b) to (a->b,b->a).

    Example:

    python cli_graph.py make_edges_bidirectional data/biggraph/s2orc_with_specter_without_scidocs/citations.tsv data/biggraph/s2orc_with_specter_without_scidocs_bidirectional/citations.tsv

    Convert from TSV:

    # Directed graph
    # Description: Directed citation graph from S2ORC
    # Nodes: 52373977 Edges: 447697727
    # FromNodeId	ToNodeId
    77499681	71492442
    77499681	3224020
    ...

    to TSV:

    # Undirected graph
    # Description: Directed citation graph from S2ORC
    # Nodes: 52373977 Edges: 447697727*2
    # FromNodeId	ToNodeId
    77499681	71492442
    71492442	77499681
    ...

    :param input_path:
    :param output_path:
    :param override:
    :return:
    """
    if os.path.exists(output_path) and not override:
        raise FileExistsError(f'Output exists already and --override is not set: {output_path}')

    logger.info(f'Reading from {input_path}')
    logger.info(f'Writing to {output_path}')

    # Parse header
    edges_count = 0
    nodes_count = 0
    lines_count = 4  # header lines
    with open(input_path) as f_in:
        for i, line in enumerate(f_in):
            if i == 2:
                match = re.search(r'Nodes: ([0-9]+) Edges: ([0-9]+)', line)
                if not match:
                    raise ValueError('could not parse TSV')

                edges_count = int(match.group(2))
                nodes_count = int(match.group(1))

                lines_count += edges_count
                break

    # Read and write lines
    with open(input_path) as f_in:
        with open(output_path, 'w') as f_out:
            for i, line in tqdm(enumerate(f_in), total=lines_count):
                if i == 0:
                    line = f'# Undirected graph\n'
                elif i == 2:
                    line = f'# Nodes: {nodes_count} Edges: {edges_count*2}\n'
                elif i > 3:
                    from_id, to_id = line.strip().split('\t')

                    # append mirrored edge
                    line += f'{to_id}\t{from_id}\n'

                f_out.write(line)

    logger.info('done')


if __name__ == '__main__':
    fire.Fire()
    sys.exit(0)
