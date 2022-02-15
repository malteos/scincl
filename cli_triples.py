import inspect
import json
import logging
import os
import random
from collections import defaultdict
from multiprocessing import Pool
from typing import Union, List, Optional

import fire
import numpy as np
from smart_open import open
from tqdm.auto import tqdm
from transformers import set_seed

from cli_graph import read_citations_from_tsv
from cli_specter import find_train_ids, BaseCorpus
from gdt.scidocs_utils import get_recomm_triples, get_triples_from_qrel, get_triples_from_class_csv
from gdt.triples_miner import TriplesMinerArguments
from gdt.triples_miner.generic import get_generic_triples
from gdt.utils import split_into_n_chunks, write_func_args, get_workers, get_graph_embeddings, read_json_mapping_files

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# See https://github.com/allenai/s2orc#metadata-schema
S2ORC_METADATA_FIELDS = ['paper_id', 'title', 'abstract', 'arxiv_id', 'doi']


def get_specter_triples(output_path: str,
                        scidocs_dir: str,
                        specter_triples_path: str,
                        graph_paper_ids_path: str,
                        graph_embeddings_path: str,
                        s2id_to_s2orc_input_path: str,
                        train_s2orc_paper_ids: Union[str, List[str]] = None,
                        train_query_s2orc_paper_ids: Union[str, List[str]] = None,
                        sample_queries_ratio: float = 1.0,
                        # sample_queries_ratio_with_specter: bool = False,
                        graph_limit: BaseCorpus = BaseCorpus.SPECTER,
                        workers: int = 10,
                        triples_miner_args: TriplesMinerArguments = None,
                        **triples_miner_kwargs):
    """
    Triple mining

    python cli_triples.py get_specter_triples ${EXP_DIR}/train_triples.csv \
        --scidocs_dir ${SCIDOCS_DIR} \
        --specter_triples_path ${SPECTER_DIR}/train_triples.csv \
        --paper_ids_path ${S2ORC_PAPER_IDS} \
        --embeddings_path ${S2ORC_EMBEDDINGS} \
        --s2id_to_s2orc_input_path ${SPECTER_DIR}/s2id_to_s2orc_paper_id.json \
        --easy_negatives_count 3 \
        --hard_negatives_count 2 \
        --ann_top_k 500 \
        --workers 50

    SPECTER:
    - To train our model we use about 146K query papers -> but train.pickle = 136_820 unique query papers
    - We empirically found it helpful to use 2 hard negatives (citations of citations) and 3 easy negatives (randomly selected papers) for each query paper.
    - SPECTER paper "This process results in about 684K training triples"

    :param sample_queries_ratio_with_specter: When sample query documents, then only sample from SPECTER queries
    :param sample_queries_ratio: Down- or up-sample query documents (default = 1.0 = no sampling)
    :param graph_limit: Limit citation graph to either SPECTER or S2ORC (= full graph = no limit). Choices: specter,s2orc
    :param triples_miner_args:
    :param ann_metric: Metric for ANN index (default: euclidean)
    :param train_query_s2orc_paper_ids: Path to JSON or list of query paper IDs used for training
    :param train_s2orc_paper_ids: Path to JSON or list of query paper IDs used for training
    :param ann_index_path: ANN index is saved to disk at this path (default: output_path.ann)
    :param ann_top_k: The lower k the harder the negatives
    :param ann_trees: More trees gives higher precision when querying ANN
    :param easy_negatives_count: SPECTER: 3 easy negatives (randomly selected papers) for each query paper.
    :param hard_negatives_count: SPECTER 2 hard negatives (citations of citations)
    :param triples_per_query: SPECTER: up to 5 training triples comprised of a query
    :param seed: Random seed
    :param graph_embeddings_path: Graph embeddings path (.h5 file)
    :param graph_paper_ids_path: Paper IDs of graph embeddings (.json files)
    :param scidocs_dir: SciDocs evaluation data dir
    :param output_path: Saves triples as CSV (columns: query_paper_id, positive_id, negative_id)
    :param workers: Threads for building ANN and mining triplets
    :param specter_triples_path: SPECTER triples data (get training paper ids: train_triples.csv)
    :param s2id_to_s2orc_input_path:
    :return:
    """
    train_query_s2orc_paper_ids_path = None

    # Log arg settings
    write_func_args(inspect.currentframe(), output_path + '.args.json')

    workers = get_workers(workers)
    triples_miner_args = TriplesMinerArguments.args_or_kwargs(triples_miner_args, triples_miner_kwargs)

    set_seed(triples_miner_args.seed)

    # SPECTER: To train our model we use about 146K query papers -> but train.pickle = 136_820 unique query papers
    # -> probably the others are part of validation!
    # query_papers_count = 136_820 #146_000
    # train_unique_papers_count = 311_860  # Number of unique papers in training set (same as SPECTER)

    # See https://github.com/allenai/specter/blob/master/scripts/pytorch_lightning_training_script/train.py#L44
    # training_size = 684100  #TODO why is this not equal to train_n? -> SPECTER paper "This process results in about 684K training triples"
    # wc-l =>  730001 data/s2orc/train_triples.csv
    # For each query paper we con- struct up to 5 training triples comprised of a query

    with open(graph_paper_ids_path) as f:
        s2orc_paper_ids = json.load(f)  # S2ORC Ids

    # Papers in train corpus and query papers
    if train_s2orc_paper_ids is not None and train_query_s2orc_paper_ids is not None:
        if isinstance(train_s2orc_paper_ids, list) and isinstance(train_query_s2orc_paper_ids, list):
            # ids are provided as arguments
            pass
        elif isinstance(train_s2orc_paper_ids, str) and isinstance(train_query_s2orc_paper_ids, str):
            # load from path
            train_query_s2orc_paper_ids_path = train_query_s2orc_paper_ids

            with open(train_s2orc_paper_ids) as f:
                train_s2orc_paper_ids = json.load(f)
            with open(train_query_s2orc_paper_ids_path) as f:
                train_query_s2orc_paper_ids = json.load(f)
        else:
            raise ValueError(f'Train S2ORC (query) paper ids not set: {type(train_s2orc_paper_ids)}')
    else:
        # Generate new train ids
        train_s2orc_paper_ids, train_query_s2orc_paper_ids = find_train_ids(scidocs_dir, specter_triples_path,
                                                                            s2id_to_s2orc_input_path, s2orc_paper_ids)

    # Load embeddings from disk # TODO loading embeddings later would be better
    graph_embeddings = get_graph_embeddings(
        graph_embeddings_path,
        # do_normalize=triples_miner_args.ann_normalize_embeddings,
        do_normalize=False,  # normalize with ANN backend
        placeholder=triples_miner_args.ann_index_path is not None and os.path.exists(triples_miner_args.ann_index_path)
    )

    # with h5py.File(graph_embeddings_path, "r") as hf:
    #     if triples_miner_args.ann_index_path is None:
    #         # load full graph embeddings
    #
    #         # reading all embeddings at once is quicker than loading with indices
    #         graph_embeddings = hf["embeddings"][:, :]
    #     else:
    #         logger.warning('Skipping graph embeddings because `ann_index_path` is set and ANN index will not be build.')
    #
    #         graph_embeddings = hf["embeddings"][:1, :] # placeholder (only the first entry)
    #
    # logger.info(f'Graph embeddings: {graph_embeddings.shape}')

    if graph_limit == BaseCorpus.S2ORC or graph_limit == BaseCorpus.NONE or graph_limit == BaseCorpus.GRAPH:
        # Do not change input graph
        # - None: no change
        # - S2ORC: Utilize full citation graph of S2ORC without filtering
        train_embeddings = graph_embeddings
        train_s2orc_paper_id_to_idx = {pid: idx for idx, pid in enumerate(s2orc_paper_ids)}
        train_idx_to_s2orc_paper_id = {idx: pid for idx, pid in enumerate(s2orc_paper_ids)}

    elif graph_limit == BaseCorpus.SPECTER:
        # Limit citation graph to the papers that are also used by SPECTER

        # Find indices in graph embeddings and extract vectors
        s2orc_paper_id_to_paper_idx = {pid: idx for idx, pid in enumerate(s2orc_paper_ids)}

        if not isinstance(train_s2orc_paper_ids, list):
            train_s2orc_paper_ids = list(train_s2orc_paper_ids)  # python sets are unordered -> convert to list!

        logger.warning('Limiting graph embedding to SPECTER')

        if triples_miner_args.ann_index_path is None or not os.path.exists(triples_miner_args.ann_index_path):
            train_embeddings = np.array(
                [graph_embeddings[s2orc_paper_id_to_paper_idx[s2orc_id], :] for s2orc_id in train_s2orc_paper_ids])
        else:
            train_embeddings = graph_embeddings  # do not filter if ANN exist

        logger.warning(f'New graph embeddings: {train_embeddings.shape}')

        train_idx_to_s2orc_paper_id = {idx: pid for idx, pid in enumerate(train_s2orc_paper_ids)}
        train_s2orc_paper_id_to_idx = {pid: idx for idx, pid in enumerate(train_s2orc_paper_ids)}

    elif graph_limit == BaseCorpus.CUSTOM:
        logger.info('Using custom graph limit')

        if triples_miner_args.ann_index_path is None or not os.path.exists(triples_miner_args.ann_index_path):
            raise ValueError(f'Custom graph limit needs precomputed ANN: {triples_miner_args.ann_index_path}')

        train_embeddings = graph_embeddings  # do not filter if ANN exist

        train_idx_to_s2orc_paper_id = {idx: pid for idx, pid in enumerate(train_s2orc_paper_ids)}
        train_s2orc_paper_id_to_idx = {pid: idx for idx, pid in enumerate(train_s2orc_paper_ids)}

    else:
        raise ValueError(f'Unsupported graph limit: {graph_limit}')

    # Query sampling
    if sample_queries_ratio != 1.0:
        if sample_queries_ratio < 1:
            # down sampling
            sample_n = int(len(train_query_s2orc_paper_ids) * sample_queries_ratio)
            logger.info(f'Down sampling to {sample_n} ({sample_queries_ratio})')

            # if sample_queries_ratio_with_specter:
            #     logger.info(f'Sampling from SPECTER queries')
            #     raise NotImplementedError()
            # else:

            # Sample from all previous generated queries
            train_query_s2orc_paper_ids = random.sample(train_query_s2orc_paper_ids, sample_n)

            # save to disk for reproduciblity
            if train_query_s2orc_paper_ids_path:
                with open(train_query_s2orc_paper_ids_path.replace('.json', f'.sample_{sample_queries_ratio}.json'), 'w') as f:
                    json.dump(
                        train_query_s2orc_paper_ids,
                        f
                    )
        else:
            # up sampling
            raise NotImplementedError()

    return get_generic_triples(train_s2orc_paper_id_to_idx,
                               train_idx_to_s2orc_paper_id,
                               train_query_s2orc_paper_ids,
                               train_embeddings,
                               # graph_paper_ids_path,
                               output_path,
                               triples_miner_args=triples_miner_args,
                               workers=workers,
                               output_csv_header='query_paper_id,positive_id,negative_id',
                               )


def worker_extract_metadata(worker_id, batch_fns, needed_paper_ids, s2orc_metadata_dir):
    batch_metadata = []

    for batch_fn in tqdm(batch_fns, desc=f'Worker {worker_id}'):
        with open(os.path.join(s2orc_metadata_dir, batch_fn)) as batch_f:
            for i, line in enumerate(batch_f):
                meta = json.loads(line)

                if meta['paper_id'] in needed_paper_ids:
                    batch_metadata.append({f: meta[f] for f in S2ORC_METADATA_FIELDS})

    return batch_metadata


def worker_extract_metadata_with_lines(worker_id, batch_fns_with_lines, s2orc_metadata_dir):
    batch_metadata = []

    for batch_fn, needed_lines in tqdm(batch_fns_with_lines, desc=f'Worker {worker_id}'):
        needed_lines = set(needed_lines)

        with open(os.path.join(s2orc_metadata_dir, batch_fn)) as batch_f:
            for i, line in enumerate(batch_f):
                if i in needed_lines:
                    meta = json.loads(line)
                    batch_metadata.append({f: meta[f] for f in S2ORC_METADATA_FIELDS})

    return batch_metadata


def get_plaintext_from_metadata(jsonl_metadata_path: str, output_path: str, override: bool = False):
    """
    Saves titles and abstract from papers as plaintext to disk (for language modeling).

    Format: "<title>: <abstract>\n"

    Usage:

    python cli_triples.py get_plaintext_from_metadata ${QUERY_DIR}/s2orc_paper_ids.json.metadata.jsonl ${QUERY_DIR}/s2orc_paper_ids.json.metadata.txt

    :param override: Override existing output file
    :param jsonl_metadata_path: Path to metadata JSONL file (see get_metadata)
    :param output_path: Save txt file at this location
    :return:
    """
    if os.path.exists(output_path) and not override:
        logger.error(f'Output exists already: {output_path}')
        return

    logger.info(f'Extracting metadata from JSONL: {jsonl_metadata_path}')
    logger.info(f'Writing output to {output_path}')

    with open(output_path, 'w') as out_f:
        with open(jsonl_metadata_path) as f:
            for line in f:
                paper = json.loads(line)

                out_f.write(paper['title'] + ": ")
                out_f.write((paper['abstract'] or '') + "\n")

    logger.info('done')


def get_metadata(input_path, output_path, s2orc_metadata_dir, workers: int = 10, id_mapping_path: str = None,
                 jsonl_metadata_path: str = None):
    """
    Extract meta data from S2ORC for triples

    python cli_triples.py get_metadata ${EXP_DIR}/train_triples.csv ${EXP_DIR}/train_metadata.jsonl \
        --s2orc_metadata_dir ${S2ORC_METADATA_DIR} --workers 50

    python cli_triples.py get_metadata ${OLD_DIR}/train_triples.csv ${OLD_DIR}/train_metadata.jsonl \
        --s2orc_metadata_dir ${S2ORC_METADATA_DIR} --workers 10

    python cli_triples.py get_metadata ${GRIDSEARCH_DIR}/s2orc_paper_ids.json ${GRIDSEARCH_DIR}/s2orc_paper_ids.json.metadata.jsonl \
        --s2orc_metadata_dir ${S2ORC_METADATA_DIR} --workers 50


    :param jsonl_metadata_path: If this JSONL path is provided, S2ORC is not used
    :param id_mapping_path: Path to ID Mapping JSON (default: s2orc_metadata_dir / s2orc_metadata_id_mapping.json)
    :param workers: Number of threads for parallel processing
    :param input_path: CSV with triples or JSON with IDs
    :param s2orc_metadata_dir: S2ORC metadata directory (.jsonl.gz files)
    :param output_path: Save JSONL file with metadata at this path
    :return:
    """

    # Default (see cli_s2orc.get_metadata_id_mapping)
    if id_mapping_path is None:
        id_mapping_path = os.path.join(s2orc_metadata_dir, 's2orc_metadata_id_mapping.json')

    if input_path.endswith('.json'):
        logger.info(f'Loading IDs from JSON: {input_path}')

        with open(input_path) as f:
            needed_paper_ids = set(json.load(f))

    else:
        # load triples from disk
        triples = []

        with open(input_path) as f:
            for i, line in enumerate(f):
                if i > 0:
                    triples.append(line.strip().split(','))

        logger.info(f'Loaded {len(triples):,} triples from {input_path}')

        needed_paper_ids = set([pid for triple in triples for pid in triple])

    logger.info(f'Needed metadata for {len(needed_paper_ids):,}')

    if jsonl_metadata_path and os.path.exists(jsonl_metadata_path):
        logger.info(f'Extracting metadata from JSONL: {jsonl_metadata_path}')

        train_metadata = []
        with open(jsonl_metadata_path) as f:
            for line in f:
                paper = json.loads(line)

                if paper['paper_id'] in needed_paper_ids:
                    train_metadata.append(paper)

    else:
        logger.info(f'Extracting metadata from S2ORC: {s2orc_metadata_dir}')

        # Meta data files
        batch_fns = [batch_fn for batch_fn in os.listdir(s2orc_metadata_dir) if batch_fn.endswith('.jsonl.gz')]
        logger.info(f'Files available: {len(batch_fns):,}')

        # Does a mapping file exists?
        if os.path.exists(id_mapping_path):
            # The mapping allows parsing only those JSON lines that contained needed metadata
            # (this should be faster than parsing all lines and deciding then based on ID)
            # ---
            # on serv-9212
            # +5min for reading the mapping file
            # +2min for extracting and writing metadata
            #
            # on  RTX309@slurm
            # +4:30min for reading mapping file
            # +2min for extracing and writing metadata
            #
            # ==> actually not faster use no mapping!
            logger.warning('DO NOT USE ID MAPPING SINCE IT IS SLOWER!')

            with open(id_mapping_path) as f:
                logger.info(f'Reading ID mapping from: {id_mapping_path}')
                id_mapping = json.load(f)  # batch_fn => [id, line_idx]

                batch_fn_to_needed_lines = defaultdict(list)  # batch_fn => list of line_idx

                # rewrite mapping
                paper_id_to_batch_fn_line_idx = {}
                for batch_fn, papers in id_mapping.items():
                    for paper_id, idx in papers:
                        paper_id_to_batch_fn_line_idx[paper_id] = [batch_fn, idx]

                for paper_id in needed_paper_ids:
                    batch_fn, line_idx = paper_id_to_batch_fn_line_idx[paper_id]
                    batch_fn_to_needed_lines[batch_fn].append(line_idx)

                batch_fn_to_needed_lines_list = list(batch_fn_to_needed_lines.items())

                logger.info(f'Extracting metadata with {workers} workers from {len(batch_fn_to_needed_lines_list)} files')

                # worker_id, batch_fns_with_lines, s2orc_metadata_dir
                worker_data = zip(
                    list(range(workers)),  # worker ids
                    split_into_n_chunks(batch_fn_to_needed_lines_list, workers),
                    # static arguments (same for all workers)
                    [s2orc_metadata_dir] * workers,
                )

                # Run threads
                with Pool(workers) as pool:
                    pool_outputs = list(pool.starmap(worker_extract_metadata_with_lines, worker_data))

        else:
            # Read all lines and check based on `needed_paper_ids`
            # ---
            # on serv-9212
            # + 4:30min
            #
            # on RTX309@slurm
            # + 4min
            logger.info(f'Extracting metadata with workers: {workers} (all files + lines)')

            # worker_id, batch_fns, needed_paper_ids, s2orc_metadata_dir
            worker_data = zip(
                list(range(workers)),  # worker ids
                split_into_n_chunks(batch_fns, workers),
                # static arguments (same for all workers)
                [needed_paper_ids] * workers,
                [s2orc_metadata_dir] * workers,
            )

            # Run threads
            with Pool(workers) as pool:
                pool_outputs = list(pool.starmap(worker_extract_metadata, worker_data))

        # Merge thread outputs
        train_metadata = [i for b in pool_outputs for i in b]

    logger.info(f'Metadata parsed. {len(train_metadata):,} train_metadata')

    if output_path:
        # Write to disk
        logger.info(f'Writing {len(train_metadata):,} metadata to {output_path}')

        with open(os.path.join(output_path), 'w') as f:
            for paper in train_metadata:
                f.write(json.dumps(paper) + '\n')

        logger.info('done')

    else:
        return train_metadata


def extract_ids_from_triples(
        triples_input_path: str,
        output_dir: str,
        paper_ids_file_name: str = 's2orc_paper_ids.json',
        query_paper_ids_file_name: str = 'query_s2orc_paper_ids.json',
    ):
    """
    Extract original IDs used by the triples (for SPECTER ids or debugging)

    Examples:

    python cli_triples.py extract_ids_from_triples ${SPECTER_DIR}/train_triples.csv ${SPECTER_DIR}/s2 s2_ids.csv query_s2_ids.csv

    python cli_triples.py extract_ids_from_triples ./data/scigdt/specter/gdt.epoch_20_768d.easy_3.hard_2.k_500/_original__train_triples.csv ./data/scigdt/specter/gdt.epoch_20_768d.easy_3.hard_2.k_500

    :param paper_ids_file_name: File name
    :param query_paper_ids_file_name: File name
    :param triples_input_path: Path to train_triples.csv
    :param output_dir: Write `s2orc_paper_ids.json` and `query_s2orc_paper_ids.json` into this directory.
    :return:
    """

    # load triples from disk
    all_ids = []
    query_ids = []

    logger.info(f'Reading from {triples_input_path}')

    with open(os.path.join(triples_input_path)) as f:
        for i, line in enumerate(f):
            if i > 0:
                triple = line.strip().split(',')

                all_ids += triple
                query_ids.append(triple[0])

    # unique
    all_ids = set(all_ids)
    query_ids = set(query_ids)

    logger.info(f'All IDs: {len(all_ids):,}; Query IDs: {len(query_ids):,}')

    # write to disk
    with open(os.path.join(output_dir, paper_ids_file_name), 'w') as f:
        if paper_ids_file_name.endswith('.csv'):
            f.write('\n'.join(list(all_ids)))
        else:
            json.dump(list(all_ids), f)
    with open(os.path.join(output_dir, query_paper_ids_file_name), 'w') as f:
        if query_paper_ids_file_name.endswith('.csv'):
            f.write('\n'.join(list(query_ids)))
        else:
            json.dump(list(query_ids), f)


def get_specter_like_triples(citations_input_path, query_papers: Union[str, int], triples_count: int,
                             output_path: str, override: bool = False, **triples_miner_kwargs):
    """
    Generate triples like proposed by the SPECTER paper:
    - positives = cited by the query
    - hard negatives = not cited by the query, but cited by the papers that are cited by the query (citations of citations)
    - easy negatives = any random paper that is not cited by the query

    Example:

    python cli_triples.py get_specter_like_triples \
        --citations_input_path ${BASE_DIR}/data/biggraph/s2orc_without_scidocs/citations.tsv \
        --query_papers 136820 \
        --triples_count 684100 \
        --output_path ${BASE_DIR}/v2_sci/specter/s2orc_without_scidocs/seed_0/train_triples.csv \
        --triples_per_query 5 --easy_negatives_count 3 --hard_negatives_count 2 --easy_positives_count 5
        --seed 0

    :param override:
    :param output_path:
    :param citations_input_path:
    :param query_papers:
    :param triples_count:
    :param triples_miner_kwargs:
    :return:
    """
    if os.path.exists(output_path) and not override:
        raise FileExistsError(f'Output exist already and --override is not set: {output_path}')

    triples_miner_args = TriplesMinerArguments(**triples_miner_kwargs)

    set_seed(triples_miner_args.seed)

    if citations_input_path.endswith('.json'):
        # read from JSON
        with open(citations_input_path) as f:
            paper_id_to_cits = json.load(f)
    else:
        # read from CSV
        cits = read_citations_from_tsv(citations_input_path)

        logger.info(f'Total citations: {len(cits)}')

        paper_id_to_cits = defaultdict(list)  # use list it's faster
        for from_id, to_id in tqdm(cits, total=len(cits), desc='Convert citations'):
            paper_id_to_cits[from_id].append(to_id)

        # write JSON
        with open(citations_input_path + '.mapping.json', 'w') as f:
            json.dump(paper_id_to_cits, f)

    # All papers in corpus
    paper_id_list = list(paper_id_to_cits.keys())

    if isinstance(query_papers, str):
        # Read paper IDs from JSON file
        with open(query_papers) as f:
            query_papers = json.load(f)

        logger.info(f'Loaded {len(query_papers):,} from file')
    elif isinstance(query_papers, int):
        # Sample N query papers from
        sample_n = query_papers * 2   # some extra queries to have matching counts

        logger.info(f'Sampling {sample_n} query papers (to have {query_papers:,})')

        query_papers = random.sample(paper_id_list, sample_n)
    else:
        raise ValueError('Invalid query papers provided')

    query_to_positives = defaultdict(set)
    query_to_hard_negatives = defaultdict(set)

    for qid in tqdm(query_papers, total=len(query_papers), desc='Generate samples for each query'):
        if qid in paper_id_to_cits:
            neighbors = set(paper_id_to_cits[qid])

            if neighbors:
                for n in neighbors:
                    query_to_positives[qid].add(n)

                    if n in paper_id_to_cits:
                        neighbors_of_neighbors = set(paper_id_to_cits[n])

                        for nn in neighbors_of_neighbors:
                            if nn == qid:
                                continue

                            if nn not in neighbors:
                                query_to_hard_negatives[qid].add(nn)

                            if len(query_to_hard_negatives[qid]) > triples_miner_args.hard_negatives_count:
                                break

                    if len(query_to_positives[qid]) > triples_miner_args.triples_per_query and len(
                            query_to_hard_negatives[qid]) > triples_miner_args.hard_negatives_count:
                        break

    triples = []

    for qid in tqdm(query_papers, desc='Generate triples'):
        if len(query_to_hard_negatives[qid]) >= triples_miner_args.hard_negatives_count and len(query_to_positives[qid]) >= triples_miner_args.triples_per_query:
            positives = random.sample(list(query_to_positives[qid]), triples_miner_args.triples_per_query)

            # hard
            hard_negatives = random.sample(list(query_to_hard_negatives[qid]), triples_miner_args.hard_negatives_count)

            # easy
            easy_negatives = []

            while len(easy_negatives) < triples_miner_args.easy_negatives_count:
                candidate = random.choice(paper_id_list)
                if candidate not in paper_id_to_cits[qid]:
                    easy_negatives.append(candidate)

            negatives = hard_negatives + easy_negatives

            query_triples = list(zip(
                [qid] * triples_miner_args.triples_per_query,  # repeat query
                positives,
                negatives
            ))

            triples += query_triples

            if len(triples) == triples_count:
                logger.info('Enough triples genreated. Stop')
                break

    # Write to disk
    logger.info(f'Writing {len(triples):,} triples to {output_path}')

    with open(output_path, 'w') as f:
        f.write('query_paper_id,positive_id,negative_id\n')

        for q, p, n in triples:
            f.write(f'{q},{p},{n}\n')

    logger.info('done')


def get_scidocs_triples(scidocs_dir: str,
                        s2id_to_s2orc_input_path: str,
                        output_path: str,
                        override: bool = False,
                        sample_classification_triples: Optional[int] = 0,
                        val_or_test_or_both: str = 'test', tasks: Optional[str] = None,
                        output_csv_header: str = 'query_paper_id,positive_id,negative_id'):
    """
    Generate "Oracle triples" based on SciDocs test data

    Example:

    python cli_triples.py get_scidocs_triples ${SCIDOCS_DIR} ${ID_MAPPINGS} ./data/scidocs_triples/train_triples.csv \
        --override --val_or_test_or_both test

     >> Writing 2,540,785 triples to ./data/scidocs_triples/train_triples.csv

    # Generate triples (downsample classification tasks for to match SPECTER triples count; expected_triples_count = 684100)
    python cli_triples.py get_scidocs_triples ${SCIDOCS_DIR} ${ID_MAPPINGS} ${QUERY_DIR}/train_triples.csv \
        --override --val_or_test_or_both test --sample_classification_triples 335000

    # Generate triples for a task-balanced task dataset (otherwise classification is overrepresented)
    python cli_triples.py get_scidocs_triples ${SCIDOCS_DIR} ${ID_MAPPINGS} ./data/scidocs_triples/both_20k_cls/train_triples.csv \
        --sample_classification_triples 20000 --override --val_or_test_or_both both

    mkdir -p ./data/scidocs_triples/test_20k_cls
    python cli_triples.py get_scidocs_triples ${SCIDOCS_DIR} ${ID_MAPPINGS} ./data/scidocs_triples/test_20k_cls/train_triples.csv \
        --sample_classification_triples 20000 --override --val_or_test_or_both test

    # 2022-01-29 14:01:02 - INFO - __main__ -   - val_recomm => 983 triples
    # 2022-01-29 14:01:02 - INFO - __main__ -   - val_coread => 4,967 triples
    # 2022-01-29 14:01:02 - INFO - __main__ -   - val_coview => 4,980 triples
    # 2022-01-29 14:01:02 - INFO - __main__ -   - val_cocite => 4,955 triples
    # 2022-01-29 14:01:02 - INFO - __main__ -   - val_cite => 4,907 triples
    # 2022-01-29 14:01:02 - INFO - __main__ -   - val_mesh => 20,000 triples
    # 2022-01-29 14:01:02 - INFO - __main__ -   - val_mag => 20,000 triples
    # 2022-01-29 14:01:02 - INFO - __main__ -   - test_recomm => 964 triples
    # 2022-01-29 14:01:02 - INFO - __main__ -   - test_coread => 4,977 triples
    # 2022-01-29 14:01:02 - INFO - __main__ -   - test_coview => 4,978 triples
    # 2022-01-29 14:01:02 - INFO - __main__ -   - test_cocite => 4,949 triples
    # 2022-01-29 14:01:02 - INFO - __main__ -   - test_cite => 4,928 triples
    # 2022-01-29 14:01:02 - INFO - __main__ -   - test_mesh => 20,000 triples
    # 2022-01-29 14:01:02 - INFO - __main__ -   - test_mag => 20,000 triples
    # Writing 120,114 triples to ./data/scidocs_triples/both_20k_cls/train_triples.csv

    # Train model with SciDocs triples
    export QUERY_DIR=${BASE_DIR}/data/scidocs_triples/both_20k_cls
    export QUERY_DIR=${BASE_DIR}/data/scidocs_triples/test_20k_cls

    ${PY} cli_pipeline.py run_specter ${QUERY_DIR} \
            --scidocs_dir ${SCIDOCS_DIR} \
            --s2orc_metadata_dir ${S2ORC_METADATA_DIR} \
            --val_or_test_or_both test \
            --eval_steps 2 --save_steps 2 \
            --workers ${WORKERS} \
            --base_model_name_or_path ${BASE_MODEL} --seed 0 --skip-queries --skip-triples

    :param output_csv_header:
    :param s2id_to_s2orc_input_path:
    :param sample_classification_triples:
    :param tasks:
    :param val_or_test_or_both:
    :param scidocs_dir:
    :param output_path:
    :param override:
    :return:
    """

    if os.path.exists(output_path) and not override:
        raise FileExistsError(f'Output exist already and --override not set')

    if tasks is None:
        tasks = ['recomm', 'coread', 'coview', 'cocite', 'cite', 'mesh', 'mag']

    if val_or_test_or_both == 'both':
        ds_list = ['val', 'test']
    else:
        ds_list = [val_or_test_or_both]

    ds_task_to_triples = defaultdict(list)

    for ds in ds_list:
        logger.info(f'Dataset {ds}')

        for task in tasks:
            logger.info(f'Task: {task}')

            if task == 'recomm':
                ds_task_to_triples[ds + '_' + task] += get_recomm_triples(scidocs_dir + f'/{task}/{ds}.csv')

            elif task in ['coread', 'coview', 'cocite', 'cite']:
                ds_task_to_triples[ds + '_' + task] += get_triples_from_qrel(scidocs_dir + f'/{task}/{ds}.qrel')

            elif task in ['mesh', 'mag']:
                task_triples = get_triples_from_class_csv(scidocs_dir + f'/{task}/{ds}.csv')

                if sample_classification_triples > 0:
                    logger.info(f'Sampling classification triples to {sample_classification_triples}')

                    task_triples = random.sample(task_triples, sample_classification_triples)

                ds_task_to_triples[ds + '_' + task] += task_triples
            else:
                raise ValueError(f'Invalid task: {task}')

    triples = []

    for ds_task, ts in ds_task_to_triples.items():
        logger.info(f'- {ds_task} => {len(ts):,} triples')

        triples += ts

    # S2-S2ORC Mappings
    s2id_to_s2orc_paper_id = read_json_mapping_files(s2id_to_s2orc_input_path)

    triples = [(s2id_to_s2orc_paper_id[q], s2id_to_s2orc_paper_id[p], s2id_to_s2orc_paper_id[n]) for q, p, n in triples if
     q in s2id_to_s2orc_paper_id and p in s2id_to_s2orc_paper_id and n in s2id_to_s2orc_paper_id]

    # write to disk
    logger.info(f'Writing {len(triples):,} triples to {output_path}')

    with open(os.path.join(output_path), 'w') as f:
        f.write(output_csv_header + '\n')
        for query_paper_id, pos_id, neg_id in triples:
            f.write(f'{query_paper_id},{pos_id},{neg_id}\n')


if __name__ == '__main__':
    fire.Fire()
