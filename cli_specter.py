import enum
import inspect
import json
import logging
import pickle
import random
from math import ceil
from typing import Union, List, Tuple, Optional

import fire
from smart_open import open
from transformers import set_seed

from gdt.utils import get_scidocs_metadata, write_func_args, read_json_mapping_files

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class BaseCorpus(str, enum.Enum):
    SPECTER = 'specter'  # Replicates SPECTER's training corpus (training IDs and queries)
    SCIDOCS = 'scidocs'  # Uses SciDocs papers for training (NOTE: this is just for debugging)
    S2ORC = 's2orc'  # Samples training and query papers from S2ORC ("specter sized")
    GRAPH = 'graph'  # Use corpus as defined by citation graph (e.g., full S2ORC without SciDocs filtered by citations)
    NONE = 'none'  # Do not sample any additional training or query papers
    CUSTOM = 'custom'  # Any other custom setting which needs other parameters TODO


def extract_triples(
        pickle_path: str,
        output_path=None,
        output_csv_sep: str = ',',
        output_csv_header: str = 'query_paper_id,positive_id,negative_id'
        ):
    """
    Extract paper IDs from SPECTER train/val data (so we can reproduce their training).

    See https://github.com/allenai/specter/issues/2#issuecomment-625428992

    Usage:

    python cli_specter.py extract_triples ./data/specter/train.pkl ./data/specter/train_triples.csv

    :param output_csv_header:
    :param output_csv_sep: Column separator for output CSV
    :param output_path: Write triples as CSV
    :param pickle_path: Path to DataReaderFromPickled file (e.g., ./data/specter/train.pkl)
    :return: List of triples (query, positive, negatives)
    """

    logger.info(f'Parsing {pickle_path}')

    from gdt.datasets.specter import DataReaderFromPickled

    datareaderfp = DataReaderFromPickled(max_sequence_length=512)
    data_instances = datareaderfp._read(pickle_path)

    query_paper_ids = []
    pos_paper_ids = []
    neg_paper_ids = []

    for item in data_instances:
        query_paper_ids.append(item['source_paper_id'].metadata)
        pos_paper_ids.append(item['pos_paper_id'].metadata)
        neg_paper_ids.append(item['neg_paper_id'].metadata)

    logger.info(f'Triples extracted: {len(query_paper_ids):,}')

    triples = zip(query_paper_ids, pos_paper_ids, neg_paper_ids)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(output_csv_header + '\n')

            for t in triples:
                f.write(output_csv_sep.join(t) + '\n')
    else:
        return list(triples)


def filter_by_inbound_citations(inbound_citations_mapping_path, min_inbound_citations, max_inbound_citations):
    """
    Return S2ORC paper ids based on citation filter.

    :param inbound_citations_mapping_path: Path to output of `cli_s2orc.get_inbound_citations_count` (dict with str (paper_id), int (count))
    :param min_inbound_citations:
    :param max_inbound_citations:
    :return:
    """
    logger.info(f'Loading inbound citations from: {inbound_citations_mapping_path}')

    with open(inbound_citations_mapping_path) as f:
        paper_id_to_inbound_citations_count = json.load(f)

    # Filter by citation count
    logger.info(f'Min/max citation count: {min_inbound_citations} <= count <= {max_inbound_citations}')
    s2orc_paper_ids_set_with_valid_inbound_citations_count = {
        paper_id
        for paper_id, count in paper_id_to_inbound_citations_count.items()
        if min_inbound_citations <= count <= max_inbound_citations
    }
    s2orc_paper_ids_set = s2orc_paper_ids_set_with_valid_inbound_citations_count

    logger.info(f'After filtering by citations => S2ORC paper IDs: {len(s2orc_paper_ids_set):,}')

    return s2orc_paper_ids_set


def find_train_ids(
        specter_triples_path: str,
        scidocs_dir: str,
        s2id_to_s2orc_input_path: str,
        s2orc_paper_ids: Union[str, List[str]],
        output_path: str = None,
        query_output_path: str = None,
        query_n_folds: int = 0,
        query_fold_k: Union[int, List[int], str] = 0,
        query_oversampling_ratio: float = 0.0,
        seed: int = 0,
        k_means_dir: Optional[str] = None,
        min_inbound_citations: int = 0,
        max_inbound_citations: int = 0,
        inbound_citations_filter: Optional[str] = None,
        inbound_citations_mapping_path: str = None,
        base_corpus: BaseCorpus = BaseCorpus.SPECTER,
        map_specter_to_s2orc: bool = True,
        ) -> Tuple[List[str], List[str]]:
    """
    Finds papers that should be part of training corpus (all papers) and query papers.

    We cannot use exactly the same training data as SPECTER since not all papers used by SPECTER are also in S2ORC.

    Find papers:
    - graph embeddings: all papers in SPECTER training + random (if needed; not in SciDocs)
    - query paper ids: all papers in SPECTER training + random (if needed; not in SciDocs)

    Down sample for few-shot learning:
    - sample_ratio
    - sample_n

    SPECTER papers count:
    - train papers = 311,860
    - query papers =  136,820

    Unique S2 IDs in Scidocs: 223,932

    Usage:

    python cli_specter.py find_train_ids ${SPECTER_DIR}/train_triples.csv \
        --scidocs_dir ${SCIDOCS_DIR} \
        --s2id_to_s2orc_input_path ${SPECTER_DIR}/s2id_to_s2orc_paper_id.json \
        --s2orc_paper_ids ${S2ORC_PAPER_IDS} \

    python cli_specter.py find_train_ids --specter_triples_path ./data/specter/train_triples.csv \
        --scidocs_dir /home/mostendorff/experiments/scidocs/data \
        --s2id_to_s2orc_input_path ./data/specter/s2id_to_s2orc_paper_id.json \
        --s2orc_paper_ids ${S2ORC_PAPER_IDS} \
        --query_fold_k 0,1,2,3,4 --query_n_folds 10 \
        ./data/gdt/

    python cli_specter.py find_train_ids ${SPECTER_DIR}/train_triples.csv ${SCIDOCS_DIR} \
        --s2id_to_s2orc_input_path ${SPECTER_DIR}/s2id_to_s2orc_paper_id.json \
        --s2orc_paper_ids ${S2ORC_PAPER_IDS} \
        --output_path ${QUERY_DIR}/s2orc_paper_ids.json \
        --query_output_path ${QUERY_DIR}/query_s2orc_paper_ids.json \
        --inbound_citations_mapping_path ${SPECTER_DIR}/s2orc_inbound_citations.json.gz \
        --inbound_citations_filter train --min_inbound_citations 5 --max_inbound_citations 500

    :param map_specter_to_s2orc: Perform mapping from SPECTER's S2IDs to S2ORC IDs (disable if input IDs are already in S2ORC)
    :param base_corpus: Training (and query) paper IDs are selected from this corpus (see BaseCorpus)
    :param exclude_specter: If enabled, IDs are sampled completely random without starting from SPECTER IDs.
    :param inbound_citations_mapping_path: Path to output of `cli_s2orc.get_inbound_citations_count` (dict with str (paper_id), int (count))
    :param inbound_citations_filter: Set to `train` or `query` to enable filter by min/max inbound citations.
    :param max_inbound_citations: Max. inbound citations count
    :param min_inbound_citations: Min. inbound citations count
    :param k_means_dir: Load k-means data from this directory; If is set, sampling from diverse centroids.
    :param query_oversampling_ratio: Sample additional n query papers, where n = ratio * specter_query_papers.
    :param query_output_path: Saves query paper IDs as JSON
    :param output_path: Saves paper IDs as JSON
    :param seed: Random seed
    :param query_fold_k: K-fold split for query papers
    :param query_n_folds: Number of folds for splitting query papers (0 = no splitting)
    :param s2id_to_s2orc_input_path:
    :param scidocs_dir:
    :param specter_triples_path:
    :param s2orc_paper_ids: Path to JSON or list of paper IDs
    :return: Tuple (papers in train corpus, query papers)
    """

    # Log arg settings
    write_func_args(inspect.currentframe(), output_path + '.args.json')

    set_seed(seed)

    if isinstance(query_fold_k, str):
        query_fold_k = [int(k) for k in query_fold_k.split(',')]  # Convert string to int list

    # S2-S2ORC Mappings
    s2id_to_s2orc_paper_id = read_json_mapping_files(s2id_to_s2orc_input_path)

    # S2ORC paper ids (available in citation graph)
    if isinstance(s2orc_paper_ids, str):
        # load from disk
        with open(s2orc_paper_ids) as f:
            s2orc_paper_ids = json.load(f)

    s2orc_paper_ids_set = set(s2orc_paper_ids)

    logger.info(f'S2ORC paper IDs (in graph): {len(s2orc_paper_ids):,}')

    # Inbound citation filter
    if inbound_citations_filter == 'train':
        s2orc_paper_ids_set = filter_by_inbound_citations(inbound_citations_mapping_path, min_inbound_citations, max_inbound_citations)

    elif inbound_citations_filter == 'query':
        raise NotImplementedError()

    # SciDocs Metadata
    scidocs_metadata = get_scidocs_metadata(scidocs_dir)
    scidocs_s2_ids = {s2id for ds, ds_metadata in scidocs_metadata.items() for s2id in ds_metadata.keys()}
    logger.info(f'Scidocs - Unique S2 IDs: {len(scidocs_s2_ids):,}')

    # Map SciDocs IDs to S2ORC IDs
    scidocs_s2orc_paper_ids = {s2id_to_s2orc_paper_id[s2id] for s2id in scidocs_s2_ids if
                               s2id in s2id_to_s2orc_paper_id}

    logger.info(f'Scidocs - Successful mapped to S2ORC: {len(scidocs_s2orc_paper_ids):,} (missing: {len(scidocs_s2_ids - set(s2id_to_s2orc_paper_id.keys())):,})')

    # SPECTER train triples from disk (see `extract_triples`)
    with open(specter_triples_path) as f:
        specter_train_triples = [line.strip().split(',') for i, line in enumerate(f) if i > 0]

    logger.info(f'SPECTER - Loaded {len(specter_train_triples):,} triples from {specter_triples_path}')

    # Paper corpus (queries, positives, negatives)
    specter_train_s2ids = {i for t in specter_train_triples for i in t}  # all ids

    # SPECTER S2IDs to S2ORC IDs
    if map_specter_to_s2orc:
        specter_train_s2orc_paper_ids = {s2id_to_s2orc_paper_id[s2id] for s2id in specter_train_s2ids
                                         if s2id in s2id_to_s2orc_paper_id}  # Map to S2ORC IDs

        logger.info(f'SPECTER - papers with S2ORC ID: '
                    f'{len(specter_train_s2orc_paper_ids):,} / {len(specter_train_s2ids):,} (missing: {len(specter_train_s2ids - set(s2id_to_s2orc_paper_id.keys())):,})')
    else:
        logger.warning('SPECTER - Do not map to S2ORC Ids')
        specter_train_s2orc_paper_ids = specter_train_s2ids

    # Train IDs in graph
    specter_train_s2orc_paper_ids = specter_train_s2orc_paper_ids & s2orc_paper_ids_set  # Overlap of two sets

    logger.info(f'SPECTER - papers in graph: '
                f'{len(specter_train_s2orc_paper_ids):,} / {len(specter_train_s2ids):,}')

    if base_corpus == BaseCorpus.SPECTER:
        logger.info('Starting with SPECTER IDs')

        # How many papers are missing? In SPECTER but not in S2ORC
        missing_papers_count = len(specter_train_s2ids) - len(specter_train_s2orc_paper_ids)

        train_s2orc_paper_ids = specter_train_s2orc_paper_ids

        # Citation graph papers that are not in SciDocs and not in predefinied training papers
        candidate_s2orc_paper_ids = s2orc_paper_ids_set - scidocs_s2orc_paper_ids - train_s2orc_paper_ids

    elif base_corpus == BaseCorpus.S2ORC:

        logger.info('Starting without SPECTER IDs (start IDs from S2ORC)')
        train_s2orc_paper_ids = set()

        missing_papers_count = len(specter_train_s2ids)  # Same number as used by SPECTER

        # Citation graph papers that are not in SciDocs and not in predefinied training papers
        candidate_s2orc_paper_ids = s2orc_paper_ids_set - scidocs_s2orc_paper_ids - train_s2orc_paper_ids

    elif base_corpus == BaseCorpus.SCIDOCS:
        logger.warning(f'Using SciDocs as base corpus (debug only!)')
        # Take IDs from SciDocs
        missing_papers_count = 0
        train_s2orc_paper_ids = scidocs_s2orc_paper_ids
        candidate_s2orc_paper_ids = set()

    elif base_corpus == BaseCorpus.GRAPH:
        logger.info('Using papers from graph as training corpus')

        # Use all papers in graph for training -> do not sample any additional papers
        train_s2orc_paper_ids = s2orc_paper_ids_set
        missing_papers_count = 0
        candidate_s2orc_paper_ids = set()

    elif base_corpus == BaseCorpus.NONE:
        logger.info('No extra base corpus -> do not sample additional training papers')

        candidate_s2orc_paper_ids = set()
        missing_papers_count = 0
        train_s2orc_paper_ids = specter_train_s2orc_paper_ids  # use training paper IDs from triples
    else:
        raise ValueError(f'Unsupported base corpus: {base_corpus}')

    logger.info(f'Missing papers: {missing_papers_count:,} (available candidates: {len(candidate_s2orc_paper_ids):,})')

    if missing_papers_count > 0:
        logger.info('Not enough training papers, add random papers')

        logger.info(f'Candidates in citation graph: {len(candidate_s2orc_paper_ids):,}')

        # Random sample from candidates
        random_train_s2orc_paper_ids = random.sample(list(candidate_s2orc_paper_ids), missing_papers_count)

        logger.info(f'Randomly sampled train paper IDs: {len(random_train_s2orc_paper_ids):,}')

        # Add random papers to predefined papers
        train_s2orc_paper_ids = train_s2orc_paper_ids.union(random_train_s2orc_paper_ids)

    logger.info(f'Final train paper IDs: {len(train_s2orc_paper_ids):,}')

    # Query papers
    specter_train_query_s2ids = {q for q, p, n in specter_train_triples}  # query ids

    if map_specter_to_s2orc:
        specter_train_query_s2orc_ids = {s2id_to_s2orc_paper_id[s2id] for s2id in specter_train_query_s2ids
                                         if s2id in s2id_to_s2orc_paper_id and s2id_to_s2orc_paper_id[
                                             s2id] in s2orc_paper_ids_set}  # map to S2ORC IDs
    else:
        logger.warning('SPECTER query - do not map to S2ORC')
        specter_train_query_s2orc_ids = specter_train_query_s2ids

    # Expected query paper count should be equal to SPECTER
    expected_query_papers_count = len(specter_train_query_s2ids)

    logger.info(f'SPECTER - query papers with S2ORC ID and in graph: {len(specter_train_query_s2orc_ids):,} / {expected_query_papers_count:,}')

    if base_corpus == BaseCorpus.SPECTER:
        logger.info('Starting with query papers from SPECTER')
        train_query_s2orc_paper_ids = list(specter_train_query_s2orc_ids)

        # How many query papers are missing? In SPECTER but not in S2ORC
        missing_query_papers_count = expected_query_papers_count - len(specter_train_query_s2orc_ids)

        # random queries: sample from train corpus but exclude existing query papers
        query_candidates = train_s2orc_paper_ids - specter_train_query_s2orc_ids

    elif base_corpus == BaseCorpus.S2ORC or base_corpus == BaseCorpus.SCIDOCS:
        logger.info('Starting with query papers from scratch (no SPECTER but all S2ORC)')
        train_query_s2orc_paper_ids = []

        # Same number as in SPECTER
        missing_query_papers_count = expected_query_papers_count

        # random queries: sample from train corpus
        query_candidates = train_s2orc_paper_ids

    elif base_corpus == BaseCorpus.GRAPH:
        # Use only SPECTER query papers
        train_query_s2orc_paper_ids = list(specter_train_query_s2orc_ids)

        missing_query_papers_count = 0
        query_candidates = None

    elif base_corpus == BaseCorpus.NONE:
        logger.info('No extra base corpus -> no additional query papers')
        missing_query_papers_count = 0
        query_candidates = None
        train_query_s2orc_paper_ids = list(specter_train_query_s2orc_ids)  # from SPECTER triples
    else:
        raise ValueError(f'Invalid base corpus: {base_corpus}')

    logger.info(f'Missing query papers: {missing_query_papers_count:,}')

    if missing_query_papers_count > 0:
        logger.info(f'Not enough query papers.. randomly sample {missing_query_papers_count:,} papers')

        random_train_query_s2orc_paper_ids = random.sample(query_candidates, missing_query_papers_count)  # TODO sample based on k means centroids

        # Missing query papers
        train_query_s2orc_paper_ids += random_train_query_s2orc_paper_ids

        logger.info(f'Adding {len(random_train_query_s2orc_paper_ids):,} random papers')

    logger.info(f'Query papers: {len(train_query_s2orc_paper_ids):,}')

    if base_corpus == BaseCorpus.SPECTER or base_corpus == BaseCorpus.S2ORC or base_corpus == BaseCorpus.SCIDOCS:
        if len(train_query_s2orc_paper_ids) != expected_query_papers_count:
            # Check numbers when base corpus is defined
            raise ValueError(f'Invalid query paper count: '
                             f'train_query_s2orc_paper_ids = {len(train_query_s2orc_paper_ids):,}; '
                             f'expected_query_papers_count = {expected_query_papers_count:,}')

    random.shuffle(train_query_s2orc_paper_ids)  # shuffle to k-fold

    # Split query papers into folds
    if query_n_folds > 1:
        fold_size = ceil(len(train_query_s2orc_paper_ids) / query_n_folds)

        # select fold
        if isinstance(query_fold_k, int):
            offset = query_fold_k * fold_size
            end = (query_fold_k + 1)*fold_size
            train_query_s2orc_paper_ids = train_query_s2orc_paper_ids[offset:end]

            logger.info(f'Split slice: {offset}:{end}')
        elif isinstance(query_fold_k, list) or isinstance(query_fold_k, tuple):
            # Multiple k (e.g., k=0,1,2,3,4 => 50% at 10 splits]
            paper_folds = []
            for k in query_fold_k:
                offset = k * fold_size
                end = (k + 1) * fold_size

                paper_folds += train_query_s2orc_paper_ids[offset:end]
                logger.info(f'Split slice (k={k}): {offset}:{end}')

            train_query_s2orc_paper_ids = paper_folds
        else:
            raise ValueError(f'Cannot parse k = {query_fold_k}')

        logger.info(f'Split into {query_n_folds} folds (k={query_fold_k}; size={fold_size}): {len(train_query_s2orc_paper_ids):,} papers')
    else:
        logger.info('No split')

    if query_oversampling_ratio > 0:
        if query_n_folds > 1:
            logger.warning(f'Query oversampling should not be used, while fold splitting is enabled!')

        query_oversampling_n = ceil(len(train_query_s2orc_paper_ids) * query_oversampling_ratio)
        query_oversampling_candidates = train_s2orc_paper_ids - set(train_query_s2orc_paper_ids)

        logger.info(f'Over-sampling {query_oversampling_n:,} additional query papers '
                    f'(ratio={query_oversampling_ratio}; candidates={len(query_oversampling_candidates):,})')

        if query_oversampling_n > len(query_oversampling_candidates):
            # sample size is greater than population -> add candidates from S2ORC
            additional_candidates_n = query_oversampling_n - len(query_oversampling_candidates)
            logger.info(f'Candidates count too low, adding more candidates from S2ORC: '
                        f'{additional_candidates_n:,} + 25% (for duplicates)')

            query_oversampling_candidates = query_oversampling_candidates.union(
                random.sample(list(candidate_s2orc_paper_ids), ceil(additional_candidates_n * 1.1))
            )

            logger.info(f'Candidates: {len(query_oversampling_candidates):,}')

        train_query_s2orc_paper_ids += random.sample(query_oversampling_candidates, query_oversampling_n)

    else:
        logger.info('No oversampling')

    # Convert to lists
    train_s2orc_paper_ids = list(train_s2orc_paper_ids)
    train_query_s2orc_paper_ids = list(train_query_s2orc_paper_ids)

    if output_path is not None and query_output_path is not None:
        # write output to disk
        with open(output_path, 'w') as f:
            json.dump(train_s2orc_paper_ids, f)
        with open(query_output_path, 'w') as f:
            json.dump(train_query_s2orc_paper_ids, f)

        logger.info(f'Output saved to {output_path}; {query_output_path}')
    else:
        return train_s2orc_paper_ids, train_query_s2orc_paper_ids


def find_train_ids_with_k_means():
    # Select diverse train IDs based on k-means centroids
    raise NotImplementedError()


def shrink_dataset(input_pickle: str, output_pickle: str, n: int):
    """

    Take the first n instances from a SPECTER dataset (pickle format)

    https://github.com/allenai/specter/blob/673346f9f76bcf422b38e0d1b448ef4414bcd4df/specter/data_utils/create_training_files.py

    Usage:

    python cli_specter.py shrink_dataset ${SPECTER_DIR}/train.pkl ${SPECTER_DIR}/train_1k.pkl 1000
    python cli_specter.py shrink_dataset ${SPECTER_DIR}/train.pkl ${SPECTER_DIR}/train_0.1.pkl 73000
    python cli_specter.py shrink_dataset ${SPECTER_DIR}/train.pkl ${SPECTER_DIR}/train_0.5.pkl 365000
    python cli_specter.py shrink_dataset ${SPECTER_DIR}/train.pkl ${SPECTER_DIR}/train_0.9.pkl 657000

    python cli_specter.py shrink_dataset ${SPECTER_DIR}/val.pkl ${SPECTER_DIR}/val_1k.pkl 1000

    Train count: 730000
    - 10%: 73000
    - 50%: 365000
    - 90%: 657000

    :param input_pickle:
    :param output_pickle:
    :param n: Limit the dataset to n instances
    :return:
    """
    instances_count = 0

    with open(input_pickle, 'rb') as f_in:
        logger.info(f'Reading from {input_pickle}')

        with open(output_pickle, 'wb') as f_out:
            logger.info(f'Writing to {output_pickle}')

            unpickler = pickle.Unpickler(f_in)
            pickler = pickle.Pickler(f_out)

            while True:
                try:
                    instance = unpickler.load()
                    instances_count += 1

                    if instances_count > n:
                        logger.info('Dataset limit reached. Stopping...')
                        break

                    pickler.dump(instance)

                except EOFError:
                    logger.info('Input EOF')
                    break


if __name__ == '__main__':
    fire.Fire()

