import json
import logging
import os
import random
from collections import defaultdict
from multiprocessing import Pool
from typing import Union, List, Optional, Set

import fire
from sklearn.model_selection import train_test_split
from smart_open import open
from tqdm.auto import tqdm
from transformers import set_seed

from gdt.utils import get_scidocs_metadata, split_into_n_chunks, read_json_mapping_files

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def worker_extract_citations(worker_id,
                             batch_fps,
                             include_paper_ids_set: Optional[Set[str]] = None,
                             include_filter_outbound: bool = False,
                             exclude_paper_ids_set: Optional[Set[str]] = None,
                             exclude_filter_outbound: bool = False):
    batch_cits = []
    batch_paper_ids = []

    for batch_fp in batch_fps:
        logger.debug(f'Worker #{worker_id} reading from {batch_fp}')

        with open(batch_fp) as batch_f:
            for i, line in enumerate(batch_f):
                meta = json.loads(line)

                if include_paper_ids_set and meta['paper_id'] not in include_paper_ids_set:
                    # skip if not part of paper ids
                    continue

                if exclude_paper_ids_set and meta['paper_id'] in exclude_paper_ids_set:
                    # skip if the citing paper should be excluded
                    continue

                batch_paper_ids.append(meta['paper_id'])

                # TODO use outbound or inbound citations? or both?
                # => no difference

                if meta['has_outbound_citations']:
                    for to_id in meta['outbound_citations']:
                        if include_paper_ids_set and include_filter_outbound and to_id not in include_paper_ids_set:  # TODO make this optional?
                            # skip if cited paper is not part of paper ids
                            continue

                        if exclude_paper_ids_set and exclude_filter_outbound and to_id in exclude_paper_ids_set:
                            # skip if the cited paper should be excluded
                            continue

                        batch_cits.append((meta['paper_id'], to_id))

                # if meta['has_inbound_citations']:
                #    for from_id in meta['inbound_citations']:
                #        batch_cits.append((from_id, meta['paper_id']))

    return batch_paper_ids, batch_cits


def write_citations(cits, out_fp, nodes_count, col_sep, line_sep, description='', include_header: bool = True):
    logger.info(f'Writing {len(cits):,} citations to {out_fp}')

    header = f'# Directed graph\n' \
        f'# Description: {description}\n' \
        f'# Nodes: {nodes_count} Edges: {len(cits)}\n' \
        f'# FromNodeId{col_sep}ToNodeId\n'

    with open(out_fp, 'w') as f:
        if include_header:
            f.write(header)

        for from_id, to_id in tqdm(cits, desc='Writing to disk', total=len(cits)):
            f.write(from_id + col_sep + to_id + line_sep)


def get_citations(s2orc_metadata_dir: str,
                  output_dir=None,
                  workers: int = 10,
                  test_ratio: float = 0.,
                  sample_n_nodes: int = 0,
                  seed: int = 0,
                  included_paper_ids: Optional[Union[str, List[str], Set[str]]] = None,
                  excluded_paper_ids: Optional[Union[str, List[str], Set[str]]] = None,
                  description: str = 'Directed citation graph from S2ORC',
                  ):
    """
    Extracts citations from S2ORC metadata.

    Examples:

    - All S2ORC papers:

    python cli_s2orc.py get_citations /data/datasets/s2orc/20200705v1/full/metadata ./data/biggraph/s2orc_full/

    - Filter for SPECTER papers:

    python cli_s2orc.py get_citations /data/datasets/s2orc/20200705v1/full/metadata ./data/biggraph/specter_train/ \
        --included_paper_ids data/specter/train_s2orc_paper_ids.json

    - Exclude SciDocs:

    python cli_s2orc.py get_citations ${S2ORC_METADATA_DIR} ./data/biggraph/s2orc_without_scidocs/ \
        --excluded_paper_ids data/scidocs_s2orc/s2orc_paper_ids.json

    - Exclude SciDocs and sample 1 million nodes

    python cli_s2orc.py get_citations ${S2ORC_METADATA_DIR} ./data/biggraph/s2orc_without_scidocs_1m/ \
        --excluded_paper_ids data/scidocs_s2orc/s2orc_paper_ids.json  --sample_n_nodes 1000000

    :param description:
    :param sample_n_nodes: Random sub-sample of nodes (papers); edges (citations) are filtered accordingly.
    :param included_paper_ids: List or path to JSON with paper IDs that the only ones for that citations are extracted
    :param excluded_paper_ids: List or path to JSON with paper IDs that excluded from returned citations (citing or cited)
    :param seed: Set random seed for train/test split
    :param test_ratio: Ratio of citation edges that are used as test set
    :param s2orc_metadata_dir: S2ORC directory with metadata files (.jsonl.gz)
    :param output_dir: Write `citations.tsv` (`citations.train.csv` and `citations.test.csv`), `nodes.csv`
        and `paper_ids.csv` (papers that do not have any citations)
    :param workers: Number of threads for parallel processing
    :return: citations, nodes, paper_ids
    """
    set_seed(seed)

    if output_dir and not os.path.exists(output_dir):
        raise NotADirectoryError(f'Output directory does not exist')

    if isinstance(included_paper_ids, str):
        logger.info(f'Loading included paper IDs from {included_paper_ids}')
        with open(included_paper_ids) as f:
            included_paper_ids = json.load(f)

    # Convert to set
    if included_paper_ids is not None:
        if isinstance(included_paper_ids, set):
            included_paper_ids_set = included_paper_ids
        else:
            included_paper_ids_set = set(included_paper_ids)
    else:
        included_paper_ids_set = None

    if isinstance(excluded_paper_ids, str):
        logger.info(f'Loading excluded paper IDs from {excluded_paper_ids}')
        with open(excluded_paper_ids) as f:
            excluded_paper_ids = json.load(f)

    # Convert to set
    if excluded_paper_ids is not None:
        if isinstance(excluded_paper_ids, set):
            excluded_paper_ids_set = excluded_paper_ids
        else:
            excluded_paper_ids_set = set(excluded_paper_ids)
    else:
        excluded_paper_ids_set = None

    line_sep = '\n'
    col_sep = '\t'

    # Meta data files
    batch_fps = [os.path.join(s2orc_metadata_dir, batch_fn) for batch_fn in os.listdir(s2orc_metadata_dir) if
                 batch_fn.endswith('.jsonl.gz')]

    logger.info(f'Extracting citation from files: {len(batch_fps):,}')

    # worker_id, batch_fns, needed_paper_ids, s2orc_metadata_dir
    worker_data = zip(
        list(range(workers)),  # worker ids
        split_into_n_chunks(batch_fps, workers),
        [included_paper_ids_set] * workers,
        [False] * workers,  # apply filter only on citing papers but not on cited papers
        [excluded_paper_ids_set] * workers,
        [True] * workers,  # apply filter on both citing and cited papers
    )

    # Run threads
    with Pool(workers) as pool:
        pool_outputs = list(pool.starmap(worker_extract_citations, worker_data))

    # Merge thread outputs
    cits = [i for batch_paper_ids, batch_cits in pool_outputs for i in batch_cits]
    paper_ids = [i for batch_paper_ids, batch_cits in pool_outputs for i in batch_paper_ids]

    # 467,588,220 citations (outbound only)
    # 935,176,440 citations (in- and outbound)
    logger.info(f'Metadata parsed. {len(cits):,} citations')

    paper_ids_set = set(paper_ids)

    # Paper IDs: 136,595,995
    logger.info(f'Paper IDs: {len(paper_ids):,}')

    # Papers with citations
    nodes = set([i for cit in cits for i in cit])
    logger.info(f'Unique nodes (from citations): {len(nodes):,}')

    if sample_n_nodes is not None and sample_n_nodes > 0:
        logger.info(f'Sampling nodes from {len(nodes):,} to {sample_n_nodes:,} ...')
        nodes = random.sample(nodes, sample_n_nodes)

        logger.info(f'Filtering edges based on new nodes ...')
        cits = [(a, b) for a, b in cits if a in nodes and b in nodes]

        logger.info(f'New citations: {len(cits):,}')

    # Write to disk
    if output_dir and os.path.exists(output_dir):
        # Papers IDs
        with open(os.path.join(output_dir, 'paper_ids.csv'), 'w') as f:
            f.write('\n'.join(paper_ids))

        #  Unique nodes (outbound only): 52,620,852
        #  Unique nodes (in- and outbound): 52,620,852

        with open(os.path.join(output_dir, 'nodes.csv'), 'w') as f:
            f.write('\n'.join(nodes))

        if test_ratio > 0:
            logger.info(f'Splitting citations into train/test set: ratio = {test_ratio}')
            train_cits, test_cits = train_test_split(cits, test_size=test_ratio)

            logger.info(f'Train: {len(train_cits):,}; Test: {len(test_cits):,}')
            write_citations(train_cits, os.path.join(output_dir, 'citations.train.tsv'), len(nodes), col_sep, line_sep,
                            description=description + ' (Train)')
            write_citations(test_cits, os.path.join(output_dir, 'citations.test.tsv'), len(nodes), col_sep, line_sep,
                            description=description + ' (Test')
        else:
            write_citations(cits, os.path.join(output_dir, 'citations.tsv'), len(nodes), col_sep, line_sep,
                            description=description)

        logger.info('done')

    else:
        # output dir is not set, return instead
        return cits, nodes, paper_ids


def worker_extract_inbound_citations(batch_fp):
    """
    Worker method for `get_inbound_citations_count`
    """
    batch_cits_counts = []

    with open(batch_fp) as batch_f:
        for i, line in enumerate(batch_f):
            meta = json.loads(line)

            if meta['has_inbound_citations']:
                batch_cits_counts.append((meta['paper_id'], len(meta['inbound_citations'])))

    return batch_cits_counts


def get_inbound_citations_count(s2orc_metadata_dir: str, output_path: str, workers: int = 10):
    """
    Extracts inbound citation count from S2ORC and saves id-count mapping as JSON file.

    Usage:

    python cli_s2orc.py get_inbound_citations_count ${S2ORC_METADATA_DIR} ${SPECTER_DIR}/s2orc_inbound_citations.json.gz \
        --workers ${WORKERS}

    :param s2orc_metadata_dir: Directory with S2ORC metadata (.jsonl.gz) files
    :param output_path: Save JSON to this path
    :param workers: Number of threads
    :return:
    """
    if os.path.exists(output_path):
        logger.error(f'Output exists already: {output_path}')
        return

    # Meta data files
    batch_fps = [os.path.join(s2orc_metadata_dir, batch_fn) for batch_fn in os.listdir(s2orc_metadata_dir) if
                 batch_fn.endswith('.jsonl.gz')]

    logger.info(f'Files available: {len(batch_fps):,}')

    # Run threads
    with Pool(workers) as pool:
        pool_outputs = list(
            tqdm(pool.imap_unordered(worker_extract_inbound_citations, batch_fps), total=len(batch_fps)))

    cits_counts = {pid: count for b in pool_outputs for pid, count in b}

    logger.info(f'Extracted citation counts for {len(cits_counts):,} papers')

    with open(output_path, 'w') as f:
        json.dump(cits_counts, f)

    logger.info(f'Saved to {output_path}')


def get_scidocs_title_mapping(scidocs_dir, s2orc_metadata_dir, output_fp, workers: int = 10):
    """
    Find S2ORC paper ids based on title

    python cli_s2orc.py get_scidocs_title_mapping /home/mostendorff/experiments/scidocs/data \
        /data/datasets/s2orc/20200705v1/full/metadata ./data/scidocs_s2id_to_s2orc_paper_id.json

    :param scidocs_dir:
    :param s2orc_metadata_dir:
    :param output_fp:
    :param workers:
    :return:
    """

    # SciDocs Metadata
    scidocs_metadata = get_scidocs_metadata(scidocs_dir)

    scidocs_paper_ids = set(
        [paper_id for ds, ds_meta in scidocs_metadata.items() for paper_id, paper_meta in ds_meta.items()])

    logger.info(f'scidocs_paper_ids = {len(scidocs_paper_ids):,}')

    scidocs_titles = [paper_meta['title'] for ds, ds_meta in scidocs_metadata.items() for paper_id, paper_meta in
                      ds_meta.items()]

    logger.info(f'scidocs_titles = {len(scidocs_titles):,}')

    unique_scidocs_titles = set(scidocs_titles)

    logger.info(f'unique_scidocs_titles = {len(unique_scidocs_titles):,}')

    # S2ORC Meta data files
    batch_fns = [batch_fn for batch_fn in os.listdir(s2orc_metadata_dir) if batch_fn.endswith('.jsonl.gz')]

    logger.info(f'Files available: {len(batch_fns):,}')

    def worker_extract_matching_titles(batch_fn):
        batch = []

        with open(os.path.join(s2orc_metadata_dir, batch_fn)) as batch_f:
            for i, line in enumerate(batch_f):
                meta = json.loads(line)

                if meta['title'] in unique_scidocs_titles:
                    batch.append((
                        meta['paper_id'], meta['title']
                    ))
        return batch

    # Run threads
    with Pool(workers) as pool:
        pool_outputs = list(tqdm(pool.imap_unordered(worker_extract_matching_titles, batch_fns), total=len(batch_fns)))

    # Merge thread outputs
    matching_titles = [i for o in pool_outputs for i in o]

    logger.info(f'Metadata parsed. {len(matching_titles):,} matching_titles')

    # Build mapping from titles to ids
    title_to_s2orc_paper_ids = defaultdict(list)
    for paper_id, t in matching_titles:
        title_to_s2orc_paper_ids[t].append(paper_id)

    s2id_to_s2orc_paper_id = {}

    ambiguous_s2orc_paper_ids = defaultdict(list)

    for ds, ds_meta in scidocs_metadata.items():
        for s2_id, paper_meta in ds_meta.items():

            if s2_id in s2id_to_s2orc_paper_id:
                # skip
                continue

            if paper_meta['title'] in title_to_s2orc_paper_ids:
                s2orc_paper_ids = title_to_s2orc_paper_ids[paper_meta['title']]

                # Ignore ambiguous paper ids
                if len(s2orc_paper_ids) == 1:
                    s2id_to_s2orc_paper_id[s2_id] = s2orc_paper_ids[0]
                else:
                    ambiguous_s2orc_paper_ids[s2_id] += s2orc_paper_ids

    logger.warning(f'Ambiguous paper ids / titles: {len(ambiguous_s2orc_paper_ids):,}')

    logger.info(f'Mapping for {len(s2id_to_s2orc_paper_id):,} IDs')

    if output_fp:
        # save to disk
        with open(output_fp, 'w') as f:
            json.dump(s2id_to_s2orc_paper_id, f)
    else:
        # return data
        return s2id_to_s2orc_paper_id


def get_s2orc_scidocs_mappings(
        s2id_to_paper_input_paths: Union[str, List[str]],
        s2id_to_s2orc_paper_id_input_paths: Union[str, List[str]],
        output_path: Union[None, str]
):
    """
    Merge mappings from S2 IDs to S2ORC IDs

    python cli_s2orc.py get_s2orc_scidocs_mappings \
        ./data/specter/id2paper.json,./data/specter_train_source_papers/id2paper.json,./data/scidocs_s2orc/id2paper.json \
        ./data/scidocs_s2id_to_s2orc_paper_id.json \
        ./data/specter/s2id_to_s2orc_paper_id.json

    :param s2id_to_paper_input_paths: List of S2 API response files (comma separated, each .json is a dict
        with S2 ID => paper metadata, the metadata has a `corpusId` field with the S2ORC ID)
    :param s2id_to_s2orc_paper_id_input_paths: List of S2ID-S2ORC ID mappings as JSON (comma separated, each .json is a
        dict with S2ID => S2ORC ID)
    :param output_path: Output path S2ID-S2ORC ID mapping JSON
    :return:
    """
    if isinstance(s2id_to_paper_input_paths, str):
        s2id_to_paper_input_paths = s2id_to_paper_input_paths.split(',')

    if isinstance(s2id_to_s2orc_paper_id_input_paths, str):
        s2id_to_s2orc_paper_id_input_paths = s2id_to_s2orc_paper_id_input_paths.split(',')

    # Load S2 API responses from disk
    s2_id_to_paper_list = []

    for fp in s2id_to_paper_input_paths:
        logger.info(f'Loading from {fp}')

        with open(fp) as f:
            s2_id_to_paper = json.load(f)
        s2_id_to_paper_list.append(s2_id_to_paper)

    # S2 ID to S2ORC mapping
    s2id_to_s2orc_paper_id = {}

    for s2_id_to_paper in s2_id_to_paper_list:
        for s2id, paper in s2_id_to_paper.items():
            if s2id not in s2id_to_s2orc_paper_id:
                s2id_to_s2orc_paper_id[s2id] = str(paper['corpusId'])

    # Predefined ID mappings (e.g., from titles)
    for fp in s2id_to_s2orc_paper_id_input_paths:
        logger.info(f'Loading from {fp}')
        with open(fp) as f:
            titles_s2id_to_s2orc_paper_id = json.load(f)

        # titles (last since probably inaccurate)
        for s2id, s2orc_id in titles_s2id_to_s2orc_paper_id.items():
            if s2id not in s2id_to_s2orc_paper_id:
                s2id_to_s2orc_paper_id[s2id] = s2orc_id

    logger.info(f'Mappings for {len(s2id_to_s2orc_paper_id):,} S2 IDs')

    if output_path:
        # write to disk
        with open(output_path, 'w') as f:
            json.dump(s2id_to_s2orc_paper_id, f)
    else:
        return s2id_to_s2orc_paper_id


def worker_extract_metadata_id_mapping(worker_id, batch_fns, s2orc_metadata_dir):
    batch_metadata = []

    for batch_fn in tqdm(batch_fns, desc=f'Worker {worker_id}'):
        with open(os.path.join(s2orc_metadata_dir, batch_fn)) as batch_f:
            for i, line in enumerate(batch_f):
                meta = json.loads(line)

                batch_metadata.append((
                    meta['paper_id'],
                    batch_fn,
                    i
                ))

    return batch_metadata


def get_metadata_id_mapping(s2orc_metadata_dir, output_path, workers: int = 10):
    """
    Extract id/file metadata mapping for S2ORC

    python cli_s2orc.py get_metadata_id_mapping ${S2ORC_METADATA_DIR} ${S2ORC_METADATA_DIR}/s2orc_metadata_id_mapping.json --workers 10

    :param s2orc_metadata_dir:
    :param output_path:
    :param workers:
    :return:
    """
    if os.path.exists(output_path):
        logger.error(f'Output already exists: {output_path}')
        return

    # Meta data files
    batch_fns = [batch_fn for batch_fn in os.listdir(s2orc_metadata_dir) if batch_fn.endswith('.jsonl.gz')]
    logger.info(f'Files available: {len(batch_fns):,}')

    logger.info(f'Extracting metadata with workers: {workers}')

    # worker_id, batch_fns, needed_paper_ids, s2orc_metadata_dir
    worker_data = zip(
        list(range(workers)),  # worker ids
        split_into_n_chunks(batch_fns, workers),
        [s2orc_metadata_dir] * workers,
    )

    # Run threads
    with Pool(workers) as pool:
        pool_outputs = list(pool.starmap(worker_extract_metadata_id_mapping, worker_data))

    # Merge thread outputs
    metadata_id_mapping = defaultdict(list)
    for b in pool_outputs:
        for paper_id, batch_fn, line_idx in b:
            metadata_id_mapping[batch_fn].append([
                paper_id, line_idx
            ])

    logger.info(f'Writing {len(metadata_id_mapping):,} metadata mappings to {output_path}')

    with open(output_path, 'w') as f:
        json.dump(metadata_id_mapping, f)


def get_s2orc_paper_ids_from_mapping(mapping_path: str, output_path: str, override: bool = False):
    """
    Get S2ORC paper IDs from mapping file

    Examples:

    python cli_s2orc.py get_s2orc_paper_ids_from_mapping \
        --mapping_path ${BASE_DIR}/data/scidocs_s2orc/s2id_to_s2orc_paper_id.latest.json \
        --output_path ${BASE_DIR}/data/scidocs_s2orc/s2orc_paper_ids.json

    python cli_s2orc.py get_s2orc_paper_ids_from_mapping \
        --mapping_path ${SPECTER_DIR}/s2id_to_s2orc_paper_id.latest.json \
        --output_path ${SPECTER_DIR}/s2orc_paper_ids.json

    :param override: Override existing output
    :param mapping_path: Path to JSON mapping file (S2 ID => S2ORC ID)
    :param output_path: Path to JSON list with S2ORC IDs
    :return:
    """

    if not override and os.path.exists(output_path):
        raise FileExistsError('Output exist already, --override not set.')

    with open(mapping_path) as f:
        mapping = json.load(f)

    logger.info(f'Mappings loaded: {len(mapping):,} from {mapping_path}')

    with open(output_path, 'w') as f:
        json.dump(list(mapping.values()), f)

    logger.info(f'Saved to {output_path}')


def worker_get_pdf_hashes(batch_fps):
    """
    Worker method for `get_pdf_hashes`

    :param batch_fps:
    :return:
    """
    logger.info(f'Starting worker. Extracting from {len(batch_fps):,} files')

    batch = []
    for batch_fp in batch_fps:
        with open(batch_fp) as batch_f:
            for i, line in enumerate(batch_f):
                parsed = json.loads(line)

                batch.append((
                    parsed['paper_id'],
                    parsed['_pdf_hash']
                ))

    return batch


def get_pdf_hashes(s2orc_pdf_parses_dir, paper_ids_output_path: str, pdf_hashes_output_path: str, workers: int = 0):
    """
    Extract PDF hashes from S2ORC. The hashes are identical with the S2 API IDs.
    See https://github.com/allenai/scidocs/issues/18#issuecomment-796865744

    Example:

    export S2ORC_PDF_PARSES_DIR=${DATASETS_DIR}/s2orc/20200705v1/full/pdf_parses
    export S2ORC_PDF_HASH_TO_ID=${DATASETS_DIR}/s2orc/20200705v1/full/pdf_hash_to_paper_id.json

    python cli_s2orc.py get_pdf_hashes ${S2ORC_PDF_PARSES_DIR} \
        ${DATASETS_DIR}/s2orc/20200705v1/full/paper_id_to_pdf_hash.json ${S2ORC_PDF_HASH_TO_ID} --workers 30

    :param s2orc_pdf_parses_dir: Path to PDF parses directory
    :param paper_ids_output_path: Mapping JSON with S2ORC ID => PDF hash
    :param pdf_hashes_output_path: Mapping JSON with PDF hash => S2ORC ID
    :param workers: Number of threads
    :return:
    """
    # all pdf parsed files names
    batch_parsed_fps = [os.path.join(s2orc_pdf_parses_dir, batch_fn) for batch_fn in os.listdir(s2orc_pdf_parses_dir) if
                        batch_fn.endswith('.jsonl.gz')]

    # Prepare worker data
    worker_data = zip(
        split_into_n_chunks(batch_parsed_fps, workers),
    )

    # Run threads
    with Pool(workers) as pool:
        pool_outputs = list(pool.starmap(worker_get_pdf_hashes, worker_data))

    # Save as JSON dict
    paper_id_to_pdf_hash = {paper_id: pdf_hash for batch in pool_outputs for paper_id, pdf_hash in batch}

    logger.info(f'Writing pdf hashes of {len(paper_id_to_pdf_hash):,} papers to {paper_ids_output_path}')

    with open(paper_ids_output_path, 'w') as f:
        json.dump(paper_id_to_pdf_hash, f)

    pdf_hash_to_paper_id = {pdf_hash: paper_id for paper_id, pdf_hash in paper_id_to_pdf_hash.items()}
    logger.info(f'Writing paper ids of {len(pdf_hash_to_paper_id):,} pdf hashes to {pdf_hashes_output_path}')

    with open(pdf_hashes_output_path, 'w') as f:
        json.dump(pdf_hash_to_paper_id, f)


def get_corpus(s2orc_paper_ids,
               s2orc_metadata_dir,
               scidocs_dir,
               specter_triples_path,
               s2id_to_s2orc_input_path,
               sample_n_nodes: int,
               citations_output_dir: str,
               paper_ids_output_dir: str,
               workers: int = 10,
               override: bool = False,
               ):
    """
    Build training corpus

    S2ORC -> 1M nodes
    - without SciDocs papers (citing or cited)
    - with SPECTER query papers (citing)
    - randomly sample 1M paper IDs
    - extract citations for the 1M papers

    Example:

    rm -r ${BASE_DIR}/data/biggraph/s2orc_without_scidocs_1m/*
    rm -r ${BASE_DIR}/data/v2_sci/s2orc_without_scidocs_1m/*

    python cli_s2orc.py get_corpus --s2orc_paper_ids ${BASE_DIR}/data/biggraph/s2orc_full/entity_names_paper_id_0.json \
        --s2orc_metadata_dir ${S2ORC_METADATA_DIR} \
        --scidocs_dir ${SCIDOCS_DIR} \
        --specter_triples_path ${SPECTER_DIR}/train_triples.csv \
        --s2id_to_s2orc_input_path ${ID_MAPPINGS}  \
        --citations_output_dir ${BASE_DIR}/data/biggraph/s2orc_without_scidocs_1m \
        --paper_ids_output_dir ${BASE_DIR}/data/v2_sci/s2orc_without_scidocs_1m \
        --workers ${WORKERS} \
        --sample_n_nodes 1000000

    :param s2orc_paper_ids:
    :param override:
    :param paper_ids_output_dir:
    :param citations_output_dir:
    :param workers:
    :param s2orc_metadata_dir:
    :param scidocs_dir:
    :param specter_triples_path:
    :param s2id_to_s2orc_input_path:
    :param sample_n_nodes:
    :return:
    """

    queries_fp = os.path.join(paper_ids_output_dir, 'query_s2orc_paper_ids.json')
    train_fp = os.path.join(paper_ids_output_dir, 's2orc_paper_ids.json')

    if not override and (
            os.path.exists(queries_fp) or os.path.exists(train_fp) or len(os.listdir(citations_output_dir)) > 0
    ):
        logger.error(f'Output of paper ids exist already or citation directory is not empty! Fix with --override')
        return

    # S2ORC paper ids (available in citation graph)
    if isinstance(s2orc_paper_ids, str):
        # load from disk
        with open(s2orc_paper_ids) as f:
            s2orc_paper_ids = json.load(f)

    s2orc_paper_ids_set = set(s2orc_paper_ids)

    logger.info(f'S2ORC paper IDs (in full corpus): {len(s2orc_paper_ids):,}')

    # S2-S2ORC Mappings
    s2id_to_s2orc_paper_id = read_json_mapping_files(s2id_to_s2orc_input_path)

    # SciDocs Metadata
    scidocs_metadata = get_scidocs_metadata(scidocs_dir)
    scidocs_s2_ids = {s2id for ds, ds_metadata in scidocs_metadata.items() for s2id in ds_metadata.keys()}
    logger.info(f'Scidocs - Unique S2 IDs: {len(scidocs_s2_ids):,}')

    # Map SciDocs IDs to S2ORC IDs
    scidocs_s2orc_paper_ids = {s2id_to_s2orc_paper_id[s2id] for s2id in scidocs_s2_ids if
                               s2id in s2id_to_s2orc_paper_id}

    logger.info(
        f'Scidocs - Successful mapped to S2ORC: {len(scidocs_s2orc_paper_ids):,} (missing: {len(scidocs_s2_ids - set(s2id_to_s2orc_paper_id.keys())):,})')

    # SPECTER train triples from disk (see `extract_triples`)
    with open(specter_triples_path) as f:
        specter_train_triples = [line.strip().split(',') for i, line in enumerate(f) if i > 0]

    logger.info(f'SPECTER - Loaded {len(specter_train_triples):,} triples from {specter_triples_path}')

    # SPECTER query
    specter_query_s2ids = {q for q, p, n in specter_train_triples}
    logger.info(f'SPECTER - Unique query S2 IDs: {len(specter_query_s2ids):,}')

    # Exclude SPECTER queries that in SciDocs
    specter_query_s2ids = specter_query_s2ids - scidocs_s2_ids
    logger.info(f'SPECTER - Queries excluding SciDocs papers: {len(specter_query_s2ids):,}')

    # SPECTER S2IDs to S2ORC IDs
    specter_query_s2orc_paper_ids = {s2id_to_s2orc_paper_id[s2id] for s2id in specter_query_s2ids
                                     if s2id in s2id_to_s2orc_paper_id}  # Map to S2ORC IDs

    logger.info(
        f'SPECTER - Queries with S2ORC IDs: {len(specter_query_s2ids & set(s2id_to_s2orc_paper_id.keys())):,} (missing: {len(specter_query_s2ids - set(s2id_to_s2orc_paper_id.keys())):,})')

    logger.info(
        f'SPECTER - Unique query S2ORC IDs: {len(specter_query_s2orc_paper_ids):,} (excluding duplicated S2ORC IDs)')

    # We use all SPECTER queries that are not in SciDocs (filtered before) but in S2ORC
    query_s2orc_paper_ids = specter_query_s2orc_paper_ids & s2orc_paper_ids_set

    logger.info(f'Query papers in SPECTER & S2ORC: {len(query_s2orc_paper_ids):,}')

    # Sampling
    logger.info(f'Nodes to be sampled: {sample_n_nodes:,}')

    missing_n_nodes = sample_n_nodes - len(query_s2orc_paper_ids)

    logger.info(f'Missing n nodes: {missing_n_nodes:,} (after queries)')

    # Sample from all S2ORC papers that are not already queries and not in SciDocs
    candidates = s2orc_paper_ids_set - query_s2orc_paper_ids - scidocs_s2orc_paper_ids

    logger.info(f'Candidates: {len(candidates):,} (S2ORC without queries and without SciDocs)')

    sampled_paper_ids = set(random.sample(candidates, missing_n_nodes))

    logger.info(f'Sampled paper IDs: {len(sampled_paper_ids):,}')

    train_s2orc_paper_ids = sampled_paper_ids.union(query_s2orc_paper_ids)

    logger.info(f'Train paper IDs: {len(train_s2orc_paper_ids):,} (sampled + queries)')

    # Output
    # citations as "citations.tsv"
    # queries as "query_s2orc_paper_ids.json"
    # train corpus as "s2orc_paper_ids.json"

    with open(queries_fp, 'w') as f:
        json.dump(list(query_s2orc_paper_ids), f)

    with open(train_fp, 'w') as f:
        json.dump(list(train_s2orc_paper_ids), f)

    get_citations(s2orc_metadata_dir=s2orc_metadata_dir,
                  output_dir=citations_output_dir,
                  workers=workers,
                  included_paper_ids=train_s2orc_paper_ids,
                  excluded_paper_ids=scidocs_s2orc_paper_ids,
                  description='Directed citation graph from S2ORC with 1M nodes without SciDocs with SPECTER queries')


def get_specter_corpus(
        s2orc_paper_ids,
        s2orc_metadata_dir,
        scidocs_dir,
        specter_triples_path,
        s2id_to_s2orc_input_path,
        sample_n_nodes: Union[str, int],
        paper_ids_output_dir: str,
        citations_output_dir: Optional[str] = None,
        workers: int = 10,
        override: bool = False,
        skip_citations: bool = False,
        seed: int = 0,
        ):
    """
    Reproduce SPECTER training corpus with S2ORC papers

    - Map SPECTER paper IDs to S2ORC paper IDs
    - Extract citations from S2ORC with SPECTER papers but without SciDocs papers
    - Generate exact SPECTER queries and SPECTER queries + random samples

    python cli_s2orc.py get_specter_corpus --s2orc_paper_ids ${BASE_DIR}/data/biggraph/s2orc_full/entity_names_paper_id_0.json \
        --s2orc_metadata_dir ${S2ORC_METADATA_DIR} \
        --scidocs_dir ${SCIDOCS_DIR} \
        --specter_triples_path ${SPECTER_DIR}/train_triples.csv \
        --s2id_to_s2orc_input_path ${ID_MAPPINGS}  \
        --citations_output_dir ${BASE_DIR}/data/biggraph/s2orc_with_specter_without_scidocs \
        --paper_ids_output_dir ${BASE_DIR}/data/v2_sci/s2orc_with_specter_without_scidocs_1m \
        --workers ${WORKERS} \
        --sample_n_nodes 1000000

    :param skip_citations:
    :param seed:
    :param s2orc_paper_ids:
    :param s2orc_metadata_dir:
    :param scidocs_dir:
    :param specter_triples_path:
    :param s2id_to_s2orc_input_path:
    :param sample_n_nodes:
    :param citations_output_dir:
    :param paper_ids_output_dir:
    :param workers:
    :param override:
    :return:
    """
    set_seed(seed)

    specter_queries_fp = os.path.join(paper_ids_output_dir, 'query_s2orc_paper_ids.specter.json')
    specter_random_queries_fp = os.path.join(paper_ids_output_dir, 'query_s2orc_paper_ids.specter_and_random.json')
    train_fp = os.path.join(paper_ids_output_dir, 's2orc_paper_ids.json')

    if not override and (
            os.path.exists(specter_queries_fp) or os.path.exists(train_fp) or (not skip_citations and len(os.listdir(citations_output_dir)) > 0)
    ):
        logger.error(f'Output of paper ids exist already or citation directory is not empty! Fix with --override')
        return

    # S2ORC paper ids (available in citation graph)
    if isinstance(s2orc_paper_ids, str):
        # load from disk
        with open(s2orc_paper_ids) as f:
            s2orc_paper_ids = json.load(f)

    s2orc_paper_ids_set = set(s2orc_paper_ids)

    logger.info(f'S2ORC paper IDs (in full corpus): {len(s2orc_paper_ids):,}')

    # S2-S2ORC Mappings
    s2id_to_s2orc_paper_id = read_json_mapping_files(s2id_to_s2orc_input_path)

    # SciDocs Metadata
    scidocs_metadata = get_scidocs_metadata(scidocs_dir)
    scidocs_s2_ids = {s2id for ds, ds_metadata in scidocs_metadata.items() for s2id in ds_metadata.keys()}
    logger.info(f'Scidocs - Unique S2 IDs: {len(scidocs_s2_ids):,}')

    # Map SciDocs IDs to S2ORC IDs
    scidocs_s2orc_paper_ids = {s2id_to_s2orc_paper_id[s2id] for s2id in scidocs_s2_ids if
                               s2id in s2id_to_s2orc_paper_id}

    logger.info(
        f'Scidocs - Successful mapped to S2ORC: {len(scidocs_s2orc_paper_ids):,} (missing: {len(scidocs_s2_ids - set(s2id_to_s2orc_paper_id.keys())):,})')

    # SPECTER train triples from disk (see `extract_triples`)
    with open(specter_triples_path) as f:
        specter_train_triples = [line.strip().split(',') for i, line in enumerate(f) if i > 0]

    logger.info(f'SPECTER - Loaded {len(specter_train_triples):,} triples from {specter_triples_path}')

    # SPECTER paper ids
    specter_s2ids = {i for t in specter_train_triples for i in t}

    logger.info(f'SPECTER - Unique S2 IDs: {len(specter_s2ids):,}')

    # SPECTER S2IDs to S2ORC IDs
    specter_s2orc_paper_ids = {s2id_to_s2orc_paper_id[s2id] for s2id in specter_s2ids
                                     if s2id in s2id_to_s2orc_paper_id}  # Map to S2ORC IDs
    logger.info(
        f'SPECTER - Papers with S2ORC IDs: {len(specter_s2ids & set(s2id_to_s2orc_paper_id.keys())):,} (missing: {len(specter_s2ids - set(s2id_to_s2orc_paper_id.keys())):,})')

    logger.info(
        f'SPECTER - Unique S2ORC IDs: {len(specter_s2orc_paper_ids):,} (excluding duplicated S2ORC IDs)')

    specter_s2orc_paper_ids_in_graph = specter_s2orc_paper_ids & s2orc_paper_ids_set

    logger.info(
        f'SPECTER - Papers in graph: {len(specter_s2orc_paper_ids_in_graph):,} (part of S2ORC corpus)')

    # Exact SPECTER query papers
    specter_query_s2ids = {q for q, p, n in specter_train_triples}
    specter_query_s2orc_paper_ids = {s2id_to_s2orc_paper_id[s2id] for s2id in specter_query_s2ids
                                     if s2id in s2id_to_s2orc_paper_id}  # Map to S2ORC IDs
    specter_query_s2orc_paper_ids_in_graph = specter_query_s2orc_paper_ids & s2orc_paper_ids_set

    logger.info(f'SPECTER queries: {len(specter_query_s2orc_paper_ids_in_graph):,} / {len(specter_query_s2ids):,} (mapped to S2ORC and in graph')

    with open(specter_queries_fp, 'w') as f:
        json.dump(list(specter_query_s2orc_paper_ids_in_graph), f)
        logger.info(f'Saved {len(specter_query_s2orc_paper_ids_in_graph):,} at {specter_queries_fp}')

    # SPECTER queries + random samples
    expected_query_count = len(specter_query_s2ids)
    missing_query_count = expected_query_count - len(specter_query_s2orc_paper_ids_in_graph)

    query_candidates = s2orc_paper_ids_set - scidocs_s2orc_paper_ids - specter_s2orc_paper_ids_in_graph

    additional_queries = random.sample(query_candidates, missing_query_count)
    specter_random_queries = list(specter_query_s2orc_paper_ids_in_graph) + additional_queries

    with open(specter_random_queries_fp, 'w') as f:
        json.dump(specter_random_queries, f)
        logger.info(f'Saved {len(specter_random_queries):,} at {specter_random_queries_fp}')

    # Exclude all citations from SciDocs but not the ones from SPECTER
    exclude_papers = scidocs_s2orc_paper_ids - specter_s2orc_paper_ids_in_graph

    # All papers in corpus
    if sample_n_nodes == 'specter':
        # Train corpus will have the same size as SPECTER
        missing_train_count = len(specter_s2ids) - len(specter_s2orc_paper_ids_in_graph)
        train_candidates = s2orc_paper_ids_set - scidocs_s2orc_paper_ids - specter_s2orc_paper_ids_in_graph
        additional_train = random.sample(train_candidates, missing_train_count)

        train_papers = list(specter_s2orc_paper_ids_in_graph) + additional_train

    elif sample_n_nodes > 0:
        # Sample exact number of papers as train corpus
        missing_train_count = sample_n_nodes - len(specter_s2orc_paper_ids_in_graph)

        if missing_query_count < 0:
            raise ValueError()

        train_candidates = s2orc_paper_ids_set - scidocs_s2orc_paper_ids - specter_s2orc_paper_ids_in_graph
        additional_train = random.sample(train_candidates, missing_train_count)

        train_papers = list(specter_s2orc_paper_ids_in_graph) + additional_train
    else:
        # Use all papers
        train_papers = list(s2orc_paper_ids_set - exclude_papers)

    with open(train_fp, 'w') as f:
        json.dump(train_papers, f)
        logger.info(f'Saved {len(train_papers):,} at {train_fp}')

    if skip_citations:
        logger.info('Skip citations')
    else:
        get_citations(s2orc_metadata_dir=s2orc_metadata_dir,
                      output_dir=citations_output_dir,
                      workers=workers,
                      excluded_paper_ids=exclude_papers,
                      description='Directed citation graph from S2ORC with SPECTER but without SciDocs')


if __name__ == '__main__':
    fire.Fire()
