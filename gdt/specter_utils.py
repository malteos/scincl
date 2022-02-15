import logging
from typing import Tuple, Set, Dict

logger = logging.getLogger(__name__)


def import_triples(specter_triples_path: str, s2id_to_s2orc_paper_id: Dict[str, str], only_queries: bool = False) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Import SPECTER triples from CSV and map to S2ORC IDs

    :param s2id_to_s2orc_paper_id: Mapping dict from S2ID to S2ORC IDs
    :param specter_triples_path: Path to CSV
    :param only_queries: Return only IDs of query papers (if not all paper IDs are returned)
    :return:
    """
    if only_queries:
        log_prefix = 'SPECTER queries'
    else:
        log_prefix = 'SPECTER all'

    # SPECTER train triples from disk (see `extract_triples`)
    with open(specter_triples_path) as f:
        specter_train_triples = [line.strip().split(',') for i, line in enumerate(f) if i > 0]

    logger.info(f'{log_prefix} - Loaded {len(specter_train_triples):,} triples from {specter_triples_path}')

    # Paper corpus (queries, positives, negatives)
    if only_queries:
        specter_s2ids = {q for q, p, n in specter_train_triples }  # query ids
    else:
        specter_s2ids = {i for t in specter_train_triples for i in t}  # all ids

    logger.info(f'{log_prefix} - Unique S2 IDs: {len(specter_s2ids):,}')

    # SPECTER S2IDs to S2ORC IDs
    specter_s2orc_paper_ids = {s2id_to_s2orc_paper_id[s2id] for s2id in specter_s2ids
                                     if s2id in s2id_to_s2orc_paper_id}  # Map to S2ORC IDs

    missing_s2ids = specter_s2ids - set(s2id_to_s2orc_paper_id.keys())

    logger.info(f'{log_prefix} - Unique S2 IDs with S2ORC IDs: {len(specter_s2ids & set(s2id_to_s2orc_paper_id.keys())):,} (missing: {len(missing_s2ids):,})')

    logger.info(f'{log_prefix} - Unique S2ORC IDs: {len(specter_s2orc_paper_ids):,} (excluding duplicated S2ORC IDs)')

    return specter_s2ids, specter_s2orc_paper_ids,  missing_s2ids
