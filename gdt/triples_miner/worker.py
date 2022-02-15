import logging
from typing import List, Dict, Union

from tqdm.auto import tqdm
from transformers import set_seed

from gdt.triples_miner import TriplesMinerArguments, AnnBackend
from gdt.triples_miner.sampling import FaissSampler, AnnoySampler

logger = logging.getLogger(__name__)


def worker_generate_triples(
        worker_id: int,
        items: List,
        paper_id_to_idx: Dict,
        idx_to_paper_id: Dict,
        ann_vector_size: int,
        seed: int,
        args: Union[TriplesMinerArguments, Dict]) -> List:
    """
    Worker function that is executed in parallel.

    :param seed: Random seed
    :param worker_id: Worker ID for logging
    :param items: Query paper IDs
    :param paper_id_to_idx: Mapping for paper embedding index to ID
    :param idx_to_paper_id: Reverse mapping
    :param ann_vector_size: Size of embeddings used for ANN
    :param args: Additional args
    :return: List of triples
    """
    triples = []
    set_seed(seed)

    if isinstance(args, dict):
        args = TriplesMinerArguments(**args)

    if args.ann_backend == AnnBackend.ANNOY:
        sampler = AnnoySampler(items, paper_id_to_idx, idx_to_paper_id, args, ann_vector_size, worker_id)

    elif args.ann_backend == AnnBackend.FAISS:
        sampler = FaissSampler(items, paper_id_to_idx, idx_to_paper_id, args, ann_vector_size, worker_id)

    else:
        raise ValueError(f'Invalid ANN Backend: {args.ann_backend}')

    # for item in tqdm(items, desc=f'Worker #{worker_id}'):
    #     query_paper_id = item[0]

    # Generate triples based on pre-computed NNs
    for query_idx, query_paper_id in tqdm(enumerate(items), total=len(items), desc=f'Triple miner worker #{worker_id}'):

        triples += sampler.get_triples_from_query(query_idx, query_paper_id, seed)

    if len(triples) != args.triples_per_query * len(items):
        raise ValueError(f'Invalid triples count: {len(triples):,} (expected: {args.triples_per_query * len(items)})')

    # set seed back to global value
    set_seed(seed)

    sampler.shutdown()

    return triples
