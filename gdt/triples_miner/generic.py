import logging
import os
from multiprocessing import Pool
from typing import Dict, List, Union

import numpy as np
from sklearn.preprocessing import normalize
from smart_open import open
from transformers import set_seed

from gdt.triples_miner import TriplesMinerArguments, AnnBackend
from gdt.triples_miner.worker import worker_generate_triples
from gdt.utils import split_into_n_chunks

logger = logging.getLogger(__name__)


def get_generic_triples(
        document_id_to_idx: Dict[str, int],
        idx_to_document_id: Dict[int, str],
        query_document_ids: List[str],
        # graph_embeddings_path: str,
        # graph_paper_ids:
        graph_embeddings: Union[List, np.ndarray],
        output_path: str,
        # triples_per_query: int = 5,
        # hard_negatives_count: int = 2,
        # easy_negatives_count: int = 3,
        # ann_trees: int = 1000,
        # ann_top_k: int = 500,
        # ann_index_path: str = None,
        # ann_metric: str = 'euclidean',
        triples_miner_args: TriplesMinerArguments,
        workers: int = 10,
        output_csv_header: str = 'query_paper_id,positive_id,negative_id',
        ):

    # assert hard_negatives_count + easy_negatives_count == triples_per_query

    logger.info(f'Query papers: {len(query_document_ids):,}')
    logger.info(f'Triples per query: {triples_miner_args.triples_per_query}')
    logger.info(f'Triple miner args: {triples_miner_args}')

    set_seed(triples_miner_args.seed)

    if triples_miner_args.ann_workers is not None and triples_miner_args.ann_workers > 0:
        workers = triples_miner_args.ann_workers

    if triples_miner_args.ann_index_path is None:
        triples_miner_args.ann_index_path = output_path + '.' + triples_miner_args.ann_backend

    graph_embedding_size = len(graph_embeddings[0])

    if os.path.exists(triples_miner_args.ann_index_path):
        # Reuse existing ANN index
        logger.info(f'Reusing existing ANN index from {triples_miner_args.ann_index_path}')
    else:
        if triples_miner_args.ann_backend == AnnBackend.ANNOY:
            # New ANN index
            ann_index = AnnoyIndex(graph_embedding_size, triples_miner_args.ann_metric)
            ann_index.set_seed(triples_miner_args.seed)

            # Length of item vector that will be indexed
            # Cosine distance is equivalent to Euclidean distance of normalized vectors = sqrt(2-2*cos(u, v))

            # normalize vectors for cosine similarity
            if triples_miner_args.ann_normalize_embeddings:
                logger.info('Normalizing graph embeddings ...')
                # graph_embeddings = normalize_in_parallel(graph_embeddings, workers)  # does not work in parallel
                graph_embeddings = normalize(graph_embeddings)

            logger.info('Adding to ANN index')
            for idx, embed in enumerate(graph_embeddings):
                ann_index.add_item(idx, embed)

            logger.info(f'Building ANN index with trees={triples_miner_args.ann_trees} and workers={workers}')

            ann_index.build(triples_miner_args.ann_trees, n_jobs=workers)

            logger.info(f'Saving ANN index to {triples_miner_args.ann_index_path}')
            ann_index.save(triples_miner_args.ann_index_path)

        elif triples_miner_args.ann_backend == AnnBackend.FAISS:
            from cli_graph import build_faiss

            logger.info(f'Use FAISS as ANN backend')

            build_faiss(
                graph_embeddings_or_path=graph_embeddings,
                index_path=triples_miner_args.ann_index_path,
                string_factory=triples_miner_args.faiss_string_factory,
                metric_type=0,  # default
                do_normalize=triples_miner_args.ann_normalize_embeddings,
                workers=triples_miner_args.ann_workers,
                seed=triples_miner_args.seed,
                # Paper IDs do not need to be specified since parent method defines train embeddings
                # See `get_specter_triples`
                paper_ids=None,
                include_paper_ids=None,
                device='all',
                batch_size=triples_miner_args.ann_batch_size,
                train_size=triples_miner_args.ann_train_size,
            )
        else:
            raise NotImplementedError(f'Cannot build ANN index with backend = {triples_miner_args.ann_backend} '
                                      f'(use extra CLI script instead)')
    # Easy negatives = random papers
    # easy_negatives = []
    #
    # for i in range(easy_negatives_count):
    #     random_papers = list(query_document_ids)
    #     random.shuffle(random_papers)
    #
    #     easy_negatives.append(random_papers)

    if workers == 1:
        logger.info(f'Triple generation with single thread')

        triples = worker_generate_triples(0, query_document_ids, document_id_to_idx, idx_to_document_id,
                                          graph_embedding_size, triples_miner_args.seed, triples_miner_args.__dict__)

    else:
        logger.info(f'Starting {workers} workers for triple generation')

        worker_data = zip(
            list(range(workers)),  # worker ids
            # split_into_n_chunks(list(zip(query_document_ids, *easy_negatives)), workers),  # items
            split_into_n_chunks(list(query_document_ids), workers),  # items

            # static arguments (same for all workers)
            [document_id_to_idx] * workers,
            [idx_to_document_id] * workers,
            [graph_embedding_size] * workers,
            [triples_miner_args.seed] * workers,
            [triples_miner_args.__dict__] * workers,
            # [ann_index_path] * workers,
            # [ann_top_k] * workers,
            # [ann_metric] * workers,
            # [triples_per_query] * workers,
            # [easy_negatives_count] * workers,
            # [hard_negatives_count] * workers,
        )

        # Run threads
        with Pool(workers) as pool:
            # pool.map(functools.partial(print_data, first=False), test_data)
            # pool_outputs = list(pool.starmap(functools.partial(worker_generate_triples,
            #                                                    paper_id_to_idx=document_id_to_idx,
            #                                                    idx_to_paper_id=idx_to_document_id,
            #                                                    ann_vector_size=graph_embeddings,
            #                                                    args=triples_miner_args), zip(
            #     list(range(workers)),  # worker ids
            #     split_into_n_chunks(list(query_document_ids), workers)
            # )))
            pool_outputs = list(pool.starmap(worker_generate_triples, worker_data))

            # takes some time to start
            # pool_outputs = list(tqdm(pool.imap_unordered(worker.generate_triples, list(zip(train_query_s2orc_paper_ids, *easy_negatives))),
            #                          total=len(train_query_s2orc_paper_ids)))

            # Merge thread outputs
            triples = [i for batch in pool_outputs for i in batch]

    logger.info(f'Triples mined: {len(triples):,}')

    if output_path:
        # write to disk
        logger.info(f'Writing {len(triples):,} triples to {output_path}')

        with open(os.path.join(output_path), 'w') as f:
            f.write(output_csv_header + '\n')
            for query_paper_id, pos_id, neg_id in triples:
                f.write(f'{query_paper_id},{pos_id},{neg_id}\n')
    else:
        return triples