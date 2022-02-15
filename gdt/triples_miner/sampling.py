import logging
import os
import pickle
import random
from typing import List, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from transformers import set_seed

from gdt.triples_miner import TriplesMinerArguments, ExampleType, ExampleDifficulty, SamplingStrategy

logger = logging.getLogger(__name__)


class BaseSampler(object):
    args: TriplesMinerArguments
    paper_id_to_centroid: Optional[Dict[str, int]] = None
    centroid_to_paper_ids: Optional[Dict[int, List[str]]] = None
    centroid_to_sorted_centroids: Optional[np.array]
    query_idx_to_range_results: Optional[Dict[int, List[Tuple[int, float]]]] = None  # paper idx => list of tuples (paper_idx, score); descending by score
    query_idx_to_random_sorted_ids = None
    query_nearest_idxs: Union[List, np.array]
    random_papers: List[str]
    random_count: int
    items: List[str]
    idx_to_paper_id: Union[List[str], Dict[int, List[str]]]
    paper_id_to_idx: Dict[str, int]
    range_fallback_counter = 0
    range_equal_scores_counter = 0

    def init_kmeans(self):
        # K Means
        if self.args.has_k_means_strategy():
            # logger.info(f'Loading K-Means data from {args.k_means_dir}')

            paper_id_to_centroid_path = os.path.join(self.args.k_means_dir, 'paper_id_to_centroid.pickle')
            if os.path.exists(paper_id_to_centroid_path):
                with open(paper_id_to_centroid_path, 'rb') as f:
                    self.paper_id_to_centroid = pickle.load(f)
            else:
                raise FileNotFoundError(f'Not exists paper_id_to_centroid: {paper_id_to_centroid_path}')

            centroid_to_paper_ids_path = os.path.join(self.args.k_means_dir, 'centroid_to_paper_ids.pickle')
            if os.path.exists(centroid_to_paper_ids_path):
                with open(centroid_to_paper_ids_path, 'rb') as f:
                    self.centroid_to_paper_ids = pickle.load(f)
            else:
                raise FileNotFoundError(f'Not exists centroid_to_paper_ids: {centroid_to_paper_ids_path}')

            centroid_to_sorted_centroids_path = os.path.join(self.args.k_means_dir, 'sorted_centroids.npy')
            if os.path.exists(centroid_to_sorted_centroids_path):
                with open(centroid_to_sorted_centroids_path, 'rb') as f:
                    self.centroid_to_sorted_centroids = np.load(f)
            else:
                raise FileNotFoundError(f'Not exists centroid_to_sorted_centroids: {centroid_to_sorted_centroids_path}')

        else:
            self.paper_id_to_centroid = None
            self.centroid_to_sorted_centroids = None
            self.centroid_to_paper_ids = None

    def init_randoms(self):
        # Prepare random paper ids (more efficient than random sampling for every query)

        self.random_count = self.args.get_random_count()
        self.random_count += 2 * self.args.get_random_without_knn_count()  # double random count since knn examples are excluded
        self.random_count += 2 * self.args.get_random_without_range_count()  # double random count since knn examples are excluded

        if self.args.random_sample_factor > 0:
            logger.info(f'User-defined random sample factor: {self.args.random_sample_factor}')
            sample_factor = self.args.random_sample_factor
        else:
            logger.info(f'Random sample factor based on random strategy count: {self.random_count}')
            sample_factor = self.random_count

        self.random_papers = list(self.items) * sample_factor

        random.shuffle(self.random_papers)

    def get_random_examples_without(self, available_random_ids, count, without_idxs=None):
        examples = []
        tries = 0
        max_tries = 100

        if without_idxs is not None:
            max_tries += len(without_idxs)  # Number of tries must depend on the number of results we exclude

        # Just take random samples (previously shuffled list - for efficiency)
        if len(available_random_ids) < count:
            raise ValueError(f'Not enough random IDs. Available: {len(available_random_ids):,}; Needed: {count:,}')

        while len(examples) < count:
            # take one example from available random ones (from beginning of list)
            candidate_id = available_random_ids.pop(0)

            if without_idxs is None:
                # No check what so ever, just add candidate as example
                examples.append(candidate_id)
            else:
                # Is candidate NOT part of knn/range?
                if self.paper_id_to_idx[candidate_id] not in without_idxs:
                    examples.append(candidate_id)
                else:
                    # put back to randoms (to end of list)
                    available_random_ids.append(candidate_id)

                tries += 1

                if tries > max_tries:
                    raise ValueError(f'Too many tries... stopping after {tries}')

        if len(examples) != count:
            raise ValueError(f'Invalid random examples: {len(examples)} (expected: {count})')

        return examples, available_random_ids

    def get_examples(self,
                     query_idx: int,
                     nearest_idxs: List[int],
                     available_random_ids: List[str],
                     example_type: ExampleType,
                     difficulty: ExampleDifficulty,
                     centroid: Optional[int] = None,
                     sorted_centroids: Optional[np.ndarray] = None,
                     #centroid_to_paper_ids: Optional[Dict[int, List[str]]] = None,
                     seed: int = 0,
                     range_results: Optional[List[Tuple[int, float]]] = None):
        """
        Generate `n` examples (papers, docs, ...) based on a given strategy.

        :param query_idx: Index of query paper
        :param range_results: Results of range search (sorted list of tuples with paper idx, score)
        :param paper_id_to_idx:
        :param seed: Random seed
        :param nearest_idxs: Pre-computed nearest neighbors as indices
        :param available_random_ids: Pop random items from this list
        :param idx_to_paper_id:
        :param args:
        :param example_type:
        :param difficulty:
        :param centroid:
        :param sorted_centroids:
        :param centroid_to_paper_ids:
        :return:
        """
        name = difficulty + '_' + example_type
        original_count = getattr(self.args, name + '_count')
        count = int(getattr(self.args, name + '_count'))
        strategy = getattr(self.args, name + '_strategy')
        k_min = getattr(self.args, name + '_k_min')
        k_max = getattr(self.args, name + '_k_max')
        repeat = bool(getattr(self.args, name + '_repeat'))
        examples = []

        # Set random seed
        set_seed(seed)

        if repeat:
            count = 1  # sample only a single example and repeat this one
            assert original_count > count

        if count > 0:
            if strategy == SamplingStrategy.KNN:
                # Select examples based on the k nearest neighbors
                examples = [self.get_paper_id_by_idx(idx) for idx in nearest_idxs[k_min:k_max]]

                if len(examples) > count:
                    # k range is greater than count -> sample
                    examples = random.sample(examples, count)
                elif len(examples) < count:
                    raise ValueError(f'Invalid KNN examples: {len(examples)} (expected: {count})')

            elif strategy == SamplingStrategy.RANDOM:
                examples, available_random_ids = self.get_random_examples_without(available_random_ids, count,
                                                                                  None)
                # # Just take random samples (previously shuffled list - for efficiency)
                # if len(available_random_ids) < count:
                #     raise ValueError(f'Not enough random IDs. Available: {len(available_random_ids):,}; Needed: {count:,}')
                #
                # examples = available_random_ids[:count]
                #
                # if len(examples) != count:
                #     raise ValueError(f'Invalid random examples: {len(examples)} (expected: {count})')
                #
                # available_random_ids = available_random_ids[count:]  # remove from randoms
            elif strategy == SamplingStrategy.RANDOM_WITHOUT_KNN:
                examples, available_random_ids = self.get_random_examples_without(available_random_ids, count, nearest_idxs)

            elif strategy == SamplingStrategy.KMEANS:
                # Select examples based on the centroids and their similarity from k means clustering
                # candidate_paper_ids = []

                if k_max - k_min == count:  # k range corresponds to number of examples
                    examples = []
                    # sample one example from each centroid (centroids sorted based on similarity to current centroid)
                    for candidate_centroid in sorted_centroids[k_min:k_max]:
                        examples.append(
                            random.sample(self.centroid_to_paper_ids[candidate_centroid], 1)
                        )
                else:
                    candidate_centroids = sorted_centroids[k_min:k_max].tolist()

                    # Sample n random centroids
                    centroids = random.sample(candidate_centroids, count)

                    # Sample one example from each centroid
                    examples = [random.sample(self.centroid_to_paper_ids[c], 1)[0] for c in centroids]

                    # # select candidates from centroids sorted based on similarity to current centroid
                    # for candidate_centroid in sorted_centroids[k_min:k_max]:
                    #     # take all paper ids from these centroids
                    #     candidate_paper_ids += centroid_to_paper_ids[candidate_centroid]
                    #
                    # if len(candidate_paper_ids) < count:
                    #     # repeat candidates if not enough papers
                    #     candidate_paper_ids = candidate_paper_ids * (count - len(candidate_paper_ids))
                    #
                    # # random sample "count" from all paper ids
                    # examples = random.sample(candidate_paper_ids, count)

            elif strategy == SamplingStrategy.RANGE:
                # Select examples based on distance/similarity to query
                #
                # Example:
                #
                # Given query document d_q and range threshold 0.5, a the range search will return n results:
                #  R=[(d_q, 1.0), (d_c1, 0.89), ..., (d_cn, 0.48)] where the similarity of the last candidate d_cn is < 0.5.
                #
                # As examples we select R[n-m:n] where m is the number of requested examples (`count`).
                # The distance/similarity of the selected examples is NOT fixed, only smaller than the threshold.

                if len(range_results) < count + 1 + self.args.min_range_results_count:
                    if self.args.range_fallback_to_random:
                        # Fallback strategy if not enough range results are available
                        examples, available_random_ids = self.get_random_examples_without(available_random_ids, count,
                                                                                          nearest_idxs)

                        self.range_fallback_counter += 1
                    else:
                        raise ValueError(f'Not enough range results returned: range_results = {len(range_results)}; '
                                         f'min_range_results_count = {self.args.min_range_results_count} + 1 + {count}'
                                         f'(enable `range_fallback_to_random` to get rid of this error)')

                else:
                    # check for different scores (if nprobe is too low, faiss will return results with the same similarity)
                    if range_results[-1][1] == range_results[-2][1]:  # compare last and second last
                        logger.warning(f'Range results with equal scores: {range_results[-count:]} '
                                       f'(to fix this increase faiss.index.nprobe)')

                        self.range_equal_scores_counter += 1

                    examples = [self.get_paper_id_by_idx(idx) for idx, score in range_results[-count:]]

            elif strategy == SamplingStrategy.RANDOM_WITHOUT_RANGE:

                range_idxs = [idx for idx, score in range_results]

                examples, available_random_ids = self.get_random_examples_without(available_random_ids, count,
                                                                                  range_idxs)

            elif strategy == SamplingStrategy.RANDOM_SORTED:
                # See FaissSampler.__init__
                examples = self.query_idx_to_random_sorted_ids[query_idx]
            else:
                raise ValueError(f'Invalid sampling strategy: {strategy}')

        if repeat:
            # Repeat examples
            assert len(examples) == 1  # should be exactly one example

            examples = examples * original_count  # repeat to match count

        return examples, available_random_ids

    def get_triples_from_query(self, query_idx, query_paper_id, seed):
        # for each query a different seed (for kmeans centroid sampling, otherwise k means has often the same examples)
        query_seed = seed + query_idx

        nearest_idxs = self.query_nearest_idxs[query_idx]
        range_results = self.query_idx_to_range_results[query_idx] if self.query_idx_to_range_results is not None else None

        # Centroid of query
        if self.args.has_k_means_strategy():
            centroid = self.paper_id_to_centroid[query_paper_id]
            sorted_centroids = self.centroid_to_sorted_centroids[centroid]
        else:
            centroid = None
            sorted_centroids = None

        if self.args.random_sample_factor > 0:
            available_random_ids = self.random_papers
        else:
            available_random_ids = self.random_papers[query_idx:query_idx + self.random_count]

        # assert len(available_random_ids) == random_count

        # Positives
        easy_positive_ids, available_random_ids = self.get_examples(
            query_idx,
            nearest_idxs, available_random_ids, ExampleType.POSITIVES, ExampleDifficulty.EASY,
            centroid, sorted_centroids, query_seed, range_results
        )
        medium_positive_ids, available_random_ids = self.get_examples(
            query_idx,
            nearest_idxs, available_random_ids, ExampleType.POSITIVES, ExampleDifficulty.MEDIUM,
            centroid, sorted_centroids, query_seed, range_results
        )
        hard_positive_ids, available_random_ids = self.get_examples(
            query_idx,
            nearest_idxs, available_random_ids, ExampleType.POSITIVES, ExampleDifficulty.HARD,
            centroid, sorted_centroids, query_seed, range_results
        )
        positive_ids = easy_positive_ids + medium_positive_ids + hard_positive_ids

        # Negatives
        easy_negative_ids, available_random_ids = self.get_examples(
            query_idx,
            nearest_idxs, available_random_ids, ExampleType.NEGATIVES, ExampleDifficulty.EASY,
            centroid, sorted_centroids, query_seed, range_results
        )
        medium_negative_ids, available_random_ids = self.get_examples(
            query_idx,
            nearest_idxs, available_random_ids, ExampleType.NEGATIVES, ExampleDifficulty.MEDIUM,
            centroid, sorted_centroids, query_seed, range_results
        )
        hard_negative_ids, _ = self.get_examples(
            query_idx,
            nearest_idxs, available_random_ids, ExampleType.NEGATIVES, ExampleDifficulty.HARD,
            centroid, sorted_centroids, query_seed, range_results
        )
        negative_ids = easy_negative_ids + medium_negative_ids + hard_negative_ids

        # Combine to triples
        query_triples = list(zip(
            [query_paper_id] * self.args.triples_per_query,
            positive_ids,
            negative_ids
        ))

        if len(query_triples) != self.args.triples_per_query:
                raise ValueError(f'Invalid triples count: {len(query_triples):,}; '
                                 f'pos = {len(positive_ids)}; '
                                 f'neg = {len(negative_ids)} (expected: {self.args.triples_per_query})')

        # save random queue again
        if self.args.random_sample_factor > 0:
            self.random_papers = available_random_ids
        else:
            pass

        return query_triples

    def get_paper_id_by_idx(self, idx):
        try:
            return self.idx_to_paper_id[idx]
        except KeyError:
            raise ValueError(f'Could not get `paper_id` for idx = {idx} '
                             f'(idx_to_paper_id = {len(self.idx_to_paper_id):,}; '
                             f'paper_id_to_idx = {len(self.paper_id_to_idx):,})')

    def shutdown(self):
        """
        This method is called after all triples have been generated.
        """
        logger.info(f'range_fallback_counter = {self.range_fallback_counter:,}')
        logger.info(f'range_equal_scores_counter = {self.range_equal_scores_counter:,}')


class AnnoySampler(BaseSampler):
    def __init__(self, items, paper_id_to_idx, idx_to_paper_id, args: TriplesMinerArguments, ann_vector_size, worker_id=0):
        from annoy import AnnoyIndex

        self.items = items
        self.paper_id_to_idx = paper_id_to_idx
        self.idx_to_paper_id = idx_to_paper_id
        self.args = args
        self.ann_vector_size = ann_vector_size
        self.worker_id = worker_id

        logger.info(
            f'Loading AnnoyIndex (worker_id={worker_id}; vector_size={self.ann_vector_size}; metric={self.args.ann_metric}; '
            f'preload={args.ann_preload})'
            f' from {args.ann_index_path}')

        # Annoy
        ann_index = AnnoyIndex(self.ann_vector_size, self.args.ann_metric)
        ann_index.load(self.args.ann_index_path, prefault=self.args.ann_preload)  # super fast, will just mmap the file

        # Search NN item by item
        self.query_nearest_idxs = []

        for query_idx, query_paper_id in tqdm(enumerate(items), total=len(items), desc=f'Annoy worker #{worker_id}'):
            # Nearest neighbors of query
            nearest_idxs = ann_index.get_nns_by_item(self.paper_id_to_idx[query_paper_id], self.args.get_k_max() + 1)[1:]  # exclude self

            self.query_nearest_idxs.append(nearest_idxs)

        if self.args.has_range_strategy():
            raise NotImplementedError('ANNOY backend does not support range search.')

        if self.args.has_random_sorted_strategy():
            raise NotImplementedError('Annoy backend does not support `random_sorted` strategy.')

        self.init_kmeans()
        self.init_randoms()


class FaissSampler(BaseSampler):
    def __init__(self, items, paper_id_to_idx, idx_to_paper_id, args: TriplesMinerArguments, ann_vector_size, worker_id = 0):
        self.items = items
        self.paper_id_to_idx = paper_id_to_idx
        self.idx_to_paper_id = idx_to_paper_id
        self.args = args
        self.ann_vector_size = ann_vector_size
        self.worker_id = worker_id

        # Faiss (search queries all at once)
        # NOTE: faiss has a limitation of either 1024 or 2048 nearest neighbors, depending on your version of cuda
        # if self.args.get_k_max() > 2048:
        #     raise ValueError(f'k_max is too large for FAISS')
        # ===> Ignore, because we do not use CUDA for search (only for building the index)

        import faiss

        logger.info(f'Loading FAISS index (path={self.args.ann_index_path}; worker_id={self.worker_id};)')

        # Load FAISS from disk
        index = faiss.read_index(args.ann_index_path)

        # From https://github.com/facebookresearch/faiss/wiki/Faster-search
        # nlist, the number of cells, and
        # nprobe, the number of cells (out of nlist) that are visited to perform a search
        # --
        # Avoid -1 results with large nprobe
        # https://github.com/facebookresearch/faiss/wiki/FAQ#what-does-it-mean-when-a-search-returns--1-ids
        index.nprobe = self.args.faiss_nprobe  # default nprobe is 1, try a few more (when k is large) (TODO try different values)

        # Reconstruct query vectors
        try:
            index.make_direct_map()
        except AttributeError:
            logger.warning('Cannot perform `make_direct_map` (this is OK if the FAISS index is not IVF)')

        query_vecs = np.vstack([index.reconstruct(paper_id_to_idx[paper_id]) for paper_id in items])

        logger.info(f'Query vectors reconstructed: {query_vecs.shape}  (worker #{worker_id})')

        # Search nearest neighbors
        scores, indices = index.search(query_vecs, self.args.get_k_max() + 1)

        self.query_nearest_idxs = indices[:, 1:]  # exclude self

        logger.info(f'Nearest neighbors retrieved: {self.query_nearest_idxs.shape} (worker #{worker_id}; k_max = {self.args.get_k_max()})')

        # Search by range (score = similarity)
        if self.args.has_range_strategy():
            #TODO do only a single search (not search and range_search; contruct regular results from range_search)
            range_max = args.get_range_max()

            logger.info(f'Performing range search (threshold = {range_max})')
            # Perform range search
            range_counts, range_scores, range_indices = index.range_search(query_vecs, range_max)
            logger.info('Range results retrieved')

            # Extract query level results
            self.query_idx_to_range_results = {}  # paper idx => list of tuples (paper_idx, score); descending by score

            # Range count statistics
            logger.info('Range counts: ')
            logger.info(pd.DataFrame(range_counts).describe())

            for i in range(len(query_vecs)):
                query_idxs = range_indices[range_counts[i]:range_counts[i + 1]]  # indicies for query i
                query_scores = range_scores[range_counts[i]:range_counts[i + 1]]

                idxs_scores = dict(zip(query_idxs.tolist(), query_scores.tolist()))
                #TODO partial search idxs[np.argpartition(-d, 5)[:5]]  with range_count
                sorted_range_results = [(idx, score) for idx, score in
                                       sorted(idxs_scores.items(), key=lambda item: -item[1])]  # descending order

                self.query_idx_to_range_results[i] = sorted_range_results

            logger.info('Range results sorted')
        else:
            self.query_idx_to_range_results = None

        # Sample random papers and get n examples with the largest/smallest distance
        if self.args.has_random_sorted_strategy():
            count = self.args.get_random_sorted_count()
            paper_ids = list(paper_id_to_idx.keys())
            self.query_idx_to_random_sorted_ids = {}

            for query_idx, query_paper_id in tqdm(enumerate(items), total=len(items), desc='Random sorted'):
                query_vec = query_vecs[query_idx]

                # Sample n papers from the corpus and take their vectors
                candidate_ids = random.sample(paper_ids, self.args.random_sorted_sample_n)
                candidate_vecs = np.vstack([index.reconstruct(self.paper_id_to_idx[paper_id]) for paper_id in candidate_ids])

                # Compute their similarity to the query paper
                # reconstructed vectors are normalized -> dot => cosine similarity
                cosim = np.dot(candidate_vecs, query_vec)

                # Sort by similarity and take `count` papers with lowest similarity
                lowest_sim_idxs = np.argpartition(cosim, count)[:count]

                # Map to back paper ids
                self.query_idx_to_random_sorted_ids[query_idx] = [self.get_paper_id_by_idx(idx) for idx in lowest_sim_idxs]

        self.init_kmeans()
        self.init_randoms()
