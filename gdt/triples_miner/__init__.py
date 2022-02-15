import enum
import logging
from typing import Optional

import dataclasses
from dataclasses import dataclass, field

from gdt import DEFAULT_SEED

logger = logging.getLogger(__name__)

DEFAULT_TRIPLES_PER_QUERY = 5
DEFAULT_EASY_POSITIVES_COUNT = 5
DEFAULT_EASY_NEGATIVES_COUNT = 3
DEFAULT_HARD_NEGATIVES_COUNT = 2
DEFAULT_HARD_NEGATIVES_K_MIN = 497
DEFAULT_HARD_NEGATIVES_K_MAX = 500


class ExampleType(str, enum.Enum):
    POSITIVES = 'positives'
    NEGATIVES = 'negatives'


class ExampleDifficulty(str, enum.Enum):
    EASY = 'easy'
    MEDIUM = 'medium'
    HARD = 'hard'


class SamplingStrategy(str, enum.Enum):
    KNN = 'knn'
    RANDOM = 'random'

    # Random candidates, compute distance with query, and select top k candidates depending on distance
    RANDOM_SORTED = 'random_sorted'

    # Random samples excluding the knn results
    RANDOM_WITHOUT_KNN = 'random_without_knn'

    # Random samples excluding the positives samples (independent from positive strategy)
    RANDOM_WITHOUT_POSITIVES = 'random_without_positives' # TODO

    # Combination of RANDOM_SORTED and RANDOM_WITHOUT_KNN
    RANDOM_WITHOUT_KNN_SORTED = 'random_without_knn_sorted'

    # Random samples excluding the range results
    RANDOM_WITHOUT_RANGE = 'random_without_range'

    # KNN_TEXT_TIFDF = 'knn_text-tfidf'  # TODO similar graph but dissimilar text
    KMEANS = 'kmeans'

    # See faiss range search https://github.com/facebookresearch/faiss/wiki/Special-operations-on-indexes#range-search
    # -> "select samples within a given distance range from the query"
    # -> for easy and hard negatives?
    RANGE = 'range'

    # TODO
    OPPOSING_KNN = 'opposing_knn'


class AnnBackend(str, enum.Enum):
    """
    Backend used for approximate nearest neighbor search
    """
    FAISS = 'faiss'
    ANNOY = 'annoy'


@dataclass
class TriplesMinerArguments:
    """
    Arguments for triple mining worker (strategy settings etc.)
    """
    ann_trees: Optional[int] = field(
        default=1000,
        metadata={"help": "Trees for ANN index"},
    )
    ann_index_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to ANN index"},
    )
    ann_metric: Optional[str] = field(
        default='euclidean',
        metadata={"help": "ANN index metric (default: euclidean; for cosine similarity use `dot` with normalized vectors)"},
    )
    ann_workers: Optional[int] = field(
        default=None,
        metadata={"help": "Use workers in parallel for ANN (None = default worker count)"}
    )
    ann_normalize_embeddings: bool = field(
        default=False,
        metadata={"help": "Normalize embeddings before indexing to ANN."}
    )
    ann_preload: bool = field(
        default=False,
        # See https://github.com/spotify/annoy#full-python-api
        metadata={"help": "Annoy: If prefault is set to True, it will pre-read the entire file into memory "
                          "(using mmap with MAP_POPULATE). Default is False."}
    )
    ann_backend: AnnBackend = field(
        default=AnnBackend.FAISS,
        metadata={"help": "Approximate nearest neighbors backend used for NN search (annoy or faiss)."}
    )
    ann_device: str = field(
        default='all',
        metadata={"help": "CUDA device for ANN training (all = all avialable GPUs)."}
    )
    ann_train_size: int = field(
        default=0,
        metadata={"help": "Training ANN on this number of samples"}
    )
    ann_batch_size: int = field(
        default=1000,
        metadata={"help": "ANN batch size"}
    )
    faiss_nprobe: Optional[int] = field(
        default=1000,
        metadata={"help": "the number of cells (out of nlist) that are visited to perform a search; "
                          "faiss default nprobe is 1, try a few more (when k is large) See https://github.com/facebookresearch/faiss/wiki/FAQ#what-does-it-mean-when-a-search-returns--1-ids"},
    )
    faiss_string_factory: Optional[str] = field(
        default='Flat',
        metadata={"help": "See https://github.com/facebookresearch/faiss/wiki/The-index-factory"}
    )
    triples_per_query: Optional[int] = field(
        default=DEFAULT_TRIPLES_PER_QUERY,
        metadata={"help": "Number of triples generated for each query document"},
    )
    k_means_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to k means output (paper_idx_to_centroid.json, centroids.npy, sorted_centroids.npy, similarities.npy)"},
    )
    range_as_similarity: bool = field(
        default=True,
        metadata={"help": "Range is measured as `similarity` (range = 1.0 => seed; 0.1 => far away; false: measured as distance)"},
    )
    min_range_results_count: Optional[int] = field(
        default=10,
        metadata={"help": "Number of results that a `range_search` must at least return (if number is too low, there is an overlap to knn)"},
    )
    range_fallback_to_random: bool = field(
        default=True,
        metadata={
            "help": "If range search does not returned enough results, do random sampling as fallback."
        },
    )
    random_sorted_sample_n: Optional[int] = field(
        default=100,
        metadata={"help": "Sample n candidates for RANDOM_SORTED strategy"},
    )
    random_sample_factor: Optional[int] = field(
        default=0,
        metadata={"help": "How many times is the corpus added to the random queue (default = 0 = depends on strategy)"},
    )
    seed: int = field(
        default=DEFAULT_SEED,
        metadata={"help": "Seed for random sampling"},
    )

    # Easy positives
    easy_positives_count: Optional[int] = field(
        default=DEFAULT_EASY_POSITIVES_COUNT,
        metadata={"help": "Number of positives generated for each query document"},
    )
    easy_positives_strategy: Optional[SamplingStrategy] = field(
        default=SamplingStrategy.KNN,
        metadata={"help": "Sampling strategy for easy positives"},
    )
    easy_positives_k_min: Optional[int] = field(
        default=0,
        metadata={"help": ""},
    )
    easy_positives_k_max: Optional[int] = field(
        default=5,
        metadata={"help": ""},
    )
    easy_positives_repeat: Optional[bool] = field(
        default=False,
        metadata={"help": "Repeat the same document for all triples"},
    )
    easy_positives_range: Optional[float] = field(
        default=None,
        metadata={"help": "Distance threshold for range search."}
    )

    # Medium positives
    medium_positives_count: Optional[int] = field(
        default=0,
        metadata={"help": "Number of positives generated for each query document"},
    )
    medium_positives_strategy: Optional[SamplingStrategy] = field(
        default=SamplingStrategy.KNN,
        metadata={"help": "Sampling strategy for easy positives"},
    )
    medium_positives_k_min: Optional[int] = field(
        default=0,
        metadata={"help": ""},
    )
    medium_positives_k_max: Optional[int] = field(
        default=0,
        metadata={"help": ""},
    )
    medium_positives_repeat: Optional[bool] = field(
        default=False,
        metadata={"help": "Repeat the same document for all triples"},
    )
    medium_positives_range: Optional[float] = field(
        default=None,
        metadata={"help": "Distance threshold for range search."}
    )

    # Hard positives
    hard_positives_count: Optional[int] = field(
        default=0,
        metadata={"help": "Number of positives generated for each query document"},
    )
    hard_positives_strategy: Optional[SamplingStrategy] = field(
        default=SamplingStrategy.KNN,
        metadata={"help": "Sampling strategy for easy positives"},
    )
    hard_positives_k_min: Optional[int] = field(
        default=0,
        metadata={"help": ""},
    )
    hard_positives_k_max: Optional[int] = field(
        default=0,
        metadata={"help": ""},
    )
    hard_positives_repeat: Optional[bool] = field(
        default=False,
        metadata={"help": "Repeat the same document for all triples"},
    )
    hard_positives_range: Optional[float] = field(
        default=None,
        metadata={"help": "Distance threshold for range search."}
    )

    # Easy negatives
    easy_negatives_count: Optional[int] = field(
        default=DEFAULT_EASY_NEGATIVES_COUNT,
        metadata={"help": "Number of negatives generated for each query document"},
    )
    easy_negatives_strategy: Optional[SamplingStrategy] = field(
        default=SamplingStrategy.RANDOM,
        metadata={"help": "Sampling strategy for negatives"},
    )
    easy_negatives_k_min: Optional[int] = field(
        default=0,
        metadata={"help": ""},
    )
    easy_negatives_k_max: Optional[int] = field(
        default=0,
        metadata={"help": ""},
    )
    easy_negatives_repeat: Optional[bool] = field(
        default=False,
        metadata={"help": "Repeat the same document for all triples"},
    )
    easy_negatives_range: Optional[float] = field(
        default=None,
        metadata={"help": "Distance threshold for range search."}
    )

    # Medium negatives
    medium_negatives_count: Optional[int] = field(
        default=0,
        metadata={"help": "Number of negatives generated for each query document"},
    )
    medium_negatives_strategy: Optional[SamplingStrategy] = field(
        default=SamplingStrategy.KNN,
        metadata={"help": "Sampling strategy for negatives"},
    )
    medium_negatives_k_min: Optional[int] = field(
        default=0,
        metadata={"help": ""},
    )
    medium_negatives_k_max: Optional[int] = field(
        default=0,
        metadata={"help": ""},
    )
    medium_negatives_repeat: Optional[bool] = field(
        default=False,
        metadata={"help": "Repeat the same document for all triples"},
    )
    medium_negatives_range: Optional[float] = field(
        default=None,
        metadata={"help": "Distance threshold for range search."}
    )

    # Hard negatives
    hard_negatives_count: Optional[int] = field(
        default=DEFAULT_HARD_NEGATIVES_COUNT,
        metadata={"help": "Number of negatives generated for each query document"},
    )
    hard_negatives_strategy: Optional[SamplingStrategy] = field(
        default=SamplingStrategy.KNN,
        metadata={"help": "Sampling strategy for negatives"},
    )
    hard_negatives_k_min: Optional[int] = field(
        default=DEFAULT_HARD_NEGATIVES_K_MIN,
        metadata={"help": ""},
    )
    hard_negatives_k_max: Optional[int] = field(
        default=DEFAULT_HARD_NEGATIVES_K_MAX,
        metadata={"help": ""},
    )
    hard_negatives_repeat: Optional[bool] = field(
        default=False,
        metadata={"help": "Repeat the same document for all triples"},
    )
    hard_negatives_range: Optional[float] = field(
        default=None,
        metadata={"help": "Distance threshold for range search."}
    )

    def stringify_strategy(self, example_type: ExampleType, example_difficulty: ExampleDifficulty):
        prefix = example_difficulty + '_' + example_type + '_'
        strategy = getattr(self, prefix + 'strategy')
        count = getattr(self, prefix + 'count')
        k_min = getattr(self, prefix + 'k_min')
        k_max = getattr(self, prefix + 'k_max')
        range = getattr(self, prefix + 'range')
        repeat = getattr(self, prefix + 'repeat')

        if count < 1:
            return ''

        s = '_'

        if example_difficulty == ExampleDifficulty.EASY:
            s += 'e'
        elif example_difficulty == ExampleDifficulty.MEDIUM:
            s += 'm'
        else:
            s += 'h'

        if example_type == ExampleType.POSITIVES:
            s += 'p'
        else:
            s += 'n'

        s += f'{count}{strategy}'

        if strategy == SamplingStrategy.KNN:
            s += f'{k_min}-{k_max}'

        if strategy == SamplingStrategy.RANGE:
            s += f'{range}'

        if repeat:
            s += 'repeat'

        return s

    def stringify(self):
        """Converts arguments into a single string that can be used for directory names"""
        s = f'seed_{self.seed}'

        for example_type in [ExampleType.POSITIVES, ExampleType.NEGATIVES]:
            for example_difficulty in [ExampleDifficulty.EASY, ExampleDifficulty.MEDIUM, ExampleDifficulty.HARD]:
                s += self.stringify_strategy(example_type, example_difficulty)

        if self.triples_per_query != DEFAULT_TRIPLES_PER_QUERY:
            s += f'_tpq{self.triples_per_query}'

        if self.has_random_sorted_strategy():
            s += f'_rsorted_n{self.random_sorted_sample_n}'

        if self.ann_backend == AnnBackend.ANNOY:
            s += f'_{self.ann_metric}{self.ann_trees}'

        return s

    def to_cli_args(self) -> str:
        """Return key-value pairs for fields different from default for CLI arguments"""
        changed_args = {}

        for f in dataclasses.fields(self):
            current_val = getattr(self, f.name)

            if f.default != current_val:
                changed_args[f.name] = current_val

        return ' '.join({f'--{k}={v}' for k, v in changed_args.items()})

    @staticmethod
    def args_or_kwargs(args, kwargs):
        if kwargs and not args:
            args = TriplesMinerArguments(**kwargs)

        args.sanity_check()

        return args

    def sanity_check(self):
        """
        Check if all settings are plausible
        """
        assert self.triples_per_query == \
               self.easy_positives_count + self.medium_positives_count + self.hard_positives_count
        assert self.triples_per_query == \
               self.easy_negatives_count + self.medium_negatives_count + self.hard_negatives_count

        # k range can be greater (random sample within range) but not smaller than count
        if self.easy_positives_count > 0 and self.easy_positives_strategy == SamplingStrategy.KNN:
            assert self.easy_positives_count <= self.easy_positives_k_max - self.easy_positives_k_min

        if self.medium_positives_count > 0 and self.medium_positives_strategy == SamplingStrategy.KNN:
            assert self.medium_positives_count <= self.medium_positives_k_max - self.medium_positives_k_min

        if self.hard_positives_count > 0 and self.hard_positives_strategy == SamplingStrategy.KNN:
            assert self.hard_positives_count <= self.hard_positives_k_max - self.hard_positives_k_min

        if self.easy_negatives_count > 0 and self.easy_negatives_strategy == SamplingStrategy.KNN:
            assert self.easy_negatives_count <= self.easy_negatives_k_max - self.easy_negatives_k_min

        if self.medium_negatives_count > 0 and self.medium_negatives_strategy == SamplingStrategy.KNN:
            assert self.medium_negatives_count <= self.medium_negatives_k_max - self.medium_negatives_k_min

        if self.hard_negatives_count > 0 and self.hard_negatives_strategy == SamplingStrategy.KNN:
            assert self.hard_negatives_count <= self.hard_negatives_k_max - self.hard_negatives_k_min

    def has_k_means_strategy(self):
        if self.easy_positives_strategy == SamplingStrategy.KMEANS \
                or self.medium_positives_strategy == SamplingStrategy.KMEANS \
                or self.hard_positives_strategy == SamplingStrategy.KMEANS \
                or self.easy_negatives_strategy == SamplingStrategy.KMEANS \
                or self.medium_negatives_strategy == SamplingStrategy.KMEANS \
                or self.hard_negatives_strategy == SamplingStrategy.KMEANS:
            return True
        else:
            return False

    def has_range_strategy(self):
        if self.easy_positives_strategy == SamplingStrategy.RANGE \
                or self.medium_positives_strategy == SamplingStrategy.RANGE \
                or self.hard_positives_strategy == SamplingStrategy.RANGE \
                or self.easy_negatives_strategy == SamplingStrategy.RANGE \
                or self.medium_negatives_strategy == SamplingStrategy.RANGE \
                or self.hard_negatives_strategy == SamplingStrategy.RANGE:
            return True
        else:
            return False

    def has_random_sorted_strategy(self):
        if self.easy_positives_strategy == SamplingStrategy.RANDOM_SORTED \
                or self.medium_positives_strategy == SamplingStrategy.RANDOM_SORTED \
                or self.hard_positives_strategy == SamplingStrategy.RANDOM_SORTED \
                or self.easy_negatives_strategy == SamplingStrategy.RANDOM_SORTED \
                or self.medium_negatives_strategy == SamplingStrategy.RANDOM_SORTED \
                or self.hard_negatives_strategy == SamplingStrategy.RANDOM_SORTED:
            return True
        else:
            return False

    def has_random_without_knn_sorted_strategy(self):
        if self.easy_positives_strategy == SamplingStrategy.RANDOM_WITHOUT_KNN_SORTED \
                or self.medium_positives_strategy == SamplingStrategy.RANDOM_WITHOUT_KNN_SORTED \
                or self.hard_positives_strategy == SamplingStrategy.RANDOM_WITHOUT_KNN_SORTED \
                or self.easy_negatives_strategy == SamplingStrategy.RANDOM_WITHOUT_KNN_SORTED \
                or self.medium_negatives_strategy == SamplingStrategy.RANDOM_WITHOUT_KNN_SORTED \
                or self.hard_negatives_strategy == SamplingStrategy.RANDOM_WITHOUT_KNN_SORTED:
            return True
        else:
            return False

    def get_k_max(self, positives=True, negatives=True):
        ks = []

        if positives:
            if self.easy_positives_strategy == SamplingStrategy.KNN:
                ks.append(self.easy_positives_k_max)

            if self.medium_positives_strategy == SamplingStrategy.KNN:
                ks.append(self.medium_negatives_k_max)

            if self.hard_positives_strategy == SamplingStrategy.KNN:
                ks.append(self.hard_positives_k_max)

        if negatives:
            if self.easy_negatives_strategy == SamplingStrategy.KNN:
                ks.append(self.easy_negatives_k_max)

            if self.medium_negatives_strategy == SamplingStrategy.KNN:
                ks.append(self.medium_negatives_k_max)

            if self.hard_negatives_strategy == SamplingStrategy.KNN:
                ks.append(self.hard_negatives_k_max)

        return max(ks)

    def get_range_max(self):
        rs = []

        if self.easy_positives_strategy == SamplingStrategy.RANGE:
            rs.append(self.easy_positives_range)

        if self.medium_positives_strategy == SamplingStrategy.RANGE:
            rs.append(self.medium_negatives_range)

        if self.hard_positives_strategy == SamplingStrategy.RANGE:
            rs.append(self.hard_positives_range)

        if self.easy_negatives_strategy == SamplingStrategy.RANGE:
            rs.append(self.easy_negatives_range)

        if self.medium_negatives_strategy == SamplingStrategy.RANGE:
            rs.append(self.medium_negatives_range)

        if self.hard_negatives_strategy == SamplingStrategy.RANGE:
            rs.append(self.hard_negatives_range)

        if self.range_as_similarity:
            # similarity -> take minimum
            return min(rs)
        else:
            # distance
            return max(rs)

    def get_count_by_strategy(self, strategy: SamplingStrategy):
        r = 0

        if self.easy_positives_strategy == strategy:
            r += self.easy_positives_count

        if self.medium_positives_strategy == strategy:
            r += self.medium_positives_count

        if self.hard_positives_strategy == strategy:
            r += self.hard_positives_count

        if self.easy_negatives_strategy == strategy:
            r += self.easy_negatives_count

        if self.medium_negatives_strategy == strategy:
            r += self.medium_negatives_count

        if self.hard_negatives_strategy == strategy:
            r += self.hard_negatives_count

        return r

    def get_random_count(self):
        """
        Number of random examples per query
        """
        return self.get_count_by_strategy(SamplingStrategy.RANDOM)

    def get_random_without_knn_count(self):
        """
        Number of RANDOM_WITHOUT_KNN examples per query
        """
        return self.get_count_by_strategy(SamplingStrategy.RANDOM_WITHOUT_KNN)

    def get_random_without_range_count(self):
        """
        Number of RANDOM_WITHOUT_RANGE examples per query
        """
        return self.get_count_by_strategy(SamplingStrategy.RANDOM_WITHOUT_RANGE)

    def get_random_sorted_count(self):
        """
        Number of random sorted examples per query
        """
        r = 0
        error_msg = 'RANDOM_SORTED strategy cannot be assigned to more than once!'

        if self.easy_positives_strategy == SamplingStrategy.RANDOM_SORTED:
            if r > 0:
                raise ValueError(error_msg)

            r += self.easy_positives_count

        if self.medium_positives_strategy == SamplingStrategy.RANDOM_SORTED:
            if r > 0:
                raise ValueError(error_msg)

            r += self.medium_positives_count

        if self.hard_positives_strategy == SamplingStrategy.RANDOM_SORTED:
            if r > 0:
                raise ValueError(error_msg)

            r += self.hard_positives_count

        if self.easy_negatives_strategy == SamplingStrategy.RANDOM_SORTED:
            if r > 0:
                raise ValueError(error_msg)

            r += self.easy_negatives_count

        if self.medium_negatives_strategy == SamplingStrategy.RANDOM_SORTED:
            if r > 0:
                raise ValueError(error_msg)

            r += self.medium_negatives_count

        if self.hard_negatives_strategy == SamplingStrategy.RANDOM_SORTED:
            if r > 0:
                raise ValueError(error_msg)

            r += self.hard_negatives_count

        return r
