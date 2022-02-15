import logging
import os
import socket
import sys
from typing import Union, List, Optional

import dataclasses
import fire
import wandb
from transformers import TrainingArguments, is_wandb_available

from cli_model import train, DEFAULT_NUM_TRAIN_EPOCHS, DEFAULT_LEARNING_RATE
from cli_specter import find_train_ids, BaseCorpus
from cli_triples import get_metadata, get_specter_triples
from gdt.models import PoolingStrategy
from gdt.triples_miner import TriplesMinerArguments, AnnBackend
from gdt.utils import get_kwargs_for_data_classes, get_workers

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[
        # logging.FileHandler("logs/pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_specter(
        output_dir: str,
        base_model_name_or_path: str,
        scidocs_dir: str,
        s2orc_metadata_dir: Optional[str] = None,
        specter_triples_path: Optional[str] = None,
        graph_paper_ids_path: Optional[str] = None,
        graph_embeddings_path: Optional[str] = None,
        s2id_to_s2orc_input_path: Optional[str] = None,
        train_s2orc_paper_ids: Optional[Union[str, List[str]]] = None,
        train_query_s2orc_paper_ids: Optional[Union[str, List[str]]] = None,
        graph_limit: BaseCorpus = BaseCorpus.SPECTER,
        workers: int = 0,
        bitfit: bool = False,
        masked_language_modeling: bool = False,
        masked_language_modeling_weight: float = 1.0,
        predict_embeddings: bool = False,
        pooling_strategy: PoolingStrategy = PoolingStrategy.CLS,
        max_sequence_length: int = 512,
        val_or_test_or_both: str = 'both',
        query_n_folds: int = 0,
        query_fold_k: Union[int, List[int], str] = 0,
        query_oversampling_ratio: float = 0.0,
        sample_queries_ratio: float = 1.0,
        corpus_seed: Optional[int] = None,
        auto_output_dir: bool = False,
        skip_queries: bool = False,
        skip_triples: bool = False,
        skip_metadata: bool = False,
        skip_train: bool = False,
        skip_eval: bool = False,
        override_triples: bool = False,
        override_queries: bool = False,
        override_metadata: bool = False,
        override_train: bool = False,
        cache_metadata: bool = False,
        gzip: bool = False,
        scidocs_cuda_device: int = -1,
        disable_specter_to_s2orc_mapping: bool = False,
        **kwargs
        ):
    """

    Runs all at once (with difference hyperparameters) -> generate triples -> train -> evaluate

    - Models are saved in $EXP_DIR/model
    - Training arguments are not needed (by default SPECTER settings are used)

    Usage:

    python cli_pipeline.py run_specter $EXP_DIR \
        --base_model_name_or_path $BASE_MODEL \
        --scidocs_dir $SCIDOCS_DIR \
        --s2orc_metadata_dir $S2ORC_METADATA_DIR \
        --specter_triples_path ${SPECTER_DIR}/train_triples.csv \
        --graph_paper_ids_path ${S2ORC_PAPER_IDS} \
        --graph_embeddings_path ${S2ORC_EMBEDDINGS}  \
        --s2id_to_s2orc_input_path ${SPECTER_DIR}/s2id_to_s2orc_paper_id.json \
        --train_s2orc_paper_ids ${EXP_DIR}/s2orc_paper_ids.json \
        --train_query_s2orc_paper_ids ${EXP_DIR}/query_s2orc_paper_ids.json \
        --ann_trees 1000 \
        --triples_per_query 5 \
        --easy_positives_count 5 --easy_positives_strategy knn --easy_positives_k_min 0 --easy_positives_k_max 5 \
        --easy_negatives_count 3 --easy_negatives_strategy random \
        --hard_negatives_count 2 --hard_negatives_strategy knn --hard_negatives_k_min 498 --hard_negatives_k_max 500 \
        --workers $WORKERS

    :param corpus_seed: Use a different random seed for corpus generation than default seed from --seed.
    :param cache_metadata: Extracts metadata for all training paper IDs and saves them as cache file
    :param scidocs_cuda_device: Use this CUDA device for SciDocs evaluation
    :param bitfit: Enable training of bias terms only
    :param override_train: Override trained model
    :param gzip: Uses gzip compression for triples.csv and metadata.json
    :param masked_language_modeling_weight: Weight factor for MLM loss
    :param predict_embeddings: Enable prediction of target embeddings as additional loss
    :param max_sequence_length: Max. tokens for training set (does not apply for test set)
    :param sample_queries_ratio: Post-sampling of query documents (performed after folds etc)
    :param output_dir: All output is saved here
    :param base_model_name_or_path: Base BERT-style Transformer model (see AutoModel.from_pretrained)
    :param scidocs_dir:
    :param s2orc_metadata_dir:
    :param specter_triples_path:
    :param graph_paper_ids_path: Path to paper IDs used in graph embeddings (json file with list)
    :param graph_embeddings_path: Path to pre-computed graph embeddings (h5 file)
    :param s2id_to_s2orc_input_path: Mapping from S2 IDs (SciDocs and SPECTER) to S2ORC (citation graph)
    :param train_s2orc_paper_ids: Path to JSON, List (default: <output_dir>/s2orc_paper_ids.json)
    :param train_query_s2orc_paper_ids: Path to JSON, List (default: <output_dir>/query_s2orc_paper_ids.json)
    :param graph_limit: Limit the underlying citation graph to a specific sub-set
    :param workers:
    :param masked_language_modeling: Enable mask language model loss
    :param pooling_strategy:
    :param val_or_test_or_both:
    :param query_n_folds:
    :param query_fold_k:
    :param query_oversampling_ratio: Pre-sampling
    :param auto_output_dir: Generate output directory based on provided settings
    :param skip_queries:
    :param skip_triples:
    :param skip_metadata:
    :param skip_train:
    :param skip_eval:
    :param override_triples: Override triples
    :param override_queries: Override queries
    :param override_metadata: Override metadata
    :return:
    """

    # Log arg settings
    # write_func_args(inspect.currentframe(), os.path.join(output_dir, 'pipeline.args.json'))

    logger.info(f'Running pipeline in {output_dir}')
    logger.info(f'Host: {socket.gethostname()}')

    triples_miner_kwargs, training_kwargs = get_kwargs_for_data_classes([TriplesMinerArguments, TrainingArguments], kwargs)
    triples_miner_args = TriplesMinerArguments(**triples_miner_kwargs)

    base_model_name = base_model_name_or_path.split('/')[-1]

    if corpus_seed is None:
        corpus_seed = triples_miner_args.seed

    if train_s2orc_paper_ids is None:
        train_s2orc_paper_ids = os.path.join(output_dir, f's2orc_paper_ids.seed_{corpus_seed}.json')

    if train_query_s2orc_paper_ids is None:
        train_query_s2orc_paper_ids = os.path.join(output_dir, f'query_s2orc_paper_ids.seed_{corpus_seed}.json')

    if triples_miner_args.ann_index_path is None:
        # auto path name
        if triples_miner_args.ann_backend == AnnBackend.FAISS:
            triples_miner_args.ann_index_path = train_s2orc_paper_ids + f'.{triples_miner_args.faiss_string_factory}.faiss'
        else:
            raise ValueError(f'cannot determine ann path for backend: {triples_miner_args.ann_backend}')

        logger.info(f'ANN index path automatically set to: {triples_miner_args.ann_index_path}')

    if auto_output_dir:
        # Automatically determining output dir
        base_output_dir = output_dir
        auto_output_dir = os.path.join(output_dir, graph_limit, f'corpus_seed_{corpus_seed}')

        if query_oversampling_ratio > 0:
            auto_output_dir = os.path.join(auto_output_dir, f'oversampling_{query_oversampling_ratio}')

        if query_n_folds > 0:
            auto_output_dir = os.path.join(auto_output_dir, f'folds_{query_n_folds}', f'k_{query_fold_k}')

        if sample_queries_ratio is not None and sample_queries_ratio < 1:
            auto_output_dir = os.path.join(auto_output_dir, f'queries_{sample_queries_ratio}')

        auto_output_dir = os.path.join(auto_output_dir, triples_miner_args.stringify())

        # Override run name
        training_kwargs['run_name'] = auto_output_dir + f' ({base_model_name})'

        output_dir = os.path.join(output_dir, auto_output_dir)

        logger.info(f'Output directory set to: {output_dir}')

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info('Created output directory')
    else:
        base_output_dir = None

    workers = get_workers(workers)

    # Determine model dir depending on settings
    model_dir = os.path.join(output_dir, f'model_{base_model_name}')

    if masked_language_modeling:
        logger.info('Enable masked_language_modeling')
        model_dir += f'_mlm'

        if masked_language_modeling_weight != 1.0:
            logger.info(f'--masked_language_modeling_weight = {masked_language_modeling_weight}')
            model_dir += f'_{masked_language_modeling_weight}'

    if pooling_strategy != PoolingStrategy.CLS:
        logger.info(f'PoolingStrategy {pooling_strategy}')
        model_dir += '_' + pooling_strategy

    if bitfit:
        # Train only bias terms
        model_dir += '_bitfit'

    if 'fp16' in training_kwargs:
        # Float precision
        model_dir += '_fp16'

    if predict_embeddings:
        logger.info('Enable predict_embeddings')
        model_dir += '_predict_embeddings'

    if 'warmup_ratio' in training_kwargs and training_kwargs['warmup_ratio'] > 0:
        logger.info('Custom warmup_ratio')
        model_dir += f'_warmup_ratio_{training_kwargs["warmup_ratio"]}'

    if 'num_train_epochs' in training_kwargs and training_kwargs['num_train_epochs'] != DEFAULT_NUM_TRAIN_EPOCHS:
        logger.info('Custom num_train_epochs')
        model_dir += f'_epochs_{training_kwargs["num_train_epochs"]}'

    if 'learning_rate' in training_kwargs and training_kwargs['learning_rate'] != DEFAULT_LEARNING_RATE:
        logger.info('Custom learning_rate')
        model_dir += f'_lr_{training_kwargs["learning_rate"]}'

    triples_path = os.path.join(output_dir, 'train_triples.csv')
    metadata_path = os.path.join(output_dir, 'train_metadata.jsonl')

    if gzip:
        # Enable gzip compression
        triples_path += '.gz'
        metadata_path += '.gz'

    if skip_queries:
        logger.info('Skipping queries')
    else:
        if os.path.exists(train_s2orc_paper_ids) and os.path.exists(train_query_s2orc_paper_ids)\
                and not override_queries:
            logger.info('Skipping queries (output exists already)')
        else:
            logger.info('Finding query ids')

            # Generate training corpus and query papers
            find_train_ids(
                specter_triples_path,
                scidocs_dir,
                s2id_to_s2orc_input_path,
                s2orc_paper_ids=graph_paper_ids_path,
                output_path=train_s2orc_paper_ids,
                query_output_path=train_query_s2orc_paper_ids,
                query_n_folds=query_n_folds,
                query_fold_k=query_fold_k,
                query_oversampling_ratio=query_oversampling_ratio,
                seed=corpus_seed,  # Use custom seed for corpus generation
                base_corpus=graph_limit,
                map_specter_to_s2orc=(not disable_specter_to_s2orc_mapping),
            )

    if skip_triples:
        logger.info('Skipping triples')
    else:
        if os.path.exists(triples_path) and not override_triples:
            logger.info('Skipping triples (output exists already)')
        else:
            logger.info('Generating triples')

            get_specter_triples(triples_path,
                                scidocs_dir,
                                specter_triples_path,
                                graph_paper_ids_path,
                                graph_embeddings_path,
                                s2id_to_s2orc_input_path,
                                train_s2orc_paper_ids,
                                train_query_s2orc_paper_ids,
                                sample_queries_ratio,
                                graph_limit,
                                workers,
                                triples_miner_args)

    if skip_metadata:
        logger.info('Skipping metadata')
    elif s2orc_metadata_dir is None:
        logger.error('Cannot extract metadata! `s2orc_metadata_dir` is not set.')
        return
    else:
        if os.path.exists(metadata_path) and not override_metadata:
            logger.info('Skipping metdata (exists already)')
        else:
            logger.info('Generating triple metadata')

            # Use metadata JSONL if file exists (this is faster than extracting from S2ORC dump)
            train_s2orc_paper_ids_metadata_path = train_s2orc_paper_ids + '.metadata.jsonl'

            if not os.path.exists(train_s2orc_paper_ids_metadata_path):
                # Cache based on full training corpus
                if cache_metadata:
                    logger.info(f'No metadata cache exists, pre-extract metadata for all paper IDs')
                    get_metadata(input_path=train_s2orc_paper_ids,
                                 output_path=train_s2orc_paper_ids_metadata_path,
                                 s2orc_metadata_dir=s2orc_metadata_dir,
                                 workers=workers,
                                 jsonl_metadata_path=train_s2orc_paper_ids_metadata_path)

                else:
                    train_s2orc_paper_ids_metadata_path = None

            # Extract metadata for triples
            get_metadata(input_path=triples_path,
                         output_path=metadata_path,
                         s2orc_metadata_dir=s2orc_metadata_dir,
                         workers=workers,
                         jsonl_metadata_path=train_s2orc_paper_ids_metadata_path)

    if skip_train:
        logger.info('Skipping train')
    else:
        if not os.path.exists(triples_path):
            logger.error('Cannot train: triples does not exist')
            return

        if not os.path.exists(metadata_path):
            logger.error('Cannot train: triples does not exist')
            return

        if os.path.exists(model_dir) and not override_train:
            logger.error(f'Model dir exists already: {model_dir}')
            return

        logger.info('Training model')

        train(
            model_dir,
            base_model_name_or_path,
            output_dir,
            scidocs_dir,
            scidocs_cuda_device=scidocs_cuda_device,
            use_dataset_cache=True,
            abstract_only=False,
            workers=workers,
            masked_language_modeling=masked_language_modeling,
            masked_language_modeling_weight=masked_language_modeling_weight,
            predict_embeddings=predict_embeddings,
            pooling_strategy=pooling_strategy,
            do_eval=False if skip_eval else True,
            val_or_test_or_both=val_or_test_or_both,
            max_sequence_length=max_sequence_length,
            graph_paper_ids_path=graph_paper_ids_path,
            graph_embeddings_path=graph_embeddings_path,
            bitfit=bitfit,
            **training_kwargs,
            # **training_args.to_sanitized_dict()
            # output_dir=model_dir
        )

    # Log additional (to Weights & Biases)
    if is_wandb_available() and hasattr(wandb.config, 'update'):
        wandb.config.update(dataclasses.asdict(triples_miner_args), allow_val_change=True)
        wandb.config.update({
            'workers': workers,
            'graph_limit': graph_limit,
            'graph_paper_ids_path': graph_paper_ids_path,
            'graph_embeddings_path': graph_embeddings_path,
            's2id_to_s2orc_input_path': s2id_to_s2orc_input_path,
            'train_s2orc_paper_ids': train_s2orc_paper_ids,
            'train_query_s2orc_paper_ids': train_query_s2orc_paper_ids,
            'query_oversampling_ratio': query_oversampling_ratio,
            'query_fold_k': query_fold_k,
            'query_n_folds': query_n_folds,
            'corpus_seed': corpus_seed,
        }, allow_val_change=True)

    # if skip_eval:
    #     logger.info('Skipping eval')
    # else:
    #     logger.info('Evaluating model')
    #
    #     evaluate(model_dir, output_dir, scidocs_dir=scidocs_dir, use_dataset_cache=True)

    logger.info('done')


if __name__ == '__main__':
    fire.Fire()
    sys.exit(0)
