import json
import logging
import math
import os
import pickle
import sys
from typing import Optional

import fire
import torch
import wandb
from smart_open import open
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import Trainer, TrainingArguments, is_wandb_available, AutoTokenizer

from experiments.environment import get_env
from gdt import DEFAULT_SEED
from gdt.datasets.scidocs import SciDocsDataset
from gdt.datasets.triples import TripleDataset
from gdt.models import PoolingStrategy, deactivate_bias_gradients
from gdt.models.auto_modeling import AutoModelForTripletLoss
from gdt.scidocs_trainer import SciDocsTrainer
from gdt.utils import get_scidocs_compute_metrics, get_workers, get_auto_train_batch_size, DictOfListsDataset, \
    get_scidocs_metadata

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Same as SPECTER:
# https://github.com/allenai/specter/blob/22af37904c1540ed870b38e4cd0120a6f6705b74/scripts/pytorch_lightning_training_script/train.py#L495
DEFAULT_NUM_TRAIN_EPOCHS = 2
DEFAULT_LEARNING_RATE = 2e-5


def train(
        model_output_dir: str,
        base_model_name_or_path: str,
        dataset_dir: str,
        scidocs_dir: Optional[str],
        scidocs_cuda_device: int = -1,
        use_dataset_cache: bool = False,
        abstract_only: bool = False,
        workers: int = 10,
        masked_language_modeling: bool = False,
        masked_language_modeling_weight: float = 1.0,
        predict_embeddings: bool = False,
        pooling_strategy: PoolingStrategy = PoolingStrategy.CLS,
        val_or_test_or_both: str = 'both',
        pairwise: bool = False,
        sample_n: int = 0,
        sample_ratio: float = .0,
        s2orc_metadata_dir: str = None,
        graph_embeddings_path: str = None,
        graph_paper_ids_path: str = None,
        max_sequence_length: int = 512,
        bitfit: bool = False,
        **training_args):
    """
    Trains a Transformer (BERT) model on triple dataset with triplet loss (and evaluates it on SciDocs)

    python cli_model.py train --model_output_dir ./data/gdt/models/gdt.epoch_20_768d.easy_3.hard_2.k_100b \
        --base_model_name_or_path scibert-scivocab-uncased \
        --dataset_dir ./data/gdt/epoch_20_768d.easy_3.hard_2.k_100 \
        --scidocs_dir ./data/scidocs \
        --num_train_epochs 1

    :param bitfit: Train bias terms only
    :param scidocs_cuda_device:
    :param max_sequence_length:
    :param predict_embeddings: Enable prediction of target embeddings as additional loss
    :param sample_n:
    :param sample_ratio:
    :param graph_paper_ids_path:
    :param graph_embeddings_path:
    :param s2orc_metadata_dir:
    :param pairwise: Enable pairwise training mode (default triple training)
    :param pooling_strategy:
    :param val_or_test_or_both: Evaluate on val, test, or val and test
    :param masked_language_modeling: Use also MLM loss for training
    :param workers:
    :param model_output_dir: Model is saved in this directory
    :param abstract_only: Use abstract only instead of "title [SEP] abstract" for dataset
    :param run_name: Run name (used for model output dir)
    :param models_dir: Base directory for models
    :param base_model_name_or_path: Start training from this checkpoint
    :param dataset_dir: Directory with train_triples.csv and train_metadata.jsonl
    :param scidocs_dir: Path to SciDocs data (for evaluation)
    :param use_dataset_cache: Load dataset from pickle file
    :param training_args: Extra training arguments. See https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments
    :return:
    """

    if training_args is None:
        training_args = {}

    env = get_env()
    n_gpus = torch.cuda.device_count()
    workers = get_workers(workers)

    default_batch_size = get_auto_train_batch_size(default_size=8, model_name_or_path=base_model_name_or_path)  # max memory for our GPUs during training
    target_batch_size = 32  # same as SPECTER

    logger.info(f'GPU allows batch size = {default_batch_size} (GPU count = {n_gpus})')

    if n_gpus * default_batch_size < target_batch_size:
        gradient_accumulation_steps = int(math.ceil(target_batch_size / (n_gpus * default_batch_size)))

        logger.info(f'extra gradient_accumulation_steps = {gradient_accumulation_steps}')
    else:
        logger.info(f'no extra gradient_accumulation_steps needed')
        gradient_accumulation_steps = 1

    # if run_name:
    # model_output_dir = os.path.join(models_dir, run_name)  # append run name to output dir

    """
    SPECTER:
    
    Based on initial experiments, we use a margin m=1 for the triplet loss. 

    For training, we use the Adam opti- mizer (Kingma and Ba, 2014)
    following the sug- gested hyperparameters in Devlin et al. (2019) 
    (LR: 2e-5, Slanted Triangular LR scheduler10 (Howard and Ruder, 2018) with number of train steps equal to training instances and cut fraction of 0.1). 

    We train the model on a single Titan V GPU (12G memory) for 2 epochs, 
    with batch size of 4 (the maximum that fit in our GPU memory) and 

    use gradient accumulation for an effective batch size of 32. 

    Each training epoch takes approximately 1-2 days to complete on the full dataset.
    """
    default_training_args = dict(
        output_dir=model_output_dir,
        overwrite_output_dir=True,
        seed=DEFAULT_SEED,
        gradient_accumulation_steps=gradient_accumulation_steps,

        per_device_train_batch_size=default_batch_size,
        per_device_eval_batch_size=default_batch_size * 2,

        learning_rate=DEFAULT_LEARNING_RATE,
        num_train_epochs=DEFAULT_NUM_TRAIN_EPOCHS,
        logging_steps=5,

        save_total_limit=3,
        save_strategy='epoch',  # no, steps, epoch
        save_steps=1,

        evaluation_strategy='epoch',  # no, steps, epoch
        eval_steps=1,
        do_eval=True,

        # dataloader_num_workers
        # disable_tqdm
        # load_best_model_at_end=True
        # metric_for_best_model='eval_avg'
        # greater_is_better=True

        report_to='wandb',
    )

    training_args = dict(default_training_args, **training_args)

    # override training args
    training_args.update(dict(
        do_train=True,
        do_eval=True,
    ))

    logger.info(f'Training arguments: {training_args}')

    training_args = TrainingArguments(**training_args)

    if not os.path.exists(base_model_name_or_path):
        if os.path.exists(os.path.join(env['bert_dir'], base_model_name_or_path)):
            base_model_name_or_path = os.path.join(env['bert_dir'], base_model_name_or_path)

    # Load model and tokenizer (through auto mode)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, use_fast=True)
    # model_config = AutoConfig.from_pretrained(base_model_name_or_path)
    # base_model = AutoModel.from_pretrained(base_model_name_or_path)

    if pairwise:
        logger.info('Pairwise training enabled')
        # model = BertForPairwiseCosineSimilarityLoss(  # TODO works currently only for BERT
        #     model_config,
        #     masked_language_modeling=masked_language_modeling,
        #     pooling_strategy=pooling_strategy,
        # )
        # train_ds = TargetEmbeddingDataset(
        #     train_ids=os.path.join(dataset_dir, 's2orc_paper_ids.json'),
        #     s2orc_metadata_dir=s2orc_metadata_dir,
        #     tokenizer=tokenizer,
        #     graph_paper_ids_path=graph_paper_ids_path,
        #     graph_embeddings_path=graph_embeddings_path,
        #     sample_n=sample_n,
        #     sample_ratio=sample_ratio
        # )
        raise NotImplementedError()
    else:
        # triples
        model = AutoModelForTripletLoss.from_pretrained(
            base_model_name_or_path,
            masked_language_modeling=masked_language_modeling,
            masked_language_modeling_weight=masked_language_modeling_weight,
            predict_embeddings=predict_embeddings,
            pooling_strategy=pooling_strategy,
        )

        # Build train dataset
        train_ds = TripleDataset(
            os.path.join(dataset_dir, 'train_triples.csv'),
            os.path.join(dataset_dir, 'train_metadata.jsonl'),
            tokenizer,
            # sample_n=10_000
            abstract_only=abstract_only,
            use_cache=use_dataset_cache,
            mask_anchor_tokens=masked_language_modeling,
            max_sequence_length=max_sequence_length,
            predict_embeddings=predict_embeddings,
            graph_embeddings_path=graph_embeddings_path,
            graph_paper_ids_path=graph_paper_ids_path
        )
        train_ds.load()

    # Test dataset from SciDocs
    test_ds = SciDocsDataset(
        scidocs_dir,
        tokenizer,
        use_cache=use_dataset_cache,
        inference_prefix='' if pairwise else 'anchor_'  # pairwise does not use any input prefix for inference
    )
    test_ds.load()

    if bitfit:
        logger.info('BitFit enabled')
        model = deactivate_bias_gradients(model)

    logger.info('Initializing trainer')

    trainer = SciDocsTrainer(
        model=model,
        args=training_args,
        eval_dataset=test_ds,
        train_dataset=train_ds,
        # compute_metrics=get_scidocs_compute_metrics(test_ds, workers=env['workers'], cuda_device=-1),
    )
    trainer.val_or_test_or_both = val_or_test_or_both
    trainer.scidocs_ds = test_ds
    trainer.scidocs_cuda_device = scidocs_cuda_device
    trainer.workers = workers

    logger.info('Starting trainer')

    training_out = trainer.train()

    logger.info(f'Trainer completed: {training_out}')

    # Log additional (to Weights & Biases)
    if is_wandb_available() and hasattr(wandb.config, 'update'):
        wandb.config.update(train_ds.get_stats())
        wandb.config.update({
            'base_model_name_or_path': base_model_name_or_path,
            'masked_language_modeling': masked_language_modeling,
            'predict_embeddings': predict_embeddings,
            'abstract_only': abstract_only,
            'use_dataset_cache': use_dataset_cache,
            'pooling_strategy': pooling_strategy,
            'max_sequence_length': max_sequence_length,
        }, allow_val_change=True)

    # save
    logger.info(f'Saving to {training_args.output_dir}')
    #TODO settings + args?
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)

    trainer.state.save_to_json(os.path.join(training_args.output_dir, 'trainer_state.json'))

    with open(os.path.join(training_args.output_dir, 'training_out.json'), 'w') as f:
        json.dump(training_out, f)

    with open(os.path.join(training_args.output_dir, 'training_args.json'), 'w') as f:
        json.dump(training_args.to_sanitized_dict(), f)


def evaluate(
        model_path: str,
        dataset_dir: str,
        scidocs_dir: str,
        use_dataset_cache: bool = False,
        val_or_test: str = 'test',
        batch_size: int = 12,
        output_dir: Optional[str] = None
        ):
    """
    Evaluate triple loss model on SciDocs

    python cli_model.py evaluate gdt_1 --dataset_dir ./data/gdt/ \
        --scidocs_dir /home/mostendorff/experiments/scidocs/data \
        --use_dataset_cache

    python cli_model.py evaluate citebert --scidocs_dir ${SCIDOCS_DIR} --dataset_dir ./data/sci/baseline_citebert --output_dir ./data/sci/baseline_citebert --batch_size 64 --use_dataset_cache
    python cli_model.py evaluate bert-base-uncased --scidocs_dir ${SCIDOCS_DIR} --dataset_dir ./data/sci/baseline_bert-base-uncased --output_dir ./data/sci/baseline_bert-base-uncased --batch_size 128 --use_dataset_cache
    python cli_model.py evaluate biobert-base-cased-v1.2 --scidocs_dir ${SCIDOCS_DIR} --dataset_dir ./data/sci/baseline_biobert-base-cased-v1.2 --output_dir ./data/sci/baseline_biobert-base-cased-v1.2 --batch_size 128 --use_dataset_cache


    :param model_path: Path to model checkpoint
    :param dataset_dir:
    :param scidocs_dir:
    :param models_dir:
    :param use_dataset_cache:
    :return:
    """

    env = get_env()

    if not os.path.exists(model_path) and os.path.exists(os.path.join(env['bert_dir'], model_path)):
        model_path = os.path.join(env['bert_dir'], model_path)

    if output_dir is None:
        output_dir = model_path + '_eval'

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, fast=True)
    model = AutoModelForTripletLoss.from_pretrained(model_path)

    # Dataset
    if use_dataset_cache and os.path.exists(os.path.join(dataset_dir, 'test_dataset.pickle')):
        logger.info('Loading test dataset from cache')
        with open(os.path.join(dataset_dir, 'test_dataset.pickle'), 'rb') as f:
            test_ds = pickle.load(f)
    else:
        test_ds = SciDocsDataset(scidocs_dir, tokenizer, )
        test_ds.load()

        # save to disk
        if use_dataset_cache:
            logger.info('Saving cache to disk ...')

            with open(os.path.join(dataset_dir, 'test_dataset.pickle'), 'wb') as f:
                pickle.dump(test_ds, f)

    logger.info('Initializing trainer')

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=False,
            per_device_eval_batch_size=batch_size,
        ),
        # eval_dataset=test_ds,
        # train_dataset=train_ds,
        # compute_metrics=get_scidocs_compute_metrics(test_ds, workers=env['workers'], cuda_device=-1),
    )

    logger.info('Predict on test set')

    pred_out = trainer.predict(test_ds)

    logger.info('Evaluate with SciDocs')

    eval_metrics = get_scidocs_compute_metrics(test_ds, workers=env['workers'], cuda_device=-1, val_or_test=val_or_test)(pred_out)

    logger.info(eval_metrics)

    with open(os.path.join(output_dir, f'eval_metrics.{val_or_test}.json'), 'w') as f:
        json.dump(eval_metrics, f)

    logger.info('done')


def evaluate_specter_pl(specter_checkpoint, scidocs_dir, output_dir, method_name, batch_size: int = 8):
    """

    export CUDA_VISIBLE_DEVICES=
    python cli_model.py evaluate_specter_pl "data/specter/save/version_0/checkpoints/ep-epoch=3_avg_val_loss-avg_val_loss=0.136.ckpt" \
        ${SCIDOCS_DIR} ${BASE_DIR}/data/specter/embeds --method_name specter_pl --batch_size 16

    python cli_scidocs.py evaluate specter_pl --scidocs_dir ${SCIDOCS_DIR} --embeddings_dir ${BASE_DIR}/data/specter/embeds

    :param specter_checkpoint:
    :param scidocs_dir:
    :param output_dir:
    :param method_name:
    :param batch_size:
    :return:
    """
    from specter_pl_train import Specter
    from transformers import BertTokenizerFast

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = get_env()

    model = Specter.load_from_checkpoint(specter_checkpoint)
    model = model.to(device)

    tokenizer = BertTokenizerFast.from_pretrained(env['bert_dir'] + '/scibert-scivocab-uncased')

    if not os.path.exists(output_dir):
        logger.info(f'Create: {output_dir}')
        os.makedirs(output_dir)

    scidocs_metadata = get_scidocs_metadata(scidocs_dir)

    ds_docs = {}
    ds_inputs = {}

    for ds, ds_metadata in scidocs_metadata.items():
        ds_docs[ds] = []

        for i, (paper_id, d) in enumerate(ds_metadata.items()):
            ds_docs[ds].append(
                d['title'] + tokenizer.sep_token + (d.get('abstract') or '')
            )

        logger.info(f'Tokenize {ds}')
        ds_inputs[ds] = tokenizer(ds_docs[ds], padding=True, truncation=True, return_tensors="pt", max_length=512)

        logger.info(f'Predict {ds}')

        dl = DataLoader(DictOfListsDataset(ds_inputs[ds]), batch_size=batch_size, shuffle=False)
        embeds = []

        for batch_inputs in tqdm(dl, total=len(dl)):
            batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
            model_out = model(**batch_inputs)
            embeds += model_out.tolist()

        if len(embeds) != len(ds_metadata):
            raise ValueError(f'Invalid embeddings count: {len(embeds)} vs {len(ds_metadata)}')

        # Write to disk
        out_fp = os.path.join(output_dir, f'{ds}__{method_name}.jsonl')

        logger.info(f'Write to {out_fp}')

        with open(out_fp, 'w') as f:
            for idx, (paper_id, metadata) in enumerate(ds_metadata.items()):
                f.write(json.dumps({
                    'paper_id': metadata['paper_id'],
                    'title': metadata['title'],
                    'embedding': embeds[idx],
                }) + '\n')

    logger.info('done. now you can run the evaluate command.')


if __name__ == '__main__':
    fire.Fire()
    sys.exit(0)

