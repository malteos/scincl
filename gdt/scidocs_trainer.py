import json
import logging
import os
import shutil
import uuid
from datetime import datetime
from typing import Optional, List, Dict

import numpy as np
import torch
from scidocs import get_mag_mesh_metrics, get_view_cite_read_metrics, get_recomm_metrics
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import IntervalStrategy

from gdt.datasets.scidocs import SciDocsDataset
from gdt.triple_trainer import TriplesTrainer
from gdt.utils import flatten

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

SCIDOCS_MAG_LABELS = [
    'Art', 'Biology', 'Business', 'Chemistry', 'Computer science', 'Economics', 'Engineering', 'Environmental science',
    'Geography', 'Geology', 'History', 'Materials science', 'Mathematics', 'Medicine', 'Philosophy', 'Physics',
    'Political science', 'Psychology', 'Sociology']

SCIDOCS_MESH_LABELS = [
    'Cardiovascular diseases', 'Chronic kidney disease', 'Chronic respiratory diseases', 'Diabetes mellitus',
    'Digestive diseases', 'HIV/AIDS', 'Hepatitis A/B/C/E', 'Mental disorders', 'Musculoskeletal disorders',
    'Neoplasms (cancer)', 'Neurological disorders'
]


class SciDocsTrainer(TriplesTrainer):
# class SciDocsTrainer(Trainer):
    scidocs_cuda_device = -1
    scidocs_ds: SciDocsDataset = None
    val_or_test_or_both: str = 'both'
    workers: int = 10

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init :obj:`compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed. It must implement the
                :obj:`__len__` method.
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (:obj:`str`, `optional`, defaults to :obj:`"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """

        if not self.args.do_eval:
            logger.info('Skipping evaluate (`args.do_eval` is False)')
            return {}

        if not (self.args.evaluation_strategy == IntervalStrategy.EPOCH and (round(self.state.epoch, 2) % self.args.eval_steps) == 0):
            logger.info(f'Skipping evaluate (current epoch={self.state.epoch} ({round(self.state.epoch, 2)}) is not in eval_steps={self.args.eval_steps})')
            return {}

        if self.scidocs_ds:
            test_dataset = self.scidocs_ds
        else:
            raise ValueError('scidocs_ds is not set!')

        test_dataloader = self.get_test_dataloader(test_dataset)

        embeddings_dir = os.path.join(test_dataset.scidocs_dir, 'embeddings')

        if not os.path.exists(embeddings_dir):
            logger.info(f'Creating embeddings dir: {embeddings_dir}')
            os.makedirs(embeddings_dir)

        model = self.model

        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        model.eval()

        embeddings = []

        logger.info('Predict on test set')

        with torch.no_grad():
            for step, inputs in enumerate(tqdm(test_dataloader, desc='Predict')):
                inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
                model_out = model(**inputs)

                step_embeds = model_out.cpu().numpy()

                embeddings += step_embeds.tolist()

        embeddings = np.array(embeddings)

        from scidocs.paths import DataPaths

        logger.info(f'Evaluating SciDocs')

        now = datetime.now()  # current date and time
        now_str = now.strftime("%Y-%m-%d_%H%M%S")
        uuid_str = str(uuid.uuid1())

        method = 'train_' + now_str + '_' + uuid_str

        logger.info(f'Run name: {method}')

        # labels_ids = pred.label_ids
        # embeddings = pred_out.predictions
        paper_ids = test_dataset.paper_ids

        paper_id_to_idx = {paper_id: idx for idx, paper_id in enumerate(paper_ids)}

        logger.info(f'Paper IDs loaded: {len(paper_id_to_idx):,}')
        logger.info(f'Embeddings loaded: {embeddings.shape}')

        # write to disk
        total_missed = 0

        for ds, ds_metadata in test_dataset.scidocs_metadata.items():
            out_fp = os.path.join(embeddings_dir, f'{ds}__{method}.jsonl')
            missed = []

            logger.info(f'Writing SciDocs files for {ds} in {out_fp}')

            with open(out_fp, 'w') as f:
                for paper_id, metadata in ds_metadata.items():
                    if paper_id in paper_id_to_idx:
                        f.write(json.dumps({
                            'paper_id': metadata['paper_id'],
                            'title': metadata['title'],
                            'embedding': embeddings[paper_id_to_idx[paper_id], :].tolist(),
                        }) + '\n')
                    else:
                        missed.append(paper_id)

            logger.info(f'Missing embbedings for {len(missed):,} papers')
            if len(missed) > 0:
                logger.warning(f'Missed IDs: {missed[:10]}')

            total_missed += len(missed)

        # point to the data, which should be in scidocs/data by default
        data_paths = DataPaths(test_dataset.scidocs_dir)

        logger.info(f'Starting SciDocs eval (workers={self.workers}; val_or_test={self.val_or_test_or_both}; cuda_device={self.scidocs_cuda_device})')

        eval_sets = ['test', 'val'] if self.val_or_test_or_both == 'both' else [self.val_or_test_or_both]
        eval_metrics = {
            'eval_method_name': method,
            'eval_missing_count': total_missed,
        }

        for eval_set in eval_sets:
            # temp dirs
            temp_dir = os.path.join(embeddings_dir, f'tmp_{method}_{eval_set}')
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)

            logger.info(f'Evaluate on: {eval_set} in {temp_dir}')

            # now run the evaluation
            scidocs_metrics = {}

            scidocs_metrics.update(
                get_mag_mesh_metrics(data_paths, os.path.join(embeddings_dir, f'mag_mesh__{method}.jsonl'), val_or_test=eval_set, n_jobs=self.workers))

            scidocs_metrics.update(get_view_cite_read_metrics(data_paths, os.path.join(embeddings_dir, f'view_cite_read__{method}.jsonl'),
                                                              val_or_test=eval_set, run_path=os.path.join(temp_dir, 'temp.run')))
            scidocs_metrics.update(
                get_recomm_metrics(data_paths, os.path.join(embeddings_dir, f'recomm__{method}.jsonl'), val_or_test=eval_set, cuda_device=self.scidocs_cuda_device, serialization_dir=os.path.join(temp_dir, "recomm-tmp")))

            # Remove temp dir
            shutil.rmtree(temp_dir, ignore_errors=True)

            # Flatten dict
            scidocs_metrics = flatten(scidocs_metrics)

            # Compute avg
            scidocs_metrics['avg'] = np.mean([v for k, v in scidocs_metrics.items()])

            # "eval_" prefix before logging
            eval_metrics.update({'eval_' + eval_set + '_' + k: v for k, v in scidocs_metrics.items()})

            logger.info(f'Results ({eval_set}): {scidocs_metrics}')

        # Clean up
        for ds, ds_metadata in test_dataset.scidocs_metadata.items():
            out_fp = os.path.join(embeddings_dir, f'{ds}__{method}.jsonl')

            logger.info(f'Deleting {out_fp}')
            os.remove(out_fp)

        self.log(eval_metrics)

        return eval_metrics

