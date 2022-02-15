import collections
import json
import logging
import multiprocessing
import os
from collections import defaultdict
from datetime import datetime
from math import ceil
from multiprocessing.pool import Pool
from typing import List, Dict, Optional, Union

import h5py
import numpy as np
from dataclasses import fields
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def get_workers(workers: int = 0) -> int:
    if workers is not None and workers > 0:
        return workers
    else:
        # TODO based on env
        logger.info(f'Workers same as CPU cores: {multiprocessing.cpu_count()}')

        return multiprocessing.cpu_count()


def flatten(d, parent_key='', sep='__'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten(dictionary, sep='__'):
    out_dict = dict()
    for key, value in dictionary.items():
        parts = key.split(sep)
        d = out_dict
        for part in parts[:-1]:
            if part not in d:
                d[part] = dict()
            d = d[part]
        d[parts[-1]] = value
    return out_dict


def get_scidocs_compute_metrics(test_ds, workers: int = 0, cuda_device: int = -1, val_or_test='test'):
    """
    Returns compute_metrics function for Huggingface Trainer

    :param val_or_test:
    :param cuda_device:
    :param test_ds: SciDocsDataset
    :param workers:
    :return:
    """
    from scidocs import get_scidocs_metrics
    from scidocs.paths import DataPaths

    def compute_metrics(pred_out):
        logger.info('Evaluating SciDocs')

        embeddings_dir = os.path.join(test_ds.scidocs_dir, 'embeddings')

        now = datetime.now()  # current date and time
        now_str = now.strftime("%Y-%m-%d_%H%M%S")

        method = 'train_' + now_str

        logger.info(f'Run name: {method}')

        # labels_ids = pred.label_ids
        embeddings = pred_out.predictions
        paper_ids = test_ds.paper_ids

        paper_id_to_idx = {paper_id: idx for idx, paper_id in enumerate(paper_ids)}

        logger.info(f'Paper IDs loaded: {len(paper_id_to_idx):,}')
        logger.info(f'Embeddings loaded: {embeddings.shape}')

        # write to disk
        for ds, ds_metadata in test_ds.scidocs_metadata.items():
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

        # point to the data, which should be in scidocs/data by default
        data_paths = DataPaths(test_ds.scidocs_dir)

        logger.info('Starting SciDocs eval')

        # now run the evaluation
        eval_metrics = get_scidocs_metrics(
            data_paths,
            os.path.join(embeddings_dir, f'mag_mesh__{method}.jsonl'),
            os.path.join(embeddings_dir, f'view_cite_read__{method}.jsonl'),
            os.path.join(embeddings_dir, f'recomm__{method}.jsonl'),
            val_or_test=val_or_test,  # set to 'val' if tuning hyperparams
            n_jobs=workers,  # the classification tasks can be parallelized
            cuda_device=cuda_device,  # the recomm task can use a GPU if this is set to 0, 1, etc
        )

        return eval_metrics

    return compute_metrics


def get_scidocs_metadata(scidocs_dir: str):
    with open(os.path.join(scidocs_dir, 'paper_metadata_mag_mesh.json')) as f1:
        with open(os.path.join(scidocs_dir, 'paper_metadata_recomm.json')) as f2:
            with open(os.path.join(scidocs_dir, 'paper_metadata_view_cite_read.json')) as f3:
                scidocs_metadata = {
                    'mag_mesh': json.load(f1),
                    'recomm': json.load(f2),
                    'view_cite_read': json.load(f3),
                }

    logger.info(f"Loaded: {len(scidocs_metadata['mag_mesh']):,} mag_mesh, {len(scidocs_metadata['recomm']):,} recomm, {len(scidocs_metadata['view_cite_read']):,} view_cite_read")

    return scidocs_metadata


def read_json_mapping_files(mapping_file_path: str) -> Dict[str, str]:
    """
    Read ID mapping (e.g., S2ID => S2ORC paper id) from JSON file (or multiple files)

    :param mapping_file_path: Comma separated list of file paths to JSON dict files.
    :return:
    """

    # Separate multiples by comma
    mapping_file_paths = mapping_file_path.split(',')
    mapping = {}

    for fp in mapping_file_paths:
        with open(fp) as f:
            current_mapping = json.load(f)

        logger.info(f'Reading mapping from {fp} ({len(current_mapping):,} IDs)')

        mapping.update(current_mapping)

    logger.info(f'Mapping loaded: {len(mapping):,}')

    return mapping


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def split_into_n_chunks(lst, n):
    """Split list into n chunks"""
    avg = ceil(len(lst) / n)
    out = []
    last = 0.0

    # print(avg)

    while last < len(lst):
        out.append(lst[int(last):int(last + avg)])
        last += avg

    return out


def write_func_args(frame, fp):
    """
    Writes current function arguments to JSON file (for settings logging)

    Usage: write_settings(inspect.currentframe(), 'settings.json')

    :param frame:
    :param fp:
    :return:
    """

    pass
    # args, _, _, values = inspect.getargvalues(frame)
    #
    # values = {k: v.__dict__ if hasattr(v, '__dict__') else v for k, v in values.items()}
    #
    # try:
    #     with open(fp, 'w') as f:
    #         json.dump(values, f)
    # except TypeError as e:
    #     logger.error(values)
    #     raise e


def get_kwargs_for_data_classes(data_classes: List, args: Dict, allow_multiple_assignments: bool = True) -> List[Dict]:

    data_classes_args = {str(data_class): {} for data_class in data_classes}

    for k, v in args.items():
        assigned = False

        for data_class in data_classes:
            dc_id = str(data_class)
            dc_fields = {f.name for f in fields(data_class)}

            if k in dc_fields:
                data_classes_args[dc_id][k] = v
                assigned = True

                if not allow_multiple_assignments:
                    # a key-value pair can be only assign to a single data class
                    break

        if not assigned:
            raise ValueError(f'Could not assign argument "{k}={v}" to any data class (available: {data_classes})')

    return list(data_classes_args.values())


def get_data_class_args(data_classes: List, args: Dict, **kwargs):
    args.update(kwargs)

    data_classes_args = defaultdict(dict)

    for k, v in args.items():
        for data_class in data_classes:
            dc_id = str(data_class)
            dc_fields = {f.name for f in fields(data_class)}

            if k in dc_fields:
                data_classes_args[dc_id][k] = v
                break

    for data_class in data_classes:
        dc_id = str(data_class)

        try:
            dc_obj = data_class(**data_classes_args[dc_id])
            yield dc_obj
        except TypeError as e:
            logging.info(data_classes_args)
            logger.info(f'Cannot initialize data class: {data_class} {dc_id} with args: {data_classes_args[dc_id]}; initial args: {args}; kwargs: {kwargs}')
            raise e


def normalize_in_parallel(x: np.ndarray, workers: int, chunk_size: int = 10_000) -> np.ndarray:
    chunks_n = int(len(x) / chunk_size)
    raise ValueError('you cant parallize this')

    with Pool(workers) as pool:
        return np.concatenate(list(tqdm(pool.imap_unordered(normalize, chunks(x, chunk_size)), total=chunks_n)), axis=0)


def get_graph_embeddings(graph_embeddings_path: str, do_normalize: bool = False, workers: int = 10,
                         paper_ids: Optional[Union[str, List[str]]] = None,
                         include_paper_ids: Optional[Union[List[str], str]] = None,
                         placeholder: bool = False):
    """
    Load graph embeddings from H5PY file

    :param placeholder: If enabled only a placeholder ndarray is returned with (1, embedding_size)
    :param graph_embeddings_path: Path to H5PY file
    :param do_normalize: Normalize embeddings (for cosine similarity)
    :param workers: Normalize with n workers
    :param paper_ids: List or path JSON with paper IDs corresponding to indices in embedding matrix
    :param include_paper_ids: List or path JSON with paper IDs to be included (if not set, all papers are used)
    :return:
    """
    logger.info(f'Loading embeddings from: {graph_embeddings_path}')

    # reading all embeddings at once is quicker than loading with indices
    with h5py.File(graph_embeddings_path, "r") as hf:
        if placeholder:
            graph_embeddings = hf["embeddings"][:1, :]  # placeholder (only the first entry)

            logger.warning('Returning only first entry as placeholder for embeddings')
        else:
            graph_embeddings = hf["embeddings"][:, :]

    logger.info(f'Graph embeddings: {graph_embeddings.shape}')

    # Filter embeddings based on paper IDs
    if isinstance(paper_ids, str) and isinstance(include_paper_ids, str):
        with open(paper_ids) as f:
            paper_ids = json.load(f)

        with open(include_paper_ids) as f:
            include_paper_ids = json.load(f)

    if isinstance(paper_ids, list) and isinstance(include_paper_ids, list):
        logger.info(f'Include papers: {len(include_paper_ids):,}')

        paper_id_to_idx = {pid: idx for idx, pid in enumerate(paper_ids)}
        include_paper_idxs = np.array([paper_id_to_idx[pid] for pid in include_paper_ids])

        graph_embeddings = graph_embeddings[include_paper_idxs]

        logger.info(f'Filtered embeddings: {graph_embeddings.shape}')

    # normalize vectors for cosine similarity
    if do_normalize:
        logger.info('Normalize embeddings')
        graph_embeddings = normalize_in_parallel(graph_embeddings, workers)

    return graph_embeddings


def get_auto_train_batch_size(model_name_or_path: str, default_size: int = 8, mask_language_modeling: bool = False) -> int:
    """
    Determine batch size that fits into avialable GPU memory
    """
    import torch
    gpu_mem = int(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024 * 1024))

    logger.info(f'GPU memory available: {gpu_mem}+1 GB')

    if mask_language_modeling:
        # elif gpu_mem == 23:
        #   return 6
        raise NotImplementedError()

    model_name = model_name_or_path.split('/')[-1]

    if model_name == 'deberta-base':
        raise NotImplementedError()
    elif model_name == 'deberta-v2-xlarge':
        raise NotImplementedError()
    else:
        # default size:
        # bert-base-uncased, bert-base-cased, scibert-scivocab-uncased
        if gpu_mem == 11 or gpu_mem == 10:  # PARTITION11
            return 3
        elif gpu_mem == 15:  # PARTITION16
            return 6
        elif gpu_mem == 23:
            return 8
        elif gpu_mem == 31:
            return 10  # 14
        elif gpu_mem == 39:
            return 14
        elif gpu_mem == 47:
            return 16
        else:
            return default_size


class ListOfDictsDataset(Dataset):
    def __init__(self, samples: List[Dict]):
        self.samples = samples

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)


class DictOfListsDataset(Dataset):
    def __init__(self, samples: Dict[str, List]):
        self.samples = samples

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.samples.items()}

    def __len__(self):
        k = list(self.samples.keys())[0]  # first key

        return len(self.samples[k])