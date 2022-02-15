import json
import logging
import os
import random
from multiprocessing import Pool
from typing import Union, List, Dict

import torch
from torch.utils.data import Dataset

from cli_triples import worker_extract_metadata
from gdt.utils import get_graph_embeddings, split_into_n_chunks

logger = logging.getLogger(__name__)


class TargetEmbeddingDataset(Dataset):
    """
    PyTorch dataset for pairwise cosine training through target embeddings (graph embeddings)
    """
    max_length = 512

    def __init__(self,
                 train_ids: Union[List[str], str],
                 s2orc_metadata_dir: str,
                 tokenizer,
                 graph_paper_ids_path: str,
                 graph_embeddings_path: str,
                 workers: int = 10,
                 sample_n: int = 0,
                 sample_ratio: float = .0,
                 ):

        self.tokenizer = tokenizer
        self.workers = workers
        self.s2orc_metadata_dir = s2orc_metadata_dir

        if isinstance(train_ids, str):
            # Load from disk as JSON
            with open(train_ids) as f:
                train_ids = json.load(f)

        logger.info(f'Train IDs: {len(train_ids)}')

        if sample_ratio > 0:
            logger.info(f'Sample ratio to {sample_ratio}')
            sample_n = int(len(train_ids) * sample_ratio)

        if sample_n > 0:
            logger.info(f'Sample to {sample_n}')

            train_ids = random.sample(train_ids, sample_n)

        # Graph embeddings
        with open(graph_paper_ids_path) as f:
            paper_ids = json.load(f)

        self.graph_embeddings = torch.tensor(
            get_graph_embeddings(graph_embeddings_path, False, workers, paper_ids, train_ids))

        logger.info(f'Graph embeddings: {self.graph_embeddings.shape}')

        metadata = self.load_metadata(train_ids, workers)

        logger.info(f'Metadata: {len(metadata)}')

        # Metadata to texts
        texts = [paper['title']
                 + self.tokenizer.sep_token
                 + (paper['abstract'] or '') for paper in metadata]

        self.tokenizer_out = self.tokenizer(
            text=texts,
            # text_pair=section_titles,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors='pt',
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_token_type_ids=False
        )

        logger.info('Tokenizer done')

        self.train_ids = train_ids

    def load_metadata(self, needed_paper_ids, workers):

        if not isinstance(needed_paper_ids, set):
            needed_paper_ids = set(needed_paper_ids)

        logger.info(f'Needed metadata for {len(needed_paper_ids):,}')

        # Meta data files
        batch_fns = [batch_fn for batch_fn in os.listdir(self.s2orc_metadata_dir) if batch_fn.endswith('.jsonl.gz')]
        logger.info(f'Files available: {len(batch_fns):,}')

        logger.info(f'Extracting metadata with workers: {workers}')

        # worker_id, batch_fns, needed_paper_ids, s2orc_metadata_dir
        worker_data = zip(
            list(range(workers)),  # worker ids
            split_into_n_chunks(batch_fns, workers),
            # static arguments (same for all workers)
            [needed_paper_ids] * workers,
            [self.s2orc_metadata_dir] * workers,
        )

        # Run threads
        with Pool(workers) as pool:
            pool_outputs = list(pool.starmap(worker_extract_metadata, worker_data))
            # pool_outputs = list(tqdm(pool.imap_unordered(worker_extract_metadata, batch_fns), total=len(batch_fns)))

        # Merge thread outputs
        train_metadata = [i for b in pool_outputs for i in b]

        return train_metadata

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.tokenizer_out.items()}

        if self.graph_embeddings is not None:
            item['labels'] = self.graph_embeddings[idx]

        return item

    def __len__(self):
        return len(self.train_ids)

    def get_stats(self, prefix: str = 'dataset_') -> Dict[str, int]:
        """
        Returns basic statistics of dataset

        :param prefix:
        :return:
        """
        return {
            prefix + 'count': len(self),
        }