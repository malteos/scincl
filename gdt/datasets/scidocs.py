import logging
import os

import torch
from torch.utils.data import Dataset

from gdt.utils import get_scidocs_metadata

logger = logging.getLogger(__name__)


class SciDocsDataset(Dataset):
    """
    PyTorch dataset for SciDocs evaluation
    """
    def __init__(self, scidocs_dir, tokenizer, sample_n=0, use_cache: bool = False, max_length: int = 512,
                 inference_prefix: str = 'anchor_'):
        self.scidocs_dir = scidocs_dir
        self.tokenizer = tokenizer
        self.scidocs_metadata = {}
        self.paper_ids = []
        self.anchor_inputs = None
        self.sample_n = sample_n
        self.use_cache = use_cache

        # get tokenizer name
        tokenizer_name = self.tokenizer.name_or_path.split('/')[-1]

        self.cache_path = os.path.join(scidocs_dir, f'scidocs_dataset.{tokenizer_name}.cache.pt')
        self.max_length = max_length
        self.inference_prefix = inference_prefix

    def load(self):

        # Metadata
        logger.info(f'Metadata from {self.scidocs_dir}')

        self.scidocs_metadata = get_scidocs_metadata(self.scidocs_dir)
        paper_id_to_text = {}
        paper_id_to_title = {}

        for ds, ds_metadata in self.scidocs_metadata.items():
            for i, (paper_id, metadata) in enumerate(ds_metadata.items()):
                if paper_id not in paper_id_to_text:
                    # SPECTER the title and abstract of a paper, separated by the [SEP] token.
                    # See https://github.com/allenai/specter#1--through-huggingface-transformers-library
                    paper_id_to_text[paper_id] = metadata['title'] + self.tokenizer.sep_token + (
                                metadata['abstract'] or '')
                    paper_id_to_title[paper_id] = metadata['title']

                    if self.sample_n > 0 and i > self.sample_n:
                        logger.warning(f'Stop sample_n={self.sample_n}')
                        break

        self.paper_ids = list(paper_id_to_text.keys())
        self.titles = list(paper_id_to_title.values())
        texts = list(paper_id_to_text.values())

        """
        if self.sample_n > 0:
            texts = texts[:self.sample_n]
            self.paper_ids = self.paper_ids[:self.sample_n]
            self.titles = self.titles[:self.sample_n]
        """

        if self.use_cache and os.path.exists(self.cache_path):
            logger.info(f'Loading cache from {self.cache_path}')
            with open(self.cache_path, 'rb') as f:
                self.anchor_inputs = torch.load(f)
        else:
            logger.info(f'Tokenize {len(texts):,}')

            self.anchor_inputs = self.tokenizer(
                text=texts,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors='pt',
                padding='max_length',
                max_length=self.max_length,
                truncation=True,
                return_token_type_ids=False
            )

            if self.use_cache:
                logger.info(f'Writing cache to {self.cache_path}')
                with open(self.cache_path, 'wb') as f:
                    torch.save(self.anchor_inputs, f)

        logger.info(f'Dataset loaded with {self.__len__():,} samples')

    def save_embeddings(self):
        pass

    def get_embed(self, model_out):
        return model_out.last_hidden_state[:, 0, :]  # [CLS] token

    def __getitem__(self, idx):
        item = {self.inference_prefix + k: v[idx] for k, v in self.anchor_inputs.items()}

        return item

    def __len__(self):
        return len(self.anchor_inputs['input_ids'])