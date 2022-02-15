import json
import logging
import os
from typing import Dict

import torch
from smart_open import open
from torch.utils.data import Dataset

from gdt.utils import get_graph_embeddings

logger = logging.getLogger(__name__)


class TripleDataset(Dataset):
    """
    PyTorch dataset for triple training
    """
    def __init__(self,
                 triples_csv_path: str,
                 metadata_jsonl_path: str,
                 tokenizer,
                 sample_n: int = 0,
                 mask_anchor_tokens: bool = False,
                 predict_embeddings: bool = False,
                 abstract_only: bool = False,
                 use_cache: bool = False,
                 max_sequence_length: int = 512,
                 mlm_probability: float = 0.15,
                 graph_embeddings_path: str = None,
                 graph_paper_ids_path: str = None):
        """
        Initialize dataset with settings (does not load data)

        :param triples_csv_path: CSV with: anchor_id, pos_id, neg_id
        :param metadata_jsonl_path: JSONL with: paper_id, title, abstract
        :param tokenizer: Huggingface tokenizer
        :param sample_n: Random sample of n triples
        :param abstract_only: Use abstract for document text (exclude title)
        """
        self.predict_embeddings = predict_embeddings
        self.use_cache = use_cache
        self.triples_csv_path = triples_csv_path
        self.metadata_jsonl_path = metadata_jsonl_path
        self.tokenizer = tokenizer
        self.sample_n = sample_n
        self.abstract_only = abstract_only  # for wikipedia
        self.mask_anchor_tokens = mask_anchor_tokens
        self.max_sequence_length = max_sequence_length
        self.mlm_probability = mlm_probability
        self.graph_embeddings_path = graph_embeddings_path
        self.graph_paper_ids_path = graph_paper_ids_path

        self.paper_id_to_metadata = {}
        # self.positive_inputs = None
        # self.negative_inputs = None
        # self.anchor_inputs = None
        self.anchor_ids = []
        self.pos_ids = []
        self.neg_ids = []

        self.paper_id_to_inputs = {}

        # Check for gzipped input
        if not os.path.exists(self.metadata_jsonl_path) and os.path.exists(self.metadata_jsonl_path + '.gz'):
            self.metadata_jsonl_path = self.metadata_jsonl_path + '.gz'

        if not os.path.exists(self.triples_csv_path) and os.path.exists(self.triples_csv_path + '.gz'):
            self.triples_csv_path = self.triples_csv_path + '.gz'

        tokenizer_name = self.tokenizer.name_or_path.split('/')[-1]

        # Cache path depends on settings
        self.paper_id_to_inputs_path = self.metadata_jsonl_path + f'.{tokenizer_name}.{self.max_sequence_length}'

        if self.mask_anchor_tokens:
            self.paper_id_to_inputs_path += '.mlm'

        if self.predict_embeddings:
            self.paper_id_to_inputs_path += '.predict_embeddings'

        self.paper_id_to_inputs_path += '.cache'

    def get_texts_from_ids(self, paper_ids):
        if self.abstract_only:
            # Wikipedia: Only "first sentence" = abstract (if null -> title)
            return [
                (self.paper_id_to_metadata[pid]['abstract'] or self.paper_id_to_metadata[pid]['title'])
                for pid in paper_ids
            ]
        else:
            # SPECTER the title and abstract of a paper, separated by the [SEP] token.
            # See https://github.com/allenai/specter#1--through-huggingface-transformers-library
            return [
                self.paper_id_to_metadata[pid]['title']
                + self.tokenizer.sep_token
                + (self.paper_id_to_metadata[pid]['abstract'] or '')
                for pid in paper_ids
            ]

    def get_inputs_from_id(self, paper_id):
        """
        Return tokenizer output (input_ids, attention_mask, ...)

        :param paper_id:
        :return:
        """
        return self.paper_id_to_inputs[paper_id]

    def mask_tokens(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Prepare masked tokens input_ids/labels for masked language modeling: 80% MASK, 10% random, 10% original.

        Based on Huggingface: transformers.data.data_collator.DataCollatorForLanguageModeling#mask_tokens
        """

        inputs = inputs.copy()  # avoid changing inputs in cache

        input_ids = inputs['input_ids'].view(1, -1)  # input ids in batch

        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        # back to single item form
        inputs.update({
            'input_ids': input_ids.squeeze(),
            'labels': labels.squeeze(),
        })

        return inputs

    def load(self):
        # Metadata
        logger.info(f'Reading from: {self.metadata_jsonl_path}')
        self.paper_id_to_metadata = {}

        with open(self.metadata_jsonl_path) as f:
            for line in f:
                metadata = json.loads(line)
                self.paper_id_to_metadata[metadata['paper_id']] = metadata

        # Tokenized papers
        if self.use_cache and os.path.exists(self.paper_id_to_inputs_path):
            logger.info(f'Loading cache from: {self.paper_id_to_inputs_path}')

            with open(self.paper_id_to_inputs_path, 'rb') as f:
                self.paper_id_to_inputs = torch.load(f)
        else:
            logger.info('Cache not requested or cache file does not exist')
            self.paper_id_to_inputs = {}

        # Triples
        logger.info(f'Reading from: {self.triples_csv_path}')

        self.anchor_ids = []
        self.pos_ids = []
        self.neg_ids = []
        no_metadata = []

        with open(self.triples_csv_path) as f:
            for i, line in enumerate(f):
                if i > 0:  # Skip header row
                    col = line.strip().split(',')
                    anchor_id, pos_id, neg_id = col

                    if anchor_id in self.paper_id_to_metadata \
                            and pos_id in self.paper_id_to_metadata \
                            and neg_id in self.paper_id_to_metadata:

                        self.anchor_ids.append(anchor_id)
                        self.pos_ids.append(pos_id)
                        self.neg_ids.append(neg_id)

                        if self.sample_n > 0 and len(self.anchor_ids) > self.sample_n:
                            logger.info(f'Stop at {i} ...')
                            break
                    else:
                        no_metadata.append(col)

        logger.info(f'Loaded {len(self.anchor_ids):,} triples')

        if len(no_metadata) > 0:
            logger.warning(f'Triples without metadata: {len(no_metadata):,}')
            logger.warning(f'Triples without metadata: {no_metadata[:3]}')

        # Tokenize each paper only once!
        paper_ids = set(self.anchor_ids + self.pos_ids + self.neg_ids)

        logger.info(f'Triples with unique papers: {len(paper_ids):,}')
        logger.info(f'Papers in cache: {len(self.paper_id_to_inputs):,}')

        tokenize_paper_ids = list(paper_ids - set(self.paper_id_to_inputs.keys()))

        logger.info(f'Tokenize papers: {len(tokenize_paper_ids):,}')

        if len(tokenize_paper_ids) > 0:
            tokenizer_out = self.tokenizer(
                text=self.get_texts_from_ids(tokenize_paper_ids),
                # text_pair=section_titles,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors='pt',
                padding='max_length',
                max_length=self.max_sequence_length,
                truncation=True,
                return_token_type_ids=False
            )

            # Store in index
            for idx, paper_id in enumerate(tokenize_paper_ids):
                self.paper_id_to_inputs[paper_id] = {k: v[idx] for k, v in tokenizer_out.items()}

            del tokenizer_out

            if self.predict_embeddings:
                # Graph embeddings
                logger.info('Loading data for embedding prediction')

                # Read ID mapping
                with open(self.graph_paper_ids_path) as f:
                    graph_paper_ids = json.load(f)

                # Load from disk and convert to torch tensor (float 32 is required to sum loss with triplet loss)
                graph_embeddings = torch.tensor(
                    get_graph_embeddings(self.graph_embeddings_path, do_normalize=False, workers=1, paper_ids=graph_paper_ids, include_paper_ids=tokenize_paper_ids),
                    dtype=torch.float32
                )

                # Store in index
                for idx, paper_id in enumerate(tokenize_paper_ids):
                    self.paper_id_to_inputs[paper_id]['target_embedding'] = graph_embeddings[idx]

                del graph_embeddings

        # Write to cache if enabled and new papers tokenized
        if self.use_cache and len(tokenize_paper_ids) > 0:
            logger.info(f'Saving cache to {self.paper_id_to_inputs_path}')

            with open(self.paper_id_to_inputs_path, 'wb') as f:
                torch.save(self.paper_id_to_inputs, f)

        logger.info(f'Dataset loaded with {self.__len__():,} samples')

    def __getitem__(self, idx):
        anchor_inputs = self.get_inputs_from_id(self.anchor_ids[idx])

        if self.mask_anchor_tokens:
            # Mask language modeling for anchor tokens
            anchor_inputs = self.mask_tokens(anchor_inputs)

        item = {'anchor_' + k: v for k, v in anchor_inputs.items()}
        item.update({'positive_' + k: v for k, v in self.get_inputs_from_id(self.pos_ids[idx]).items()})
        item.update({'negative_' + k: v for k, v in self.get_inputs_from_id(self.neg_ids[idx]).items()})

        return item

    def __len__(self):
        return len(self.anchor_ids)

    def get_stats(self, prefix: str = 'dataset_') -> Dict[str, int]:
        """
        Returns basic statistics of dataset

        :param prefix:
        :return:
        """
        return {
            prefix + 'count': len(self),
            prefix + 'anchor_count': len(self.anchor_ids),
            prefix + 'anchor_unique_count': len(set(self.anchor_ids)),
            prefix + 'positive_count': len(self.pos_ids),
            prefix + 'positive_unique_count': len(set(self.pos_ids)),
            prefix + 'negative_count': len(self.neg_ids),
            prefix + 'negative_unique_count': len(set(self.neg_ids)),
        }


