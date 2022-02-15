import logging
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.bert import BertPreTrainedModel, BertModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead

from gdt.models import PoolingStrategy

logger = logging.getLogger(__name__)


class BertForPairwiseCosineSimilarityLoss(BertPreTrainedModel):
    def _reorder_cache(self, past, beam_idx):
        pass

    def __init__(self, config, loss_func=None,
                 masked_language_modeling: bool = False,
                 pooling_strategy: PoolingStrategy = PoolingStrategy.CLS
                 ):
        super().__init__(config)

        self.masked_language_modeling = masked_language_modeling
        self.pooling_strategy = pooling_strategy

        logger.info(f'Mask language modeling: {self.masked_language_modeling}')
        logger.info(f'Pooling strategy: {self.pooling_strategy}')

        self.bert = BertModel(config)

        if self.masked_language_modeling:
            self.cls = BertOnlyMLMHead(config)
        else:
            self.cls = None

        if loss_func:
            self.loss_func = loss_func
        else:
            # default
            self.loss_func = nn.MSELoss()
            # L1Loss, SmoothL1Loss

        # self.masked_lm_loss

    def get_pooled_embedding(self, bert_out, strategy: Optional[PoolingStrategy] = None):
        """
        Reduce BERT output to a single document embedding depending on pooling strategy.

        :param strategy:
        :param bert_out: [batch_size, seq_length, hidden_size]
        :return: [batch_size, hidden_size]
        """

        if strategy is None:
            # use class property by default
            strategy = self.pooling_strategy

        if strategy == PoolingStrategy.CLS:
            return bert_out.last_hidden_state[:, 0, :]  # [CLS] token
        elif strategy == PoolingStrategy.MEAN:
            #TODO exclude pad tokens with attention_mask!!!
            return torch.mean(bert_out.last_hidden_state[:, :, :], dim=1)  # mean of all last hidden states
        elif strategy == PoolingStrategy.CONCAT:
            return torch.cat((
                self.get_pooled_embedding(bert_out, PoolingStrategy.CLS),
                self.get_pooled_embedding(bert_out, PoolingStrategy.MEAN)
            ), dim=1)
        else:
            raise ValueError(f'Invalid PoolingStrategy: {strategy}')

    @staticmethod
    def cosine_pairwise(x: torch.Tensor):
        # works as sklearn.metrics.pairwise.cosine_similarity
        # See https://github.com/pytorch/pytorch/issues/11202
        x = x.unsqueeze(0).permute((1, 2, 0))
        cos_sim_pairwise = F.cosine_similarity(x, x.unsqueeze(1), dim=-2)
        cos_sim_pairwise = cos_sim_pairwise.permute((2, 0, 1))
        return cos_sim_pairwise.squeeze()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        """
        Loss is the MSE of the cosine similarity between the language model embeddings and the target embeddings (citation graph embeddings).

        :param input_ids: See BERT
        :param attention_mask:
        :param token_type_ids:
        :param labels: Target embeddings (batch_size, target_embedding_dim)  (target embeddings == graph embeddings)
        :return:
        """

        bert_embeds = self.get_pooled_embedding(self.bert(input_ids, attention_mask, token_type_ids))

        if labels is not None:
            target_embeds = labels

            target_cosine = self.cosine_pairwise(target_embeds)
            bert_cosine = self.cosine_pairwise(bert_embeds)

            # diagonal + lower left triangle are irrelevant for loss (= 1 or duplicate)
            # -> set to zero
            target_cosine = torch.triu(target_cosine, diagonal=1)
            bert_cosine = torch.triu(bert_cosine, diagonal=1)

            loss = self.loss_func(bert_cosine, target_cosine)

            return loss, bert_embeds

        else:
            # only inference
            return bert_embeds

