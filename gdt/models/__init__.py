import enum
import logging

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

logger = logging.getLogger(__name__)


class PoolingStrategy(str, enum.Enum):
    MEAN = 'mean'  # Mean of hidden states of last layer
    CLS = 'cls'  # Hidden state of CLS token
    CONCAT = 'concat'  # Concatenate d=[CLS;MEAN]


def get_pooled_embedding(model_out, strategy: PoolingStrategy):
    """
    Reduce BERT output to a single document embedding depending on pooling strategy.

    :param bert_out: [batch_size, seq_length, hidden_size]
    :return: [batch_size, hidden_size]
    """

    if strategy == PoolingStrategy.CLS:
        return model_out.last_hidden_state[:, 0, :]  # [CLS] token
    elif strategy == PoolingStrategy.MEAN:
        #TODO exclude pad tokens with attention_mask!!!
        # see for example https://github.com/lucidrains/RETRO-pytorch/blob/main/retro_pytorch/retrieval.py#L196
        return torch.mean(model_out.last_hidden_state[:, :, :], dim=1)  # mean of all last hidden states
    elif strategy == PoolingStrategy.CONCAT:
        return torch.cat((
            get_pooled_embedding(model_out, PoolingStrategy.CLS),
            get_pooled_embedding(model_out, PoolingStrategy.MEAN)
        ), dim=1)
    else:
        raise ValueError(f'Invalid PoolingStrategy: {strategy}')


def get_loss_func(loss_func=None):
    if loss_func is None:
        return nn.TripletMarginLoss(margin=1.0, p=2)   # SPECTER: margin = 1; L2 norm distance => p = 2
    else:
        return loss_func


def get_mlm(has_mlm: bool, mlm_weight: float, mlm_head):
    logger.info(f'MLM: {has_mlm}')

    if has_mlm:
        if mlm_head is None:
            raise ValueError('The selected base model does not support MLM heads (probably Roberta)')
        return True, mlm_weight, mlm_head
    else:
        return False, None, None


def get_predict_embeddings_and_loss_func(predict_embeddings):
    logger.info(f'Predict embeddings: {predict_embeddings}')

    if predict_embeddings:
        return True, nn.MSELoss()  # MSE by default
    else:
        return False, None


def triplet_forward(cls, base_model, **inputs):
    """
    Generic forward function for diverse base models

    :param cls: Instance of XforTripletLoss (e.g., BERTForTripletLoss, ...)
    :param base_model: Instance of language model (e.g., BERTModel)
    :param inputs:
    :return:
    """
    # TODO pos + neg as labels?
    log_round = 4

    # split input into triple based on key prefix
    anchor_inputs = {k.replace('anchor_', ''): v for k, v in inputs.items() if k.startswith('anchor_') and 'target_embedding' not in k}
    positive_inputs = {k.replace('positive_', ''): v for k, v in inputs.items() if k.startswith('positive_') and 'target_embedding' not in k}
    negative_inputs = {k.replace('negative_', ''): v for k, v in inputs.items() if k.startswith('negative_') and 'target_embedding' not in k}

    anchor_out = base_model(**{k: v for k, v in anchor_inputs.items() if k != 'labels'})
    anchor_embed = get_pooled_embedding(anchor_out, cls.pooling_strategy)

    if len(positive_inputs) > 0 and len(negative_inputs) > 0:
        positive_out = base_model(**positive_inputs)
        negative_out = base_model(**negative_inputs)

        positive_embed = get_pooled_embedding(positive_out, cls.pooling_strategy)
        negative_embed = get_pooled_embedding(negative_out, cls.pooling_strategy)

        triplet_loss = cls.loss_func(anchor_embed, positive_embed, negative_embed)
        loss = triplet_loss

        # Mask language modeling?
        # See
        # - DeCLUTR https://github.com/JohnGiorgi/DeCLUTR/blob/bb1e6c7cc03d7ecf139c9ed30b97174dcf863953/declutr/model.py#L162
        # - BertForMaskedLM -> DataCollatorForLanguageModeling
        if cls.masked_language_modeling:
            labels = anchor_inputs['labels']
            sequence_output = anchor_out[0]
            prediction_scores = cls.cls(sequence_output)

            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            mlm_loss = loss_fct(prediction_scores.view(-1, cls.config.vocab_size), labels.view(-1))

            if hasattr(cls, 'extra_logs'):  # Log losses separately (see TriplesTrainer)
                cls.extra_logs = {
                    'triplet_loss':  round(triplet_loss.item(), log_round),
                    'mlm_loss': round(mlm_loss.item(), log_round),
                    'weighted_mlm_loss': cls.masked_language_modeling_weight * round(mlm_loss.item(), log_round),
                }
            loss = triplet_loss + cls.masked_language_modeling_weight * mlm_loss

        if cls.predict_embeddings:
            # Loss from direct prediction of target embeddings (graph embeddings)
            anchor_loss = cls.predict_embeddings_loss_func(anchor_embed, inputs['anchor_target_embedding'])
            positive_loss = cls.predict_embeddings_loss_func(positive_embed, inputs['positive_target_embedding'])
            negative_loss = cls.predict_embeddings_loss_func(negative_embed, inputs['negative_target_embedding'])

            # Sum and convert to float 32 (to make compatible with triplet loss)
            predict_embeddings_loss = anchor_loss + positive_loss + negative_loss
            # predict_embeddings_loss = anchor_loss.type(torch.float32) + positive_loss.type(
            #     torch.float32) + negative_loss.type(torch.float32)

            if hasattr(cls, 'extra_logs'):  # Log losses separately (see TriplesTrainer)
                cls.extra_logs = {
                    'triplet_loss': round(triplet_loss.item(), log_round),
                    'anchor_loss': round(anchor_loss.item(), log_round),
                    'positive_loss': round(positive_loss.item(), log_round),
                    'negative_loss': round(negative_loss.item(), log_round),
                    'predict_embeddings_loss': round(predict_embeddings_loss.item(), log_round),
                }

            # logger.info(f'triplet_loss = {triplet_loss} ({triplet_loss.dtype})')
            # logger.info(f'predict_embeddings_loss = {predict_embeddings_loss} ({predict_embeddings_loss.dtype})')

            loss = triplet_loss + predict_embeddings_loss  # TODO weighted?

        return loss, anchor_out, positive_out, negative_out
    else:
        # logger.info('inference only')
        pass

    # only inference
    return anchor_embed


def deactivate_bias_gradients(model: nn.Module, needle_name: str = 'bias'):
    """
    Turn off the model parameters requires_grad except the trainable bias terms
    (aka "BitFit" https://arxiv.org/pdf/2106.10199v2.pdf)
    """
    deactivated = []
    activated = []

    for name, param in model.named_parameters():
        if needle_name in name:
            param.requires_grad = True
            activated.append(name)
        else:
            param.requires_grad = False
            deactivated.append(name)

    logger.info(f'Activated parameters: {len(activated)} ({activated[:10]} ...)')
    logger.info(f'Deactivated parameters: {len(deactivated)} ({deactivated[:3]} ...)')

    return model
