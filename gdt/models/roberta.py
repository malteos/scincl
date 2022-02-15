from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel

from gdt.models import PoolingStrategy, get_loss_func, triplet_forward, get_mlm, get_predict_embeddings_and_loss_func


class RobertaForTripletLoss(RobertaPreTrainedModel):
    def __init__(
            self,
            config,
            loss_func=None,
            masked_language_modeling: bool = False,
            masked_language_modeling_weight: float = 1.0,
            predict_embeddings: bool = False,
            pooling_strategy: PoolingStrategy = PoolingStrategy.CLS
    ):
        super().__init__(config)

        self.roberta = RobertaModel(config)
        self.pooling_strategy = pooling_strategy
        self.masked_language_modeling, self.masked_language_modeling_weight, self.cls = get_mlm(masked_language_modeling, masked_language_modeling_weight, None)
        self.loss_func = get_loss_func(loss_func)
        self.predict_embeddings, self.predict_embeddings_loss_func = get_predict_embeddings_and_loss_func(
            predict_embeddings)
        self.extra_logs = {}

        self.init_weights()

    def forward(self, **inputs):
        return triplet_forward(self, self.roberta, **inputs)

    def _reorder_cache(self, past, beam_idx):
        raise NotImplementedError()
