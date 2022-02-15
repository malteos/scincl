from collections import OrderedDict

from transformers import PretrainedConfig, AutoConfig, \
    RobertaConfig, BertConfig, DebertaConfig, DebertaV2Config

from gdt.models.bert import BertForTripletLoss
from gdt.models.deberta import DebertaForTripletLoss
from gdt.models.deberta_v2 import DebertaV2ForTripletLoss
from gdt.models.roberta import RobertaForTripletLoss


MODEL_FOR_TRIPLET_LOSS_MAPPING = OrderedDict(
    [
        (RobertaConfig, RobertaForTripletLoss),
        (BertConfig, BertForTripletLoss),
        (DebertaConfig, DebertaForTripletLoss),
        (DebertaV2Config, DebertaV2ForTripletLoss),
        (RobertaConfig, RobertaForTripletLoss),
    ]
)


class AutoModelForTripletLoss:
    r"""
        :class:`~transformers.AutoModelForSequenceClassification` is a generic model class
        that will be instantiated as one of the sequence classification model classes of the library
        when created with the `AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path)`
        class method.

        This class cannot be instantiated using `__init__()` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoModelForTripletLoss is designed to be instantiated "
            "using the `AutoModelForTripletLoss.from_pretrained(pretrained_model_name_or_path)` or "
            "`AutoModelForTripletLoss.from_config(config)` methods."
        )

    @classmethod
    def from_config(cls, config):
        r""" Instantiates one of the base model classes of the library
        from a configuration.

        Args:
            config (:class:`~transformers.PretrainedConfig`):
                The model class to instantiate is selected based on the configuration class:

                - isInstance of `distilbert` configuration class: :class:`~transformers.DistilBertForSequenceClassification` (DistilBERT model)
                - isInstance of `albert` configuration class: :class:`~transformers.AlbertForSequenceClassification` (ALBERT model)
                - isInstance of `camembert` configuration class: :class:`~transformers.CamembertForSequenceClassification` (CamemBERT model)
                - isInstance of `xlm roberta` configuration class: :class:`~transformers.XLMRobertaForSequenceClassification` (XLM-RoBERTa model)
                - isInstance of `roberta` configuration class: :class:`~transformers.RobertaForSequenceClassification` (RoBERTa model)
                - isInstance of `bert` configuration class: :class:`~transformers.BertForSequenceClassification` (Bert model)
                - isInstance of `xlnet` configuration class: :class:`~transformers.XLNetForSequenceClassification` (XLNet model)
                - isInstance of `xlm` configuration class: :class:`~transformers.XLMForSequenceClassification` (XLM model)
                - isInstance of `flaubert` configuration class: :class:`~transformers.FlaubertForSequenceClassification` (Flaubert model)


        Examples::

            config = BertConfig.from_pretrained('bert-base-uncased')    # Download configuration from S3 and cache.
            model = AutoModelForSequenceClassification.from_config(config)  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
        """
        for config_class, model_class in MODEL_FOR_TRIPLET_LOSS_MAPPING.items():
            if isinstance(config, config_class):
                return model_class(config)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in MODEL_FOR_TRIPLET_LOSS_MAPPING.keys()),
            )
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r""" Instantiates one of the sequence classification model classes of the library
        from a pre-trained model configuration.

        The `from_pretrained()` method takes care of returning the correct model class instance
        based on the `model_type` property of the config object, or when it's missing,
        falling back to using pattern matching on the `pretrained_model_name_or_path` string.

        The model class to instantiate is selected as the first pattern matching
        in the `pretrained_model_name_or_path` string (in the following order):
            - contains `distilbert`: :class:`~transformers.DistilBertForSequenceClassification` (DistilBERT model)
            - contains `albert`: :class:`~transformers.AlbertForSequenceClassification` (ALBERT model)
            - contains `camembert`: :class:`~transformers.CamembertForSequenceClassification` (CamemBERT model)
            - contains `xlm-roberta`: :class:`~transformers.XLMRobertaForSequenceClassification` (XLM-RoBERTa model)
            - contains `roberta`: :class:`~transformers.RobertaForSequenceClassification` (RoBERTa model)
            - contains `bert`: :class:`~transformers.BertForSequenceClassification` (Bert model)
            - contains `xlnet`: :class:`~transformers.XLNetForSequenceClassification` (XLNet model)
            - contains `flaubert`: :class:`~transformers.FlaubertForSequenceClassification` (Flaubert model)

        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated)
        To train the model, you should first set it back in training mode with `model.train()`

        Args:
            pretrained_model_name_or_path: either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a string with the `identifier name` of a pre-trained model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing model weights saved using :func:`~transformers.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
                - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            model_args: (`optional`) Sequence of positional arguments:
                All remaining positional arguments will be passed to the underlying model's ``__init__`` method

            config: (`optional`) instance of a class derived from :class:`~transformers.PretrainedConfig`:
                Configuration for the model to use instead of an automatically loaded configuation. Configuration can be automatically loaded when:

                - the model is a model provided by the library (loaded with the ``shortcut-name`` string of a pretrained model), or
                - the model was saved using :func:`~transformers.PreTrainedModel.save_pretrained` and is reloaded by suppling the save directory.
                - the model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a configuration JSON file named `config.json` is found in the directory.

            state_dict: (`optional`) dict:
                an optional state dictionnary for the model to use instead of a state dictionary loaded from saved weights file.
                This option can be used if you want to create a model from a pretrained configuration but load your own weights.
                In this case though, you should check if using :func:`~transformers.PreTrainedModel.save_pretrained` and :func:`~transformers.PreTrainedModel.from_pretrained` is not a simpler option.

            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.

            force_download: (`optional`) boolean, default False:
                Force to (re-)download the model weights and configuration files and override the cached versions if they exists.

            resume_download: (`optional`) boolean, default False:
                Do not delete incompletely recieved file. Attempt to resume the download if such a file exists.

            proxies: (`optional`) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
                The proxies are used on each request.

            output_loading_info: (`optional`) boolean:
                Set to ``True`` to also return a dictionnary containing missing keys, unexpected keys and error messages.

            kwargs: (`optional`) Remaining dictionary of keyword arguments:
                These arguments will be passed to the configuration and the model.

        Examples::

            model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')    # Download model and configuration from S3 and cache.
            model = AutoModelForSequenceClassification.from_pretrained('./test/bert_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            assert model.config.output_attention == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
            model = AutoModelForSequenceClassification.from_pretrained('./tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)

        """
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        for config_class, model_class in MODEL_FOR_TRIPLET_LOSS_MAPPING.items():
            if isinstance(config, config_class):
                return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in MODEL_FOR_TRIPLET_LOSS_MAPPING.keys()),
            )
        )
