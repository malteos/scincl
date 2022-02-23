import logging
import pickle
from typing import Dict

import fire

logger = logging.getLogger(__name__)

# pytorch packages

# pytorch lightning packages
#import pytorch_lightning as pl
#from pytorch_lightning.loggers import TensorBoardLogger
#from pytorch_lightning.callbacks import ModelCheckpoint

# huggingface transformers packages

# allennlp dataloading packages
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.tokenizers import Tokenizer
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers.word_splitter import WordSplitter
from allennlp.data.tokenizers.token import Token


class DataReaderFromPickled(DatasetReader):
    """
    This is copied from https://github.com/allenai/specter/blob/673346f9f76bcf422b38e0d1b448ef4414bcd4df/specter/data.py#L61:L109 without any change
    """
    def __init__(self,
                 lazy: bool = False,
                 word_splitter: WordSplitter = None,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_sequence_length: int = 256,
                 concat_title_abstract: bool = None
                 ) -> None:
        """
        Dataset reader that uses pickled preprocessed instances
        Consumes the output resulting from data_utils/create_training_files.py
        the additional arguments are not used here and are for compatibility with
        the other data reader at prediction time
        """
        self.max_sequence_length = max_sequence_length
        self.token_indexers = token_indexers
        self._concat_title_abstract = concat_title_abstract
        super().__init__(lazy)

    def _read(self, file_path: str):
        """
        Args:
            file_path: path to the pickled instances
        """
        with open(file_path, 'rb') as f_in:
            unpickler = pickle.Unpickler(f_in)
            while True:
                try:
                    instance = unpickler.load()
                    # compatibility with old models:
                    # for field in instance.fields:
                    #     if hasattr(instance.fields[field], '_token_indexers') and 'base_model' in instance.fields[field]._token_indexers:
                    #         if not hasattr(instance.fields['source_title']._token_indexers['base_model'], '_truncate_long_sequences'):
                    #             instance.fields[field]._token_indexers['base_model']._truncate_long_sequences = True
                    #             instance.fields[field]._token_indexers['base_model']._token_min_padding_length = 0
                    if self.max_sequence_length:
                        for paper_type in ['source', 'pos', 'neg']:
                            if self._concat_title_abstract:
                                tokens = []
                                title_field = instance.fields.get(f'{paper_type}_title')
                                abst_field = instance.fields.get(f'{paper_type}_abstract')
                                if title_field:
                                    tokens.extend(title_field.tokens)
                                if tokens:
                                    tokens.extend([Token('[SEP]')])
                                if abst_field:
                                    tokens.extend(abst_field.tokens)
                                if title_field:
                                    title_field.tokens = tokens
                                    instance.fields[f'{paper_type}_title'] = title_field
                                elif abst_field:
                                    abst_field.tokens = tokens
                                    instance.fields[f'{paper_type}_title'] = abst_field
                                else:
                                    yield None
                                # title_tokens = get_text_tokens(query_title_tokens, query_abstract_tokens, abstract_delimiter)
                                # pos_title_tokens = get_text_tokens(pos_title_tokens, pos_abstract_tokens, abstract_delimiter)
                                # neg_title_tokens = get_text_tokens(neg_title_tokens, neg_abstract_tokens, abstract_delimiter)
                                # query_abstract_tokens = pos_abstract_tokens = neg_abstract_tokens = []
                            for field_type in ['title', 'abstract', 'authors', 'author_positions']:
                                field = paper_type + '_' + field_type
                                if instance.fields.get(field):
                                    instance.fields[field].tokens = instance.fields[field].tokens[
                                                                    :self.max_sequence_length]
                                if field_type == 'abstract' and self._concat_title_abstract:
                                    instance.fields.pop(field, None)
                    yield instance
                except EOFError:
                    break

def extract_paper_ids(input_fp, output_fp=None):
    """
    Extract triples as paper Ids from SPECTER pickle files into CSV
    
    python specter_dataset.py extract_paper_ids ./data/specter/val.pkl ./data/specter/val_triples.csv
    python specter_dataset.py extract_paper_ids ./data/specter/train.pkl ./data/specter/train_triples.csv
    
    """
    datareaderfp = DataReaderFromPickled(max_sequence_length=512)
    data_instances = datareaderfp._read(input_fp)
                    
    triples = []
    
    for item in data_instances:
        triples.append((
            item['source_paper_id'].metadata,
            item['pos_paper_id'].metadata,
            item['neg_paper_id'].metadata))
        
    if output_fp:
        # write output
        with open(output_fp, 'w') as f:
            f.write('query_paper_id,positive_id,negative_id\n')
            for query_paper_id, pos_id, neg_id in triples:
                f.write(f'{query_paper_id},{pos_id},{neg_id}\n')
            
        logger.info('done')
    else:
        # return
        return triples
    

if __name__ == '__main__':
    fire.Fire()