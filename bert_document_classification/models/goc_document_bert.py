from ..document_bert import BertForDocumentClassification
from .util import get_model_path
import configargparse

def _initialize_arguments(p: configargparse.ArgParser):
    p.add('--model_storage_directory', help='The directory caching all model runs')
    args = p.parse_args()

class GOCBert(BertForDocumentClassification):
    def __init__(self, device='cpu', batch_size=10):
        p = configargparse.ArgParser(default_config_files=["predict_config.ini"])
        args = _initialize_arguments(p)
        model_path = args.model_storage_directory
        
        self.labels = "pos, neg".split(', ')

        super().__init__(device=device,
                         batch_size=batch_size,
                         bert_batch_size=7,
                         bert_model_path=model_path,
                         architecture='DocumentBertLSTM',
                         labels=self.labels)
