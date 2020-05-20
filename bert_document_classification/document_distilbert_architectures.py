from pytorch_transformers.modeling_bert import BertPreTrainedModel
from transformers import DistilBertConfig, DistilBertModel, PretrainedConfig, PreTrainedModel
from transformers import DistilBertForSequenceClassification
from transformers import DistilBertForMaskedLM
from torch import nn
import torch
from .transformer import TransformerEncoderLayer, TransformerEncoder

import logging

logging.basicConfig(level=logging.INFO)

from torch.nn import LSTM
class DocumentBertLSTM(nn.Module):
    """
    BERT output over document in LSTM
    """

    def __init__(self, bert_model_config: PretrainedConfig):
        super(DocumentBertLSTM, self).__init__()
        self.bert = DistilBertModel.from_pretrained("/home/mrbarnes/models/bluebert_train_output/")
        self.bert_batch_size= bert_model_config.bert_batch_size
        self.dropout = nn.Dropout(p=bert_model_config.dropout)
        self.pre_classifier = nn.Sequential(
            nn.Linear(bert_model_config.hidden_size, bert_model_config.hidden_size),
            nn.Tanh()
        )
        self.lstm = LSTM(bert_model_config.hidden_size,bert_model_config.hidden_size)
        self.classifier = nn.Sequential(
            nn.Dropout(p=bert_model_config.dropout),
            nn.Linear(bert_model_config.hidden_size, bert_model_config.num_labels),
            nn.Tanh()
        )

        for name, param in self.bert.named_parameters():
            print(name)

    #input_ids, token_type_ids, attention_masks
    def forward(self, document_batch: torch.Tensor, document_sequence_lengths: list, device='cuda'):

        #contains all BERT sequences
        #bert should output a (batch_size, num_sequences, bert_hidden_size)
        bert_output = torch.zeros(size=(document_batch.shape[0],
                                              min(document_batch.shape[1],self.bert_batch_size),
                                              self.bert.config.hidden_size), dtype=torch.float, device=device)

        #only pass through bert_batch_size numbers of inputs into bert.
        #this means that we are possibly cutting off the last part of documents.

        for doc_id in range(document_batch.shape[0]):
            print(type(self.bert(input_ids=document_batch[doc_id][:self.bert_batch_size,0],
                                            attention_mask=document_batch[doc_id][:self.bert_batch_size,2])[0]))
            bert_output[doc_id][:self.bert_batch_size] = self.dropout(self.pre_classifier(self.bert(input_ids=document_batch[doc_id][:self.bert_batch_size,0],
                                            attention_mask=document_batch[doc_id][:self.bert_batch_size,2])[0][:,0,:]))

        output, (_, _) = self.lstm(bert_output.permute(1,0,2))

        last_layer = output[-1]

        prediction = self.classifier(last_layer)

        assert prediction.shape[0] == document_batch.shape[0]
        return prediction

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True

    def unfreeze_bert_encoder_last_layers(self):
        for name, param in self.bert.named_parameters():
            if "transformer.layer.5" in name or "pooler" in name:
                param.requires_grad = True
    def unfreeze_bert_encoder_pooler_layer(self):
        for name, param in self.bert.named_parameters():
            if "pooler" in name:
                param.requires_grad = True


class DocumentBertLinear(BertPreTrainedModel):
    """
    BERT output over document into linear layer
    """

    def __init__(self, bert_model_config: DistilBertConfig):
        super(DocumentBertLinear, self).__init__(bert_model_config)
        self.bert = DistilBertModel(bert_model_config)
        self.bert_batch_size= self.bert.config.bert_batch_size
        self.dropout = nn.Dropout(p=bert_model_config.hidden_dropout_prob)

        self.classifier = nn.Sequential(
            nn.Dropout(p=bert_model_config.hidden_dropout_prob),
            nn.Linear(bert_model_config.hidden_size * self.bert_batch_size, bert_model_config.num_labels),
            nn.Tanh()
        )

    #input_ids, token_type_ids, attention_masks
    def forward(self, document_batch: torch.Tensor, document_sequence_lengths: list, device='cuda'):

        #contains all BERT sequences
        #bert should output a (batch_size, num_sequences, bert_hidden_size)
        bert_output = torch.zeros(size=(document_batch.shape[0],
                                              min(document_batch.shape[1],self.bert_batch_size),
                                              self.bert.config.hidden_size), dtype=torch.float, device='cuda')

        #only pass through bert_batch_size numbers of inputs into bert.
        #this means that we are possibly cutting off the last part of documents.

        for doc_id in range(document_batch.shape[0]):
            bert_output[doc_id][:self.bert_batch_size] = self.dropout(self.bert(document_batch[doc_id][:self.bert_batch_size,0],
                                            token_type_ids=document_batch[doc_id][:self.bert_batch_size,1],
                                            attention_mask=document_batch[doc_id][:self.bert_batch_size,2])[1])


        prediction = self.classifier(bert_output.view(bert_output.shape[0], -1))
        assert prediction.shape[0] == document_batch.shape[0]
        return prediction

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True

    def unfreeze_bert_encoder_last_layers(self):
        for name, param in self.bert.named_parameters():
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True
    def unfreeze_bert_encoder_pooler_layer(self):
        for name, param in self.bert.named_parameters():
            if "pooler" in name:
                param.requires_grad = True


class DocumentBertMaxPool(BertPreTrainedModel):
    """
    BERT output over document into linear layer
    """

    def __init__(self, bert_model_config: DistilBertConfig):
        super(DocumentBertMaxPool, self).__init__(bert_model_config)
        self.bert = DistilBertModel(bert_model_config)
        self.bert_batch_size= self.bert.config.bert_batch_size
        self.dropout = nn.Dropout(p=bert_model_config.hidden_dropout_prob)

        # self.transformer_encoder = TransformerEncoderLayer(d_model=bert_model_config.hidden_size,
        #                                            nhead=6,
        #                                            dropout=bert_model_config.hidden_dropout_prob)
        #self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=6, norm=nn.LayerNorm(bert_model_config.hidden_size))
        self.classifier = nn.Sequential(
            nn.Dropout(p=bert_model_config.hidden_dropout_prob),
            nn.Linear(bert_model_config.hidden_size, bert_model_config.num_labels),
            nn.Tanh()
        )

    #input_ids, token_type_ids, attention_masks
    def forward(self, document_batch: torch.Tensor, document_sequence_lengths: list, device='cuda'):

        #contains all BERT sequences
        #bert should output a (batch_size, num_sequences, bert_hidden_size)
        bert_output = torch.zeros(size=(document_batch.shape[0],
                                              min(document_batch.shape[1],self.bert_batch_size),
                                              self.bert.config.hidden_size), dtype=torch.float, device='cuda')

        #only pass through bert_batch_size numbers of inputs into bert.
        #this means that we are possibly cutting off the last part of documents.

        for doc_id in range(document_batch.shape[0]):
            bert_output[doc_id][:self.bert_batch_size] = self.dropout(self.bert(document_batch[doc_id][:self.bert_batch_size,0],
                                            token_type_ids=document_batch[doc_id][:self.bert_batch_size,1],
                                            attention_mask=document_batch[doc_id][:self.bert_batch_size,2])[1])


        prediction = self.classifier(bert_output.max(dim=1)[0])
        assert prediction.shape[0] == document_batch.shape[0]
        return prediction

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True

    def unfreeze_bert_encoder_last_layers(self):
        for name, param in self.bert.named_parameters():
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True
    def unfreeze_bert_encoder_pooler_layer(self):
        for name, param in self.bert.named_parameters():
            if "pooler" in name:
                param.requires_grad = True


class DocumentBertTransformer(BertPreTrainedModel):
    """
    BERT -> TransformerEncoder -> Max over attention output.
    """

    def __init__(self, bert_model_config: DistilBertConfig):
        super(DocumentBertTransformer, self).__init__(bert_model_config)
        self.bert = DistilBertModel(bert_model_config)
        self.bert_batch_size= self.bert.config.bert_batch_size
        self.dropout = nn.Dropout(p=bert_model_config.hidden_dropout_prob)

        encoder_layer = TransformerEncoderLayer(d_model=bert_model_config.hidden_size,
                                                   nhead=6,
                                                   dropout=bert_model_config.hidden_dropout_prob)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=6)
        self.classifier = nn.Sequential(
            nn.Dropout(p=bert_model_config.hidden_dropout_prob),
            nn.Linear(bert_model_config.hidden_size, bert_model_config.num_labels),
            nn.Tanh()
        )

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True

    def unfreeze_bert_encoder_last_layers(self):
        for name, param in self.bert.named_parameters():
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True
    def unfreeze_bert_encoder_pooler_layer(self):
        for name, param in self.bert.named_parameters():
            if "pooler" in name:
                param.requires_grad = True

    #input_ids, token_type_ids, attention_masks
    def forward(self, document_batch: torch.Tensor, document_sequence_lengths: list):

        #contains all BERT sequences
        #bert should output a (batch_size, num_sequences, bert_hidden_size)
        bert_output = torch.zeros(size=(document_batch.shape[0],
                                              min(document_batch.shape[1],self.bert_batch_size),
                                              self.bert.config.hidden_size), dtype=torch.float, device='cuda')

        #only pass through bert_batch_size numbers of inputs into bert.
        #this means that we are possibly cutting off the last part of documents.
        for doc_id in range(document_batch.shape[0]):
            bert_output[doc_id][:self.bert_batch_size] = self.dropout(self.bert(document_batch[doc_id][:self.bert_batch_size,0],
                                            token_type_ids=document_batch[doc_id][:self.bert_batch_size,1],
                                            attention_mask=document_batch[doc_id][:self.bert_batch_size,2])[1])

        transformer_output = self.transformer_encoder(bert_output.permute(1,0,2))

        prediction = self.classifier(transformer_output.permute(1,0,2).max(dim=1)[0])
        assert prediction.shape[0] == document_batch.shape[0]
        return prediction

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True

    def unfreeze_bert_encoder_last_layers(self):
        for name, param in self.bert.named_parameters():
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True
    def unfreeze_bert_encoder_pooler_layer(self):
        for name, param in self.bert.named_parameters():
            if "pooler" in name:
                param.requires_grad = True
