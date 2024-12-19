from transformers import BertModel, BertTokenizer
import torch.nn as nn

class BertClassifier(nn.Module):
    """Modèle de classification basé sur BERT"""
    def __init__(self, num_classes, dropout=0.1):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        dropout_output = self.dropout(pooled_output)
        return self.linear(dropout_output)