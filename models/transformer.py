import copy

import math
import torch
from torch.nn import TransformerEncoderLayer, ModuleList
from torch import nn

class Classifier(nn.Module):
    def __init__(self, in_planes, num_classes):
        super(Classifier, self).__init__()

        self.in_planes = in_planes
        self.num_classes = num_classes
        self.ee_fc = nn.Linear(in_planes, num_classes)

    def forward(self, x):
        return self.ee_fc(x)

# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 200):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerModel(nn.Module):

    def __init__(self,
                 args,
                 ntoken: int = 32000,
                 d_model: int = 192,
                 nhead: int = 2,
                 nlayers: int = 4,
                 dropout: float = 0.1,
                 max_len: int = 200,
                 d_hid: int = 768):
        super().__init__()

        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        encoder_layer = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)

        self.layers = _get_clones(encoder_layer, nlayers)

        self.embedding = nn.Embedding(ntoken, d_model)
        self.hidden_dim = d_model


        self.ee1 = Classifier(d_model, args.class_num)
        self.ee2 = Classifier(d_model, args.class_num)
        self.ee3 = Classifier(d_model, args.class_num)
        self.ee4 = Classifier(d_model, args.class_num)
        self.ee_classifiers = [self.ee1, self.ee2, self.ee3, self.ee4]

    def forward(self, x, return_feat=False):
        x = self.embedding(x) * math.sqrt(self.hidden_dim)
        x = self.pos_encoder(x)

        outs = []

        for idx, l in enumerate(self.layers):
            x = l(x)
            outs.append(self.ee_classifiers[idx](x[:,0])) # [CLS] token

        if return_feat:
            return outs, x[:,0]
        return outs

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for _ in range(N)])

def transformer(args, params):
    return TransformerModel(args)