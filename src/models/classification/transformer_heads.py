import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import Transformer
from torch.nn.modules import TransformerEncoder

from src.models.classification.heads import AbstractHead
from src.registry import HEADS


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


@HEADS.register_class
class TransformerHead(AbstractHead):
    def __init__(self, in_features, out_features, normalize=True):
        super().__init__(in_features, out_features)
        self.fc = nn.Linear(in_features, out_features)
        self.normalize = normalize
        self.init_weights()

    def forward(self, x: Tensor, targets: Tensor = None) -> Tensor:
        x = torch.flatten(x, start_dim=1)  # bsz, src_seq_len, d_model -> bsz, src_seq_len * d_model
        x = self.fc(x)  # bsz, src_seq_len * d_model -> bsz, tgt_seq_len * d_model
        if self.normalize:
            x = F.normalize(x, p=2, dim=-1)
        return x

    def init_weights(self):
        initrange = 0.1
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)


@HEADS.register_class
class CVTransformer(AbstractHead):
    def __init__(self, in_features, out_features=None, d_model=512, max_src_len=7, max_tgt_len=1,
                 nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=512, dropout=0.1, activation='relu', normalize=True):
        super().__init__(in_features, out_features)
        self.src_pe = PositionalEncoding(d_model, max_src_len)
        self.transformer = Transformer(d_model=d_model, nhead=nhead,
                                       num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward, dropout=dropout, activation=activation)
        self.head = TransformerHead(in_features=d_model * max_src_len, out_features=d_model * max_tgt_len,
                                    normalize=normalize)
        if not out_features:
            out_features = d_model
        self.out_features = out_features

    def forward(self, x: Tensor, targets: Tensor = None) -> Tensor:
        src = x.permute(1, 0, 2)  # bsz, src_seq_len, d_model -> src_seq_len, bsz, d_model
        src = self.src_pe.forward(src)

        memory = self.transformer.encoder.forward(src)  # -> src_seq_len, bsz, d_model
        memory = memory.permute(1, 0, 2)  # ->  bsz, src_seq_len, d_model
        y = self.head(memory)  # -> bsz, tgt_seq_len, d_model
        return y


@HEADS.register_class
class CVTransformerEncoder(AbstractHead):
    def __init__(self, in_features, out_features=None, d_model=512, max_src_len=7,
                 nhead=8, num_encoder_layers=6, normalize=True):
        super().__init__(in_features, out_features)
        self.src_pe = PositionalEncoding(d_model, max_src_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.normalize = normalize
        if not out_features:
            out_features = d_model
        self.out_features = out_features

    def forward(self, x: Tensor, targets: Tensor = None) -> Tensor:
        src = x.permute(1, 0, 2)  # bsz, src_seq_len, d_model -> src_seq_len, bsz, d_model
        src = self.src_pe.forward(src)

        memory = self.encoder.forward(src)  # -> src_seq_len, bsz, d_model
        y = memory[-1]  # -> bsz, d_model
        if self.normalize:
            y = F.normalize(y, p=2, dim=-1)
        return y


@HEADS.register_class
class LSTMHead(AbstractHead):
    def __init__(self, in_features=512, out_features=None, hidden_size=512, num_layers=3,
                 batch_first=False, bias=False, dropout=0, bidirectional=False):
        """

        :param in_features: number of expected features in the input x
        :param out_features: number of output features
        :param hidden_size: number of features in the hidden state h
        :param num_layers: number of recurrent layers
        :param batch_first: if true, input and output tensors are provided as (batch, seq, feature)
        :param bias: if true, add bias weights b_ih and b_hh
        :param dropout: probability of dropout, dropout added after each LSTM layer
        :param bidirectional: true if bidirectional
        """
        super().__init__(in_features, out_features)
        self.lstm = torch.nn.LSTM(input_size=in_features, hidden_size=hidden_size, num_layers=num_layers,
                                  batch_first=batch_first, bias=bias, dropout=dropout,
                                  bidirectional=bidirectional)
        if not out_features:
            out_features = hidden_size
        self.out_features = out_features

    def forward(self, x: Tensor, targets: Tensor = None) -> Tensor:
        # output of shape (seq_len, batch, num_directions * hidden_size)
        # h_n of shape (num_layers * num_directions, batch, hidden_size), initial hidden state
        # c_n of shape (num_layers * num_directions, batch, hidden_size), initial cell state
        output, (h_n, c_n) = self.lstm(x)
        output = F.normalize(output[-1], p=2, dim=-1)
        return output


@HEADS.register_class
class LAttQE_Head(AbstractHead):
    # https://filipradenovic.github.io/publications/grb20.pdf
    def __init__(self, in_features, out_features=None, num_neighbour=7, nhead=64, num_encoder_layers=3, normalize=True):
        super().__init__(in_features, out_features)
        self.num_neighbour = num_neighbour
        self.in_features = in_features
        self.out_features = in_features
        self.src_pe = PositionalEncoding(in_features, self.num_neighbour)
        encoder_layer = nn.TransformerEncoderLayer(d_model=in_features, nhead=nhead)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.weights = nn.Parameter(torch.ones(self.num_neighbour))
        self.normalize = normalize
        self.init_weights()

    def forward(self, x: Tensor, targets: Tensor = None) -> Tensor:
        src = x.permute(1, 0, 2)  # bsz, src_seq_len, d_model -> src_seq_len, bsz, d_model
        src = self.src_pe.forward(src)
        memory = self.encoder.forward(src)  # -> src_seq_len, bsz, d_model
        memory = memory.view(self.num_neighbour, -1)  # -> src_seq_len, bsz*d_model
        y = F.softmax(self.weights, dim=0) @ memory  # -> bsz*d_model
        y = y.view(-1, self.in_features)
        if self.normalize:
            y = F.normalize(y, p=2, dim=-1)
        return y
