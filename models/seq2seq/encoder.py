import torch.nn as nn
from util import RNNWrapper
from abc import ABC, abstractmethod
from .embeddings import embeddings_factory


def encoder_factory(args, metadata):
    embed = embeddings_factory(args, metadata)
    return SimpleEncoder(
        rnn_cls=getattr(nn, args.encoder_rnn_cell),  # gets LSTM or GRU constructor from nn module
        embed=embed,
        embed_size=args.embedding_size,
        hidden_size=args.encoder_hidden_size,
        num_layers=args.encoder_num_layers,
        dropout=args.encoder_rnn_dropout,
        bidirectional=args.encoder_bidirectional
    )


class Encoder(ABC, nn.Module):
    """
 
    """

    @abstractmethod
    def forward(self, input, h_0=None):
        pass

    @property
    @abstractmethod
    def hidden_size(self):
        pass

    @property
    @abstractmethod
    def bidirectional(self):
        pass

    @property
    @abstractmethod
    def num_layers(self):
        pass


class SimpleEncoder(Encoder):
    """
 
    """

    def __init__(self, rnn_cls, embed, embed_size, hidden_size, num_layers=1, dropout=0.2,
                 bidirectional=False):
        super(SimpleEncoder, self).__init__()

        self._hidden_size = hidden_size
        self._bidirectional = bidirectional
        self._num_layers = num_layers

        self.embed = embed
        self.rnn = RNNWrapper(rnn_cls(input_size=embed_size,
                                      hidden_size=hidden_size,
                                      num_layers=num_layers,
                                      dropout=dropout,
                                      bidirectional=bidirectional))

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def bidirectional(self):
        return self._bidirectional

    @property
    def num_layers(self):
        return self._num_layers

    def forward(self, input, h_0=None):
        embedded = self.embed(input)
        outputs, h_n = self.rnn(embedded, h_0)
        return outputs, h_n
