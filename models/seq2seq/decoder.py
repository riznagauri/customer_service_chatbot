import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from .embeddings import embeddings_factory
from .attention import attention_factory
from .decoder_init import decoder_init_factory


def bahdanau_decoder_factory(args, embed, attn, init, metadata):
    return BahdanauDecoder(
        rnn_cls=getattr(nn, args.encoder_rnn_cell),  # gets LSTM or GRU constructor from nn module
        embed=embed,
        attn=attn,
        init_hidden=init,
        vocab_size=metadata.vocab_size,
        embed_size=args.embedding_size,
        rnn_hidden_size=args.decoder_hidden_size,
        encoder_hidden_size=args.encoder_hidden_size * (2 if args.encoder_bidirectional else 1),
        num_layers=args.decoder_num_layers,
        dropout=args.decoder_rnn_dropout
    )


def luong_decoder_factory(args, embed, attn, init, metadata):
    return LuongDecoder(
        rnn_cls=getattr(nn, args.encoder_rnn_cell),  # gets LSTM or GRU constructor from nn module
        embed=embed,
        attn=attn,
        init_hidden=init,
        vocab_size=metadata.vocab_size,
        embed_size=args.embedding_size,
        rnn_hidden_size=args.decoder_hidden_size,
        attn_hidden_projection_size=args.luong_attn_hidden_size,
        encoder_hidden_size=args.encoder_hidden_size * (2 if args.encoder_bidirectional else 1),
        num_layers=args.decoder_num_layers,
        dropout=args.decoder_rnn_dropout,
        input_feed=args.luong_input_feed
    )


decoder_map = {
    'bahdanau': bahdanau_decoder_factory,
    'luong': luong_decoder_factory
}


def decoder_factory(args, metadata):
    """
    Returns instance of ``Decoder`` based on provided args.
    """
    # TODO what if attention type is 'none' ?
    embed = embeddings_factory(args, metadata)
    attn = attention_factory(args)
    init = decoder_init_factory(args)
    return decoder_map[args.decoder_type](args, embed, attn, init, metadata)


class Decoder(ABC, nn.Module):
    """

    """

    def __init__(self, *args):
        super(Decoder, self).__init__()
        self._args = []
        self._args_init = {}

    def forward(self, t, input, encoder_outputs, h_n, **kwargs):
        """
    
        """
        assert (t == 0 and not kwargs) or (t > 0 and kwargs)

        extra_args = []
        for arg in self.args:
            if t > 0 and arg not in kwargs:
                raise AttributeError("Mandatory arg \"%s\" not present among method arguments" % arg)
            extra_args.append(self.args_init[arg](encoder_outputs, h_n) if t == 0 else kwargs[arg])

        output, attn_weights, *out = self._forward(t, input, encoder_outputs, *extra_args)
        return output, attn_weights, {k: v for k, v in zip(self.args, out)}

    @abstractmethod
    def _forward(self, *args):
        """
    
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def hidden_size(self):
        raise AttributeError

    @property
    @abstractmethod
    def num_layers(self):
        raise AttributeError

    @property
    @abstractmethod
    def has_attention(self):
        raise AttributeError

    @property
    def args(self):
        """
        List of additional arguments concrete subclass wants to receive.
        """
        return self._args

    @args.setter
    def args(self, value):
        self._args = value

    @property
    def args_init(self):
        """
 
        """
        return self._args_init

    @args_init.setter
    def args_init(self, value):
        self._args_init = value


class BahdanauDecoder(Decoder):
    """
    
    """

    LAST_STATE = 'last_state'

    args = [LAST_STATE]

    def __init__(self, rnn_cls, embed, attn, init_hidden, vocab_size, embed_size, rnn_hidden_size, encoder_hidden_size,
                 num_layers=1, dropout=0.2):
        super(BahdanauDecoder, self).__init__()

        self.args_init = {
            self.LAST_STATE: lambda encoder_outputs, h_n: self.initial_hidden(h_n)
        }

        if rnn_hidden_size % 2 != 0:
            raise ValueError('RNN hidden size must be divisible by 2 because of maxout layer.')

        self._hidden_size = rnn_hidden_size
        self._num_layers = num_layers

        self.initial_hidden = init_hidden
        self.embed = embed
        self.rnn = rnn_cls(input_size=embed_size + encoder_hidden_size,
                           hidden_size=rnn_hidden_size,
                           num_layers=num_layers,
                           dropout=dropout)
        self.attn = attn
        self.out = nn.Linear(in_features=rnn_hidden_size // 2, out_features=vocab_size)

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def num_layers(self):
        return self._num_layers

    @property
    def has_attention(self):
        return True

    def _forward(self, t, input, encoder_outputs, last_state):
        embedded = self.embed(input)

        # if RNN is LSTM state is tuple
        last_hidden = last_state[0] if isinstance(last_state, tuple) else last_state

        # prepare rnn input
        attn_weights, context = self.attn(t, last_hidden[-1], encoder_outputs)
        rnn_input = torch.cat([embedded, context], dim=1)
        rnn_input = rnn_input.unsqueeze(0)  # (batch, embed + enc_h) -> (1, batch, embed + enc_h)

        # calculate rnn output
        _, state = self.rnn(rnn_input, last_state)

        # if RNN is LSTM state is tuple
        hidden = state[0] if isinstance(state, tuple) else state

        # maxout layer (with k=2)
        top_layer_hidden = hidden[-1]  # (batch, rnn_hidden)
        batch_size = top_layer_hidden.size(0)
        maxout_input = hidden[-1].view(batch_size, -1, 2)  # (batch, rnn_hidden) -> (batch, rnn_hidden/2, 2) k=2
        maxout, _ = maxout_input.max(dim=2)  # (batch, rnn_hidden/2)

        # calculate logits
        output = self.out(maxout)

        return output, attn_weights, state


class LuongDecoder(Decoder):
    """
    .
    """

    LAST_STATE = 'last_state'
    LAST_ATTN_HIDDEN = 'last_attn_hidden'

    args = [LAST_STATE]

    def __init__(self, rnn_cls, embed, attn, init_hidden, vocab_size, embed_size, rnn_hidden_size,
                 attn_hidden_projection_size, encoder_hidden_size, num_layers=1, dropout=0.2, input_feed=False):
        super(LuongDecoder, self).__init__()

        if input_feed:
            self.args += [self.LAST_ATTN_HIDDEN]

        self.args_init = {
            self.LAST_STATE: lambda encoder_outputs, h_n: self.initial_hidden(h_n),
            self.LAST_ATTN_HIDDEN: lambda encoder_outputs, h_n: self.last_attn_hidden_init(h_n.size(1))  # h_n.size(1) == batch_size
        }

        self._hidden_size = rnn_hidden_size
        self._num_layers = num_layers
        self.initial_hidden = init_hidden

        self.input_feed = input_feed
        self.attn_hidden_projection_size = attn_hidden_projection_size

        rnn_input_size = embed_size + (attn_hidden_projection_size if input_feed else 0)
        self.embed = embed
        self.rnn = rnn_cls(input_size=rnn_input_size,
                           hidden_size=rnn_hidden_size,
                           num_layers=num_layers,
                           dropout=dropout)
        self.attn = attn
        self.attn_hidden_lin = nn.Linear(in_features=rnn_hidden_size + encoder_hidden_size,
                                         out_features=attn_hidden_projection_size)
        self.out = nn.Linear(in_features=attn_hidden_projection_size, out_features=vocab_size)

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def num_layers(self):
        return self._num_layers

    @property
    def has_attention(self):
        return True

    def last_attn_hidden_init(self, batch_size):
        return torch.zeros(batch_size, self.attn_hidden_projection_size) if self.input_feed else None

    def _forward(self, t, input, encoder_outputs, last_state, last_attn_hidden=None):
        assert (self.input_feed and last_attn_hidden is not None) or (not self.input_feed and last_attn_hidden is None)

        embedded = self.embed(input)

        # prepare rnn input
        rnn_input = embedded
        if self.input_feed:
            rnn_input = torch.cat([rnn_input, last_attn_hidden], dim=1)
        rnn_input = rnn_input.unsqueeze(0)  # (batch, rnn_input_size) -> (1, batch, rnn_input_size)

        # rnn output
        _, state = self.rnn(rnn_input, last_state)

        # if RNN is LSTM state is tuple
        hidden = state[0] if isinstance(state, tuple) else state

        # attention context
        attn_weights, context = self.attn(t, hidden[-1], encoder_outputs)
        attn_hidden = torch.tanh(self.attn_hidden_lin(torch.cat([context, hidden[-1]], dim=1)))  # (batch, attn_hidden)

        # calculate logits
        output = self.out(attn_hidden)

        return output, attn_weights, state, attn_hidden
