import torch
import torch.nn as nn
import random
import string
from constants import SOS_TOKEN, EOS_TOKEN
from .sampling import GreedySampler, RandomSampler, BeamSearch


class Seq2SeqTrain(nn.Module):
    def __init__(self, encoder, decoder, vocab_size, teacher_forcing_ratio=0.5):
        """
        """
        super(Seq2SeqTrain, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocab_size = vocab_size
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def forward(self, question, answer):
        answer_seq_len = answer.size(0)
        outputs = None

        # encode question sequence
        encoder_outputs, h_n = self.encoder(question)

        kwargs = {}
        input_word = answer[0]  # sos for whole batch
        for t in range(answer_seq_len - 1):
            output, attn_weights, kwargs = self.decoder(t, input_word, encoder_outputs, h_n, **kwargs)

            out = output.unsqueeze(0)  # (batch_size, vocab_size) -> (1, batch_size, vocab_size)
            outputs = out if outputs is None else torch.cat([outputs, out], dim=0)

            teacher_forcing = random.random() < self.teacher_forcing_ratio
            if teacher_forcing:
                input_word = answer[t + 1]  # +1 input word for next timestamp
            else:
                _, argmax = output.max(dim=1)
                input_word = argmax  # index of most probable token (for whole batch)

        return outputs


class Seq2SeqPredict(nn.Module):
    """

    """
    def __init__(self, encoder, decoder, field):
        super(Seq2SeqPredict, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sos_idx = field.vocab.stoi[SOS_TOKEN]
        self.eos_idx = field.vocab.stoi[EOS_TOKEN]
        self.field = field
        self.samplers = {
            'greedy': GreedySampler(),
            'random': RandomSampler(),
            'beam_search': BeamSearch()
        }

    def decode_sequence(self, tokens_idx):
        """

        """
        seq = ''
        for idx in tokens_idx:
            tok = self.field.vocab.itos[idx]
            if tok not in string.punctuation and tok[0] != '\'':
                seq += ' '
            seq += tok
        return seq.strip()

    def forward(self, questions, sampling_strategy, max_seq_len):
        # raw strings to tensor
        q = self.field.process([self.field.preprocess(question) for question in questions])

        # encode question sequence
        encoder_outputs, h_n = self.encoder(q)

        # sample output sequence
        sequences, lengths = self.samplers[sampling_strategy].sample(encoder_outputs, h_n, self.decoder, self.sos_idx,
                                                                     self.eos_idx, max_seq_len)

        # torch tensors -> python lists
        batch_size = sequences.size(0)
        sequences, lengths = sequences.tolist(), lengths.tolist()

        # decode output (token idx -> token string)
        seqs = []
        for batch in range(batch_size):
            seq = sequences[batch][:lengths[batch]]
            seqs.append(self.decode_sequence(seq))

        return seqs
