import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod


class SequenceSampler(ABC):
    """
  
    """

    @abstractmethod
    def sample(self, encoder_outputs, h_n, decoder, sos_idx, eos_idx, max_length):
        raise NotImplementedError


class GreedySampler(SequenceSampler):
    """
    """
    def sample(self, encoder_outputs, h_n, decoder, sos_idx, eos_idx, max_length):
        batch_size = encoder_outputs.size(1)
        sequences = None

        input_word = torch.tensor([sos_idx] * batch_size)
        kwargs = {}
        for t in range(max_length):
            output, attn_weights, kwargs = decoder(t, input_word, encoder_outputs, h_n, **kwargs)
            _, argmax = output.max(dim=1)  # greedily take the most probable word
            input_word = argmax
            argmax = argmax.unsqueeze(1)  # (batch) -> (batch, 1) because of concatenating to sequences
            sequences = argmax if sequences is None else torch.cat([sequences, argmax], dim=1)

        # ensure there is EOS token at the end of every sequence (important for calculating lengths)
        end = torch.tensor([eos_idx] * batch_size).unsqueeze(1)  # (batch, 1)
        sequences = torch.cat([sequences, end], dim=1)

        # calculate lengths
        _, lengths = (sequences == eos_idx).max(dim=1)

        return sequences, lengths


class RandomSampler(SequenceSampler):
    """

    """
    def sample(self, encoder_outputs, h_n, decoder, sos_idx, eos_idx, max_length):
        batch_size = encoder_outputs.size(1)
        sequences = None

        input_word = torch.tensor([sos_idx] * batch_size)
        kwargs = {}
        for t in range(max_length):
            output, attn_weights, kwargs = decoder(t, input_word, encoder_outputs, h_n, **kwargs)
            indices = torch.multinomial(F.softmax(output, dim=1), 1)  # roulette-wheel selection of tokens with probability as weights (batch, 1)
            input_word = indices.squeeze(1)  # (batch, 1) -> (batch)
            sequences = indices if sequences is None else torch.cat([sequences, indices], dim=1)

        # ensure there is EOS token at the end of every sequence (important for calculating lengths)
        end = torch.tensor([eos_idx] * batch_size).unsqueeze(1)  # (batch, 1)
        sequences = torch.cat([sequences, end], dim=1)

        # calculate lengths
        _, lengths = (sequences == eos_idx).max(dim=1)

        return sequences, lengths


class Sequence:
    def __init__(self, log_prob, tokens, kwargs):
        self.log_prob = log_prob
        self.tokens = tokens
        self.kwargs = kwargs

    def new_seq(self, tok, log_prob, eos_idx):
        log_prob = log_prob if self.tokens[-1] != eos_idx else 0
        return Sequence(self.log_prob + log_prob, self.tokens + [tok], self.kwargs)

    @property
    def score(self):
        # TODO add alpha
        return self.log_prob * ((5 + len(self.tokens)) / 6)


class BeamSearch(SequenceSampler):
    """

    """
    def __init__(self, beam_width=10, alpha=1):
        self.beam_width = beam_width
        self.alpha = alpha
        self.denominator = 1 / (6**alpha)

    def sample(self, encoder_outputs, h_n, decoder, sos_idx, eos_idx, max_length):
        batch_size = encoder_outputs.size(1)
        sequences = None

        for batch in range(batch_size):
            seq = self._sample(encoder_outputs[:, batch, :].unsqueeze(1), h_n[:, batch, :].unsqueeze(1), decoder, sos_idx, eos_idx, max_length).unsqueeze(0)
            sequences = seq if sequences is None else torch.cat([sequences, seq], dim=1)

        # ensure there is EOS token at the end of every sequence (important for calculating lengths)
        end = torch.tensor([eos_idx] * batch_size).unsqueeze(1)  # (batch, 1)
        sequences = torch.cat([sequences, end], dim=1)

        # calculate lengths
        _, lengths = (sequences == eos_idx).max(dim=1)

        return sequences, lengths

    def _sample(self, encoder_outputs, h_n, decoder, sos_idx, eos_idx, max_length):
        seqs = [Sequence(0, [sos_idx], {})]
        for t in range(max_length):
            new_seqs = []
            for seq in seqs:
                input_word = torch.tensor(seq.tokens[-1]).long().view(1)
                output, _, kwargs = decoder(t, input_word, encoder_outputs, h_n, **seq.kwargs)
                seq.kwargs = kwargs

                output = F.log_softmax(output.squeeze(0), dim=0).tolist()
                for seq in seqs:
                    for tok, out in enumerate(output):
                        new_seqs.append(seq.new_seq(tok, out, eos_idx))

            new_seqs = sorted(new_seqs, key=lambda seq: seq.score)
            seqs = new_seqs[-self.beam_width:]
        return torch.tensor(seqs[-1].tokens)
