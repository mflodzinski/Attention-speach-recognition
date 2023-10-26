import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class AttentionBase(nn.Module):
    def __init__(self):
        super(AttentionBase, self).__init__()

    def _attention_scores(self, dec_out: Tensor, enc_out: Tensor) -> Tensor:
        """
        Calculate attention scores.

        Args:
            dec_out (torch.Tensor): The dec_out state of the decoder.
                Shape: (batch_size, decoder_hidden_size).
            enc_out (torch.Tensor): The output sequence from the encoder.
                Shape: (batch_size, seq_length, encoder_hidden_size).

        Returns:
            torch.Tensor: Attention scores.
                Shape: (batch_size, seq_length).
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def forward(self, dec_out: Tensor, enc_out: Tensor) -> Tensor:
        """
        Calculate the context vector using attention mechanism.

        Args:
            dec_out (torch.Tensor): The dec_out state of the decoder.
                Shape: (batch_size, decoder_hidden_size).
            enc_out (torch.Tensor): The output sequence from the encoder.
                Shape: (batch_size, seq_length, encoder_hidden_size).

        Returns:
            torch.Tensor: The context vector.
                Shape: (batch_size, 1, encoder_hidden_size).
        """
        attention_scores = self._attention_scores(dec_out, enc_out)
        attention_weights = F.softmax(attention_scores, dim=1)
        context = attention_weights.permute(0, 2, 1).bmm(enc_out)
        return context


class BahdanauAttention(AttentionBase):
    def __init__(self, encoder_hidden_size: int, decoder_hidden_size: int):
        super(BahdanauAttention, self).__init__()
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.W = nn.Linear(encoder_hidden_size, decoder_hidden_size, bias=False)
        self.v = nn.Parameter(torch.rand(encoder_hidden_size))

    def _attention_scores(self, dec_out: Tensor, enc_out: Tensor) -> Tensor:
        """
        Calculate attention scores using Bahdanau mechanism.

        Args:
            dec_out (torch.Tensor): The dec_out state of the decoder.
                Shape: (batch_size, decoder_hidden_size).
            enc_out (torch.Tensor): The output sequence from the encoder.
                Shape: (batch_size, seq_length, encoder_hidden_size).

        Returns:
            torch.Tensor: Attention scores using Bahdanau mechanism.
                Shape: (batch_size, seq_length).
        """
        concatenated = torch.cat((dec_out, enc_out), dim=2)
        energy = torch.tanh(self.W(concatenated))
        attention_scores = torch.sum(self.v * energy, dim=2)
        return attention_scores


class LuongAttention(AttentionBase):
    def __init__(self, encoder_hidden_size: int, decoder_hidden_size: int):
        super(LuongAttention, self).__init__()
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.W = nn.Linear(encoder_hidden_size, decoder_hidden_size, bias=False)

    def _attention_scores(self, dec_out: Tensor, enc_out: Tensor) -> Tensor:
        """
        Calculate attention scores using Luong mechanism.

        Args:
            dec_out (torch.Tensor): The dec_out state of the decoder.
                Shape: (batch_size, decoder_hidden_size).
            enc_out (torch.Tensor): The output sequence from the encoder.
                Shape: (batch_size, seq_length, encoder_hidden_size).

        Returns:
            torch.Tensor: Attention scores using Luong mechanism.
                Shape: (batch_size, seq_length).
        """
        attention_scores = self.W(enc_out)
        attention_scores = attention_scores.bmm(dec_out.permute(0, 2, 1))
        return attention_scores
