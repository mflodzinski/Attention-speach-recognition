import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBase(nn.Module):
    def __init__(self):
        super(AttentionBase, self).__init__()

    def _attention_scores(
        self, hidden: torch.Tensor, encoder_outputs: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate attention scores.

        Args:
            hidden (torch.Tensor): The hidden state of the decoder.
                Shape: (batch_size, decoder_hidden_size).
            encoder_outputs (torch.Tensor): The output sequence from the encoder.
                Shape: (batch_size, seq_length, encoder_hidden_size).

        Returns:
            torch.Tensor: Attention scores.
                Shape: (batch_size, seq_length).
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def forward(
        self, hidden: torch.Tensor, encoder_outputs: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the context vector using attention mechanism.

        Args:
            hidden (torch.Tensor): The hidden state of the decoder.
                Shape: (batch_size, decoder_hidden_size).
            encoder_outputs (torch.Tensor): The output sequence from the encoder.
                Shape: (batch_size, seq_length, encoder_hidden_size).

        Returns:
            torch.Tensor: The context vector.
                Shape: (batch_size, 1, encoder_hidden_size).
        """
        attention_scores = self._attention_scores(hidden, encoder_outputs)
        attention_weights = F.softmax(attention_scores, dim=1)
        context = attention_weights.bmm(encoder_outputs)
        return context


class BahdanauAttention(AttentionBase):
    def __init__(self, encoder_hidden_size: int, decoder_hidden_size: int):
        super(BahdanauAttention, self).__init__()
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.W = nn.Linear(encoder_hidden_size, decoder_hidden_size, bias=False)
        self.v = nn.Parameter(torch.rand(encoder_hidden_size))

    def _attention_scores(
        self, hidden: torch.Tensor, encoder_outputs: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate attention scores using Bahdanau mechanism.

        Args:
            hidden (torch.Tensor): The hidden state of the decoder.
                Shape: (batch_size, decoder_hidden_size).
            encoder_outputs (torch.Tensor): The output sequence from the encoder.
                Shape: (batch_size, seq_length, encoder_hidden_size).

        Returns:
            torch.Tensor: Attention scores using Bahdanau mechanism.
                Shape: (batch_size, seq_length).
        """
        seq_len = encoder_outputs.size(1)
        hidden = hidden.repeat(1, seq_len, 1)
        concatenated = torch.cat((hidden, encoder_outputs), dim=2)
        energy = torch.tanh(self.W(concatenated))
        attention_scores = torch.sum(self.v * energy, dim=2)
        return attention_scores


class LuongAttention(AttentionBase):
    def __init__(self, encoder_hidden_size: int, decoder_hidden_size: int):
        super(LuongAttention, self).__init__()
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.W = nn.Linear(encoder_hidden_size, decoder_hidden_size, bias=False)

    def _attention_scores(
        self, hidden: torch.Tensor, encoder_outputs: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate attention scores using Luong mechanism.

        Args:
            hidden (torch.Tensor): The hidden state of the decoder.
                Shape: (batch_size, decoder_hidden_size).
            encoder_outputs (torch.Tensor): The output sequence from the encoder.
                Shape: (batch_size, seq_length, encoder_hidden_size).

        Returns:
            torch.Tensor: Attention scores using Luong mechanism.
                Shape: (batch_size, seq_length).
        """
        attention_scores = self.W(encoder_outputs).permute(0, 2, 1)
        attention_scores = hidden.bmm(attention_scores)
        return attention_scores
