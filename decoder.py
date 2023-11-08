from typing import Tuple
from torch import Tensor
import torch
import torch.nn as nn


class Decoder(nn.Module):
    """
    Decoder module for RNN-Transducer.

    Args:
        vocab_size (int): The size of the vocabulary.
        emb_dim (int): Dimension of word embeddings.
        pad_idx (int): Index of the padding token in the vocabulary.
        hidden_size (int): Size of the hidden RNN layers.
        n_layers (int): Number of RNN layers.
        rnn_type (str): Type of RNN cell (e.g., 'lstm', 'gru', 'rnn').
        dropout (float): Dropout probability for RNN layers.
    """

    available_rnns = {
        "lstm": nn.LSTM,
        "gru": nn.GRU,
        "rnn": nn.RNN,
    }

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        pad_idx: int,
        hidden_size: int,
        n_layers: int,
        rnn_type: str,
        dropout: float,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        rnn_cell = self.available_rnns[rnn_type.lower()]

        self.rnn = rnn_cell(
            input_size=emb_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
        )

    def forward(
        self,
        inputs: Tensor,
        input_lengths: Tensor = None,
        hidden_states: Tensor = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward propage a `inputs` (targets) for training.

        Args:
            inputs (torch.LongTensor): A target sequence passed to decoder. `IntTensor` of size ``(batch, seq_length)``
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
            hidden_states (torch.FloatTensor): A previous hidden state of decoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``

        Returns:
            (Tensor, Tensor):

            * decoder_outputs (torch.FloatTensor): A output sequence of decoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
            * hidden_states (torch.FloatTensor): A hidden state of decoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
        """
        embedded = self.emb(inputs)
        input_lengths = input_lengths.add(1)
        if input_lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded.transpose(0, 1),
                input_lengths.cpu(),
                enforce_sorted=False,
            )
            outputs, hidden_states = self.rnn(embedded, hidden_states)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        else:
            outputs, hidden_states = self.rnn(embedded, hidden_states)
        return outputs.permute(1, 0, 2), hidden_states

    def get_zeros_hidden_state(
        self, batch_size: int, device: str
    ) -> Tuple[Tensor, Tensor]:
        return (
            torch.zeros((self.n_layers, batch_size, self.hidden_size)).to(device),
            torch.zeros((self.n_layers, batch_size, self.hidden_size)).to(device),
        )
