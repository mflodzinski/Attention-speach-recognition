from torch import Tensor
import torch.nn as nn


class Encoder(nn.Module):
    """
    Encoder module for RNN-Transducer.

    Args:
        input_size (int): The input feature dimension.
        hidden_size (int): The size of the hidden states in the RNN.
        n_layers (int): The number of RNN layers.
        rnn_type (str): The type of RNN cell, e.g., "lstm," "gru," or "rnn."
        dropout (float): The dropout probability.
        is_bidirectional (bool): Whether the RNN is bidirectional.

    """

    available_rnns = {
        "lstm": nn.LSTM,
        "gru": nn.GRU,
        "rnn": nn.RNN,
    }

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_layers: int,
        rnn_type: str,
        dropout: float,
        is_bidirectional: bool,
    ) -> None:
        super().__init__()
        rnn_cell = self.available_rnns[rnn_type.lower()]

        self.rnn = rnn_cell(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=is_bidirectional,
        )

    def forward(self, x: Tensor) -> Tensor:
        out, *_ = self.rnn(x)
        return out
