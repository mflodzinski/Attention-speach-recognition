import torch
import torch.nn as nn
from typing import Tuple


class Decoder(nn.Module):
    """
    Decoder module for a Neural Transducer.

    Args:
        output_size (int): The size of the output vocabulary.
        hidden_size (int): The dimension of the hidden state in the RNN.
        num_layers (int, optional): The number of recurrent layers.
        rnn_type (str, optional): The type of RNN cell to use.
        dropout (float, optional): Dropout rate for the RNN layers.
    """

    supported_rnns = {
        "lstm": nn.LSTM,
        "gru": nn.GRU,
        "rnn": nn.RNN,
    }

    def __init__(
        self,
        output_size: int,
        hidden_size: int,
        num_layers: int = 2,
        rnn_type: str = "gru",
        dropout: float = 0.1,
    ):
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        rnn_cell = self.supported_rnns[rnn_type.lower()]

        self.rnn = rnn_cell(
            hidden_size,
            hidden_size,
            num_layers,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self, input: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the Decoder.

        Args:
            input: The input tensor representing previous output symbols.
            hidden: The initial hidden state of the RNN.

        Returns:
            The prediction and the updated hidden state.
        """
        embedded = self.embedding(input)
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.fc(output)
        return prediction, hidden
