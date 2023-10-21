import torch.nn as nn
from torch import Tensor


class Encoder(nn.Module):
    """
    Encoder module for a RNN-Transducer.

    Args:
        input_size (int): The size of the input data.
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
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        rnn_type: str = "lstm",
        dropout: float = 0.1,
    ):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        rnn_cell = self.supported_rnns[rnn_type.lower()]

        self.rnn = rnn_cell(
            input_size,
            hidden_size,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the Encoder.

        Args:
            x: Input data of shape (batch_size, sequence_length, input_size).

        Returns:
            - output: The output sequence from the Encoder.
            - hidden: The final hidden state of the Encoder.
        """
        output, hidden = self.rnn(x)
        return output, hidden
