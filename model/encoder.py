import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Encoder module for a Neural Transducer.

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
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.dropout = nn.Dropout(dropout)
        rnn_cell = self.supported_rnns[rnn_type.lower()]

        self.rnn = rnn_cell(
            input_size,
            hidden_size,
            num_layers,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Encoder.

        Args:
            x: Input data of shape (batch_size, sequence_length, input_size).

        Returns:
            - encoder output: The output sequence from the Encoder.
            - hidden state: The final hidden state of the Encoder.
        """
        output, hidden = self.rnn(x)
        output = self.dropout(output)
        return output, hidden


if __name__ == "__main__":
    input_size = 80
    hidden_size = 256

    encoder = Encoder(input_size, hidden_size)
    input_sequence = torch.randn(32, 100, input_size)
    output, hidden, cell = encoder(input_sequence)
