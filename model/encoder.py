import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Encoder module for a Neural Transducer.

    Args:
        input_size (int): The dimension of the input data.
        hidden_state_size (int): The dimension of the hidden state in the RNN.
        num_layers (int, optional): The number of recurrent layers.
        rnn_type (str, optional): The type of RNN.
        dropout (float, optional): The dropout rate to apply within the RNN layers.
        bidirectional (bool, optional): Whether to use bidirectional RNN layers.

    Methods:
        forward(input_seq): Forward pass through the encoder.
    """

    supported_rnns = {
        "lstm": nn.LSTM,
        "gru": nn.GRU,
        "rnn": nn.RNN,
    }

    def __init__(
        self,
        input_size: int,
        hidden_state_size: int,
        num_layers: int = 2,
        rnn_type: str = "lstm",
        dropout: float = 0.1,
        bidirectional: bool = True,
    ):
        super(Encoder, self).__init__()
        self.hidden_state_size = hidden_state_size
        rnn_cell = self.supported_rnns[rnn_type.lower()]
        self.dropout = nn.Dropout(dropout)

        self.rnn = rnn_cell(
            input_size,
            hidden_state_size,
            num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional,
        )

    def forward(self, input_seq: torch.Tensor):
        output, (hidden, cell) = self.rnn(input_seq)
        output = self.dropout(output)
        return output, hidden, cell


if __name__ == "__main__":
    input_size = 80
    hidden_size = 256

    encoder = Encoder(input_size, hidden_size)
    input_sequence = torch.randn(32, 100, input_size)
    output, hidden, cell = encoder(input_sequence)
