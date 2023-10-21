import torch.nn as nn

from typing import Tuple
from torch import Tensor


class Decoder(nn.Module):
    """
    Decoder module for a RNN-Transducer.

    Args:
        num_classes (int): number of classification
        hidden_state_dim (int, optional): hidden state dimension of decoder
        output_dim (int, optional): output dimension of encoder and decoder
        num_layers (int, optional): number of decoder layers
        rnn_type (str, optional): type of rnn cell
        sos_id (int, optional): start of sentence identification
        eos_id (int, optional): end of sentence identification
        dropout (float, optional): dropout probability of decoder
    """

    supported_rnns = {
        "lstm": nn.LSTM,
        "gru": nn.GRU,
        "rnn": nn.RNN,
    }

    def __init__(
        self,
        num_classes: int,
        hidden_size: int,
        num_layers: int = 2,
        rnn_type: str = "lstm",
        blank_id: int = 0,
        sos_id: int = 1,
        eos_id: int = 2,
        dropout: float = 0.1,
    ):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.blank_id = blank_id
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.embedding = nn.Embedding(num_classes, hidden_size)
        rnn_cell = self.supported_rnns[rnn_type.lower()]

        self.rnn = rnn_cell(
            hidden_size,
            hidden_size,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

    def forward(
        self, y: Tensor, hidden: Tuple[Tensor, Tensor] = None
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        Forward pass of the Decoder.

        Args:
            y: The tensor representing previous output symbol (batch_size, 1).
            hidden: The initial hidden state of the RNN (h_n, c_n).

        Returns:
            - output: An output sequence from the Decoder (batch_size, 1, hidden_size).
            - hidden: An updated hidden state of the Decoder (h_n, c_n).
        """
        embedded = self.embedding(y)
        output, hidden = self.rnn(embedded, hidden)
        return output, hidden
