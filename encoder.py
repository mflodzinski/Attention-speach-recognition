from torch import Tensor
import torch.nn as nn
import torch


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
        is_bidirectional: bool = True,
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

    def forward(self, inputs: Tensor, input_lengths: Tensor):
        """
        Forward propagate a `inputs` for  encoder training.

        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            (Tensor, Tensor)

            * outputs (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
            * output_lengths (torch.LongTensor): The length of output tensor. ``(batch)``
        """
        print(inputs.size())
        inputs = nn.utils.rnn.pack_padded_sequence(
            inputs.transpose(0, 1),
            input_lengths.cpu(),
            enforce_sorted=False,
        )
        print(inputs.data.size())
        outputs, hidden_states = self.rnn(inputs)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        return outputs.permute(1, 0, 2), input_lengths
