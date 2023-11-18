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
        output_size: int,
        n_layers: int,
        rnn_type: str,
        dropout: float,
        is_bidirectional: bool = True,
    ) -> None:
        super().__init__()
        rnn_cell = self.available_rnns[rnn_type.lower()]
        self.out_proj = nn.Linear(hidden_size << 1 if is_bidirectional else hidden_size, output_size)

        self.rnn = rnn_cell(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=is_bidirectional,
        )

    def forward(self, inputs, input_lengths):
        assert inputs.dim() == 3

        if input_lengths is not None:
            sorted_seq_lengths, indices = torch.sort(input_lengths, descending=True)
            inputs = inputs[indices]
            inputs = nn.utils.rnn.pack_padded_sequence(inputs, sorted_seq_lengths.cpu(), batch_first=True)

        self.rnn.flatten_parameters()
        outputs, hidden = self.rnn(inputs)

        if input_lengths is not None:
            _, desorted_indices = torch.sort(indices, descending=False)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
            outputs = outputs[desorted_indices]

        logits = self.out_proj(outputs)

        return logits, hidden