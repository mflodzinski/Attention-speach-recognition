from typing import Tuple
import torch
import torch.nn as nn

class Decoder(nn.Module):

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
        output_size: int,
        n_layers: int,
        rnn_type: str,
        dropout: float,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.out_proj = nn.Linear(hidden_size, output_size)
        rnn_cell = self.available_rnns.get(rnn_type.lower())

        self.rnn = rnn_cell(
            input_size=emb_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
        )

    def forward(self, inputs, length=None, hidden=None):
        embed_inputs = self.emb(inputs)

        if length is not None:
            sorted_seq_lengths, indices = torch.sort(length, descending=True)
            embed_inputs = embed_inputs[indices]
            embed_inputs = nn.utils.rnn.pack_padded_sequence(
                embed_inputs, sorted_seq_lengths.cpu(), batch_first=True)

        self.rnn.flatten_parameters()
        outputs, hidden = self.rnn(embed_inputs, hidden)

        if length is not None:
            _, desorted_indices = torch.sort(indices, descending=False)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
            outputs = outputs[desorted_indices]

        outputs = self.out_proj(outputs)

        return outputs, hidden

    def get_zeros_hidden_state(
        self, batch_size: int, device: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros((self.n_layers, batch_size, self.hidden_size), device=device),
            torch.zeros((self.n_layers, batch_size, self.hidden_size), device=device),
        )
