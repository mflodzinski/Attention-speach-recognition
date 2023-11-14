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
        n_layers: int,
        rnn_type: str,
        dropout: float,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        rnn_cell = self.available_rnns.get(rnn_type.lower())

        self.rnn = rnn_cell(
            input_size=emb_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        input_lengths: torch.Tensor = None,
        hidden_states: Tuple[torch.Tensor, torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        embedded = self.emb(inputs)

        if input_lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded,
                input_lengths.cpu(),
                enforce_sorted=False,
                batch_first=True,
            )
            outputs, hidden_states = self.rnn(embedded, hidden_states)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
            outputs = self.out_proj(outputs)
        else:
            outputs, hidden_states = self.rnn(embedded, hidden_states)
            outputs = self.out_proj(outputs)
        return outputs, hidden_states


    def get_zeros_hidden_state(
        self, batch_size: int, device: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros((self.n_layers, batch_size, self.hidden_size), device=device),
            torch.zeros((self.n_layers, batch_size, self.hidden_size), device=device),
        )
