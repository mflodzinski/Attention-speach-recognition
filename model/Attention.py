import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBase(nn.Module):
    def __init__(self):
        super(AttentionBase, self).__init__()

    def _attention_scores(
        self, hidden: torch.Tensor, encoder_outputs: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement this method.")

    def forward(
        self, hidden: torch.Tensor, encoder_outputs: torch.Tensor
    ) -> torch.Tensor:
        attention_scores = self._attention_scores(hidden, encoder_outputs)
        attention_weights = F.softmax(attention_scores, dim=1).unsqueeze(1)
        context = attention_weights.bmm(encoder_outputs)
        return context


class BahdanauAttention(AttentionBase):
    def __init__(self, hidden_size: int):
        super(BahdanauAttention, self).__init__(hidden_size)
        self.hidden_size = hidden_size

        self.W = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Parameter(torch.rand(hidden_size))

    def _attention_scores(
        self, hidden: torch.Tensor, encoder_outputs: torch.Tensor
    ) -> torch.Tensor:
        seq_len = encoder_outputs.size(1)
        hidden = hidden.repeat(1, seq_len, 1)
        concatenated = torch.cat((hidden, encoder_outputs), dim=2)
        energy = torch.tanh(self.W(concatenated))
        attention_scores = torch.sum(self.v * energy, dim=2)
        return attention_scores


class LuongAttention(AttentionBase):
    def __init__(self, hidden_size: int):
        super(LuongAttention, self).__init__(hidden_size)
        self.hidden_size = hidden_size
        self.W = nn.Linear(hidden_size, hidden_size, bias=False)

    def _attention_scores(
        self, hidden: torch.Tensor, encoder_outputs: torch.Tensor
    ) -> torch.Tensor:
        attention_scores = hidden.bmm(self.W(encoder_outputs).permute(0, 2, 1))
        attention_scores = attention_scores.squeeze(1)
        return attention_scores
