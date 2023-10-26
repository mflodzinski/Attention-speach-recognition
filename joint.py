from torch import Tensor
import torch
import torch.nn as nn


class Joint(nn.Module):
    """
    Joint module for RNN-Transducer.

    Args:
        input_size: The size of the input features.
        vocab_size: The size of the target vocabulary.
    """

    def __init__(self, input_size: int, vocab_size: int) -> None:
        super().__init__()
        self.fc = nn.Linear(input_size, vocab_size)

    def forward(self, f: Tensor, g: Tensor) -> Tensor:
        out = f + g
        out = self.fc(out)
        return torch.softmax(out, dim=-1)
