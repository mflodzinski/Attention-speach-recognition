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

    def forward(self, encoder_outputs: Tensor, decoder_outputs: Tensor) -> Tensor:
        """
        Joint `encoder_outputs` and `decoder_outputs`.

        Args:
            encoder_outputs (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
            decoder_outputs (torch.FloatTensor): A output sequence of decoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``

        Returns:
            * outputs (torch.FloatTensor): outputs of joint `encoder_outputs` and `decoder_outputs`..
        """ 
        if encoder_outputs.dim() == 3 and decoder_outputs.dim() == 3:
            encoder_outputs.unsqueeze_(dim=2)
            decoder_outputs.unsqueeze_(dim=1)

        outputs = encoder_outputs + decoder_outputs
        outputs = self.fc(outputs)
        return outputs
