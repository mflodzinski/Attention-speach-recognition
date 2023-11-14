import torch
import torch.nn as nn
from torch import Tensor
from math import ceil

from encoder import Encoder
from decoder import Decoder
from joint import Joint
import torch.nn.functional as F

class Model(nn.Module):
    """
    Initialize a neural network model for sequence-to-sequence tasks.

    Args:
    - `decoder_params`: Parameters for the decoder component.
    - `encoder_params`: Parameters for the encoder component.
    - `joint_params`: Parameters for the joint component.
    - `phi_idx`: Index representing the "phi" token.
    - `pad_idx`: Index representing the "pad" token.
    - `sos_idx`: Index representing the "sos" token.
    - `device`: The device to run the model on.
    """

    def __init__(
        self,
        decoder_params: dict,
        encoder_params: dict,
        joint_params: dict,
        phi_idx: int,
        pad_idx: int,
        sos_idx: int,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.decoder = Decoder(**decoder_params).to(device)
        self.encoder = Encoder(**encoder_params).to(device)
        self.joint = Joint(**joint_params).to(device)
        self.device = device
        self.phi_idx = phi_idx
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx

    def forward(
        self,
        inputs: Tensor,
        input_lengths: Tensor,
        targets: Tensor,
        target_lengths: Tensor,
    ) -> Tensor:
        """
        Forward propagate a `inputs` and `targets` pair for training.

        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
            targets (torch.LongTensr): A target sequence passed to decoder. `IntTensor` of size ``(batch, seq_length)``
            target_lengths (torch.LongTensor): The length of target tensor. ``(batch)``

        Returns:
            * predictions (torch.FloatTensor): Result of model predictions.
        """
        encoder_outputs, _ = self.encoder(inputs, input_lengths)
        concat_targets = F.pad(targets, pad=(1, 0, 0, 0), value=3)
        decoder_outputs, _ = self.decoder(concat_targets, target_lengths.add(1))
        outputs = self.joint(encoder_outputs, decoder_outputs)
        outputs = torch.nn.functional.log_softmax(outputs, dim=-1)
        return outputs

    # def forward(self, x: Tensor, max_length: int) -> Tensor:
    #     """
    #     Forward pass of the model for sequence generation.

    #     Args:
    #     - `x`: The input tensor of shape (B, sequence_length, input_size).
    #     - `max_length`: The maximum length for sequence generation.

    #     Returns:
    #     - `Tensor`: The output tensor of the generated sequence.
    #     """
    #     batch_size, sequence_length, *_ = x.shape
    #     counter = self.init_counter(batch_size, sequence_length)
    #     counter_limit = self.init_counter_limit(batch_size, sequence_length)
    #     prev_preds = self.init_prev_preds(batch_size)

    #     enc_outs = self.encoder(x)
    #     enc_outs = enc_outs.reshape(batch_size * sequence_length, -1)
    #     h, c = self.decoder.get_zeros_hidden_state(batch_size, self.device)

    #     t = 0
    #     result = None

    #     while True:
    #         preds, h, c = self.predict_next(prev_preds, h, c, counter, enc_outs)
    #         curr_preds = torch.argmax(preds, dim=-1)
    #         prev_preds = self.overwrite_preds(prev_preds, curr_preds)
    #         counter, terminate_mask = self.update_states(
    #             curr_preds, counter, counter_limit
    #         )
    #         if t == 0:
    #             result = preds
    #         else:
    #             result = torch.cat([result, preds], dim=1)

    #         t += 1
    #         if (terminate_mask.sum().item() == batch_size) or (max_length == t):
    #             print(terminate_mask.sum().item() == batch_size)
    #             break

    #     return result

    def overwrite_preds(self, prev_preds, curr_preds):
        """
        Keep the last character in the predicted characters, replacing phi
        characters with the last character from the previous predictions.
        Args:
            prev_preds: Previous predicted characters (B, 1).
            curr_preds: Latest predicted characters (B, 1).
        Returns:
            Tensor: Updated predicted characters with phi characters replaced.
        """
        is_phi = curr_preds == self.phi_idx
        return is_phi * prev_preds + (~is_phi) * curr_preds

    def update_states(self, curr_preds, counter, counter_limit):
        """
        Update the positional-related tensors (counter and termination mask) based
        on the current predicted characters.
        Args:
            curr_preds: Current predicted characters (B, 1).
            counter: Counter tensor indicating the current position in the input (B, 1).
            counter_limit: Counter limit that stores the limit of the pointers (B, 1).
        Returns:
            Tuple: A tuple of updated counter, and termination mask.
        """
        counter = counter + (curr_preds.cpu() == self.phi_idx).squeeze()
        counter = torch.minimum(counter, counter_limit)
        terminate_mask = counter >= counter_limit
        return counter, terminate_mask

    def predict_next(self, prev_preds, h, c, counter, enc_outs):
        """
        Predict the next characters based on previous predicted characters and
        hidden states.
        Args:
            prev_preds: Previous predicted characters (B, 1).
            h: Hidden state from the prediction network.
            c: Cell state from the prediction network.
            counter: Counter tensor indicating the current position in the input (B, 1).
            enc_outs: Results from the encoder (B * sequence_length, input_size).
        Returns:
            Tuple: A tuple of predicted characters, updated hidden state, and cell state.
        """
        out, h, c = self.decoder(prev_preds, h, c)
        fy = enc_outs[counter, :].unsqueeze(dim=1)
        preds = self.joint(fy, out)
        return preds, h, c

    # def predict_next(self, prev_preds, h, c, counter, enc_outs):
    #     out, h, c = self.decoder(prev_preds, h, c)
    #     win_start_vec = counter * self.win_size
    #     win_end_vec = win_start_vec + self.win_size
    #     # print(enc_outs.size(0), counter)
    #     windows = []
    #     for win_start, win_end in zip(win_start_vec, win_end_vec):
    #         window = enc_outs[win_start:win_end, :]

    #         # print(window.size())
    #         windows.append(window)

    #     # import numpy as np
    #     # print(np.array(windows).shape)
    #     windows = torch.stack(windows, dim=0)
    #     fy = self.attn(out, windows)
    #     preds = self.joint(fy, out)
    #     return preds, h, c

    def init_counter_limit(self, batch_size: int, max_size: int) -> Tensor:
        num_windows = max_size
        return torch.arange(0, batch_size * num_windows, num_windows) + num_windows - 1

    def init_prev_preds(self, batch_size: int) -> Tensor:
        return torch.LongTensor([[self.sos_idx]] * batch_size).to(self.device)

    def init_counter(self, batch_size: int, max_size: int) -> Tensor:
        num_windows = max_size
        return torch.arange(0, batch_size * num_windows, num_windows)
