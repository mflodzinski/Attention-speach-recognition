import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from encoder import Encoder
from decoder import Decoder
from attention import BahdanauAttention, LuongAttention


class RNNTransducer(nn.Module):
    """
    RNN-Transducer model.

    Args:
        num_classes (int): The number of output classes (e.g., vocabulary size).
        input_size (int): The dimension of the input feature vectors.
        num_encoder_layers (int, optional): The number of layers in the encoder.
        num_decoder_layers (int, optional): The number of layers in the decoder.
        encoder_hidden_size (int, optional): The dimension of the encoder's hidden state.
        decoder_hidden_size (int, optional): The dimension of the decoder's hidden state.
        rnn_type (str, optional): The type of RNN cell to use.
        encoder_dropout (float, optional): Dropout probability for the encoder.
        decoder_dropout (float, optional): Dropout probability for the decoder.

    Inputs: inputs, input_lengths, targets, target_lengths
        - inputs (torch.Tensor): Input feature sequences with shape (batch_size, sequence_length, input_size).
        - input_lengths (torch.Tensor): Lengths of input sequences in the batch.
        - targets (torch.Tensor): Target sequences with shape (batch_size, target_sequence_length).
        - target_lengths (torch.Tensor): Lengths of target sequences in the batch.

    Returns:
        torch.Tensor: The model's output sequence of log probabilities with shape (batch_size, target_sequence_length, num_classes).
    """

    def __init__(
        self,
        num_classes: int,
        input_size: int,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 1,
        encoder_hidden_size: int = 320,
        decoder_hidden_size: int = 512,
        rnn_type: str = "lstm",
        encoder_dropout: float = 0.2,
        decoder_dropout: float = 0.2,
        attn_win_size: int = 10,
    ):
        super(RNNTransducer, self).__init__()
        self.num_classes = num_classes
        self.win_size = attn_win_size
        self.encoder = Encoder(
            input_size=input_size,
            hidden_size=encoder_hidden_size,
            num_layers=num_encoder_layers,
            rnn_type=rnn_type,
            dropout=encoder_dropout,
        )
        self.decoder = Decoder(
            num_classes=num_classes,
            hidden_size=decoder_hidden_size,
            num_layers=num_decoder_layers,
            rnn_type=rnn_type,
            dropout=decoder_dropout,
        )
        self.fc = nn.Linear(
            encoder_hidden_size + decoder_hidden_size, num_classes, bias=False
        )
        self.attn = LuongAttention(encoder_hidden_size, decoder_hidden_size)

    def _joint(self, encoder_features: Tensor, decoder_features: Tensor) -> Tensor:
        """
        Combine encoder and decoder features and pass through a linear layer.

        Args:
            encoder_features: Features from the encoder
            decoder_features: Features from the decoder

        Returns:
            outputs: Joint output features
        """
        outputs = torch.cat((encoder_features, decoder_features), dim=-1)
        outputs = self.fc(outputs)
        return outputs

    def forward(
        self,
        inputs: Tensor,
        outputs: Tensor,
        hidden_state: Tensor = None,
    ) -> Tensor:
        """
        Perform the forward pass of the RNN-Transducer model.

        Args:
            inputs: Input sequence passed to the encoder
            outputs: Already predicted sequence passed to the decoder
            hidden_state: Hidden state passed to the decoder
        Returns:
            outputs: Model predictions
            hidden_state: Updated hidden state of the decoder
        """
        encoder_features = self.encoder(inputs)
        decoder_features, hidden_state = self.decoder(outputs, hidden_state)
        outputs = self._joint(encoder_features, decoder_features)
        return outputs, hidden_state

    @torch.no_grad()
    def _decode(self, encoder_output: Tensor, max_length: int) -> Tensor:
        token_list = []
        bos_token = torch.LongTensor([[self.decoder.sos_id]])
        decoder_output, hidden_state = self.decoder(bos_token)

        for t in range(0, max_length, self.win_size):
            window = encoder_output[t : t + self.win_size]

            pred_tokens, decoder_output, hidden_state = self._decode_window(
                window, decoder_output, hidden_state
            )
            token_list.extend(pred_tokens.tolist())
        return torch.LongTensor(token_list)

    @torch.no_grad()
    def _decode_window(
        self, window: Tensor, decoder_output: Tensor, hidden_state: Tensor
    ) -> Tensor:
        token_list = []
        window = window.unsqueeze(0)

        for _ in range(self.win_size):
            context = self.attn(decoder_output, window)
            step_output = self._joint(context.view(-1), decoder_output.view(-1))
            step_output = step_output.softmax(dim=0)
            pred_token = step_output.argmax(dim=0)
            pred_token = int(pred_token.item())
            if pred_token != self.decoder.blank_id:
                token_list.append(pred_token)
                token = torch.LongTensor([[pred_token]])
                decoder_output, hidden_state = self.decoder(token, hidden_state)
            else:
                break
        return torch.LongTensor(token_list), decoder_output, hidden_state

    @torch.no_grad()
    def _pad_encoder_outputs(self, encoder_outputs: Tensor) -> Tensor:
        encoder_seq_length = encoder_outputs.size(1)
        if encoder_seq_length % self.win_size != 0:
            padding = self.win_size - (encoder_seq_length % self.win_size)
        else:
            padding = 0

        encoder_outputs = F.pad(encoder_outputs, (0, 0, 0, padding, 0, 0))
        return encoder_outputs

    @torch.no_grad()
    def recognize(self, inputs: Tensor) -> Tensor:
        """
        Recognize input speech. This method consists of the forward of the encoder and the decode() of the decoder.

        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            * predictions (torch.FloatTensor): Result of model predictions.
        """
        token_list = []
        encoder_outputs, _ = self.encoder(inputs)
        encoder_outputs = self._pad_encoder_outputs(encoder_outputs)
        max_length = encoder_outputs.size(1)

        for encoder_output in encoder_outputs:
            decoded_seq = self._decode(encoder_output, max_length)
            token_list.append(decoded_seq)

        token_list = torch.stack(token_list, dim=1).transpose(0, 1)

        return token_list


if __name__ == "__main__":
    # Define your input, input_lengths, targets, and target_lengths tensors
    input_size = 13  # Example input dimension
    num_classes = 27  # Example number of classes

    import numpy as np

    path = "preprocessed_data/TRAIN/DR2/FAJW0/SA2.npy"
    mfcc_data = np.load(path)
    mfcc_tensor = torch.Tensor(mfcc_data)
    inputs = mfcc_tensor.unsqueeze(0)  # Add batch dimension
    model = RNNTransducer(num_classes=num_classes, input_size=input_size)
    predictions = model.recognize(inputs)
    print(predictions)
