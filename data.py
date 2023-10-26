import math
import pandas as pd
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Union, Tuple
from hprams import hprams
from torchaudio.transforms import Resample, MelSpectrogram
import torchaudio


class BaseData:
    def __init__(
        self,
        tokenizer,
        max_len: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_len = max_len

    def _get_padded_aud(
        self,
        aud_path: Union[str, Path],
        max_duration: int,
    ) -> Tensor:
        max_len = 1 + math.ceil(
            max_duration * hprams.data.sampling_rate / hprams.data.hop_length
        )
        aud = self.prepare_audio(aud_path)
        n = max_len - aud.shape[1]
        zeros = torch.zeros(size=(1, n, aud.shape[-1]))
        return torch.cat([zeros, aud], dim=1)

    def _get_padded_tokens(self, text: str) -> Tensor:
        text = self.prepare_text(text)
        tokens = self.tokenizer.tokens2ids(text)
        eos_idx = self.tokenizer.special_tokens[self.tokenizer._eos_key][1]
        tokens.append(eos_idx)
        length = self.max_len - len(tokens)
        pad_idx = self.tokenizer.special_tokens[self.tokenizer._pad_key][1]
        tokens = tokens + [pad_idx] * length
        return torch.LongTensor(tokens)

    def prepocess_lines(self, data: str) -> List[str]:
        return [item.split(hprams.data.sep) for item in data]

    def prepare_audio(self, audio_path: Union[str, Path]) -> Tensor:
        x, sr = torchaudio.load(audio_path, normalize=True)
        x = Resample(sr, hprams.data.sampling_rate)(x)
        x = MelSpectrogram(
            hprams.data.sampling_rate, n_mels=hprams.data.n_mel_channels
        )(x)
        x = x.permute(0, 2, 1)
        return x

    def prepare_text(self, text: str) -> str:
        text = text.lower()
        text = text.replace(" ", "_")
        text = "".join(char if char.isalpha() else "" for char in text)
        text = text.strip()
        return text


class DataLoader(BaseData):
    def __init__(
        self,
        file_path: Union[str, Path],
        tokenizer,
        batch_size: int,
        max_len: int,
    ) -> None:
        super().__init__(tokenizer, max_len)
        self.batch_size = batch_size
        self.df = pd.read_csv(file_path)
        self.num_examples = len(self.df)
        self.idx = 0

    def __len__(self):
        length = self.num_examples // self.batch_size
        return length + 1 if self.num_examples % self.batch_size > 0 else length

    def get_max_duration(self, start_idx: int, end_idx: int) -> float:
        return self.df[hprams.data.csv_file_keys.duration].iloc[start_idx:end_idx].max()

    def get_audios(self, start_idx: int, end_idx: int) -> Tensor:
        max_duration = self.get_max_duration(start_idx, end_idx)
        result = [
            self._get_padded_aud(path, max_duration)
            for path in self.df[hprams.data.csv_file_keys.path].iloc[start_idx:end_idx]
        ]
        result = torch.stack(result, dim=1)
        return torch.squeeze(result)

    def get_texts(self, start_idx: int, end_idx: int) -> Tuple[Tensor, torch.IntTensor]:
        args = self.df[hprams.data.csv_file_keys.text].iloc[start_idx:end_idx]
        lengths = [len(x) + 1 for x in args.values]
        result = torch.stack([self._get_padded_tokens(text) for text in args], dim=0)
        print(result)
        return result, torch.IntTensor(lengths)

    def __iter__(self):
        self.idx = 0
        while self.idx * self.batch_size < self.num_examples:
            start = self.idx * self.batch_size
            end = min((self.idx + 1) * self.batch_size, self.num_examples)
            self.idx += 1
            yield self.get_audios(start, end), *self.get_texts(start, end)
