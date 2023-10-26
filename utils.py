from typing import Union
from datetime import datetime
from torch.nn import Module
from pathlib import Path
import torch
import os
import pandas as pd


def get_formated_date() -> str:
    """Used to generate time stamp
    Returns:
        str: a formated string represnt the current time stap
    """
    t = datetime.now()
    return f"{t.year}{t.month}{t.day}-{t.hour}{t.minute}{t.second}"


def load_stat_dict(model: Module, model_path: Union[str, Path]) -> None:
    """Used to load the weigths for the given model
    Args:
        model (Module): the model to load the weights into
        model_path (Union[str, Path]): tha path of the saved weigths
    """
    model.load_state_dict(torch.load(model_path))


def transform_string(text):
    text = text.lower()
    text = "".join(char if char.isalpha() or char.isspace() else "" for char in text)
    text = text.replace(" ", "_")
    return text


def extract_data(input_csv_path, output_csv_path):
    df = pd.read_csv(input_csv_path)
    data_dir = os.path.dirname(input_csv_path)

    audio_paths = []
    texts = []
    durations = []

    for _, row in df.iterrows():
        path_from_data_dir = row["path_from_data_dir"]

        if path_from_data_dir.endswith(".WAV"):
            wav_path = data_dir + "/" + path_from_data_dir
            txt_path = wav_path.replace(".WAV", ".TXT")

            if os.path.exists(txt_path):
                with open(txt_path, "r") as text_file:
                    content = text_file.read().split(" ")
                    end_time = float(content[1]) / 16000
                    text = " ".join(content[2:]).strip()
                    text = transform_string(text)

                audio_paths.append(wav_path)
                texts.append(text)
                durations.append(end_time)

    new_df = pd.DataFrame(
        {"audio_path": audio_paths, "text": texts, "duration": durations}
    )
    new_df.to_csv(output_csv_path, index=False)
