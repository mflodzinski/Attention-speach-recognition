import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from HTKFeat import MFCC_HTK


def filter_paths_by_extension(
    csv_file_path: str, extension: str, column_name="path_from_data_dir"
) -> pd.Series:
    df = pd.read_csv(csv_file_path)
    paths = df[column_name]
    folder_name = csv_file_path.split("/")[0]
    filtered_paths = paths[paths.str.endswith(extension)]
    filtered_paths = folder_name + "/" + filtered_paths
    return filtered_paths


def _get_file_path_to_save(file_path: str) -> str:
    file_path_to_save = "preprocessed_" + file_path
    file_path_to_save = file_path_to_save.replace(".WAV", ".npy")
    return file_path_to_save


def preprocess_audio_files(audio_file_paths_df: pd.Series) -> None:
    mfcc_htk = MFCC_HTK()

    for audio_file_path in audio_file_paths_df:
        file_path_to_save = _get_file_path_to_save(audio_file_path)
        print(file_path_to_save)
        signal = mfcc_htk.load_raw_signal(audio_file_path)
        mfccs = mfcc_htk.get_feats(signal)

        os.makedirs(os.path.dirname(file_path_to_save), exist_ok=True)
        np.save(file_path_to_save, mfccs)


if __name__ == "__main__":
    data_folders = ["data/train_data.csv", "data/test_data.csv"]
    for data_folder in data_folders:
        audio_file_paths_df = filter_paths_by_extension(data_folder, ".WAV")
        preprocess_audio_files(audio_file_paths_df)
