import os
import pandas as pd


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


if __name__ == "__main__":
    path_to_train_csv = "data/train_data.csv"
    path_to_test_csv = "data/test_data.csv"
    train_file_paths = "preprocessed_data/train.csv"
    test_file_paths = "preprocessed_data/test.csv"
    extract_data(path_to_train_csv, train_file_paths)
    extract_data(path_to_test_csv, test_file_paths)
