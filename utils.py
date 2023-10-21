import os


def get_all_file_paths(directory):
    file_paths = []
    for root, _, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)

    return file_paths


def save_file_paths(directory, save_path):
    file_paths = get_all_file_paths(directory)
    with open(save_path, "w") as f:
        for file_path in file_paths:
            f.write(file_path + "\n")


if __name__ == "__main__":
    directory = "preprocessed_data"
    train_file_paths = "preprocessed_data/train_paths.txt"
    test_file_paths = "preprocessed_data/test_paths.txt"
    save_file_paths(directory, train_file_paths)
    save_file_paths(directory, test_file_paths)
