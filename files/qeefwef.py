import pandas as pd

# Replace 'your_file.csv' with the actual path to your CSV file
csv_file_path = "files/test.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Calculate the length of each "path" in the DataFrame and find the maximum
max_path_length = df["text"].str.len().max()

print(f"The maximum length of the 'path' column is: {max_path_length}")
