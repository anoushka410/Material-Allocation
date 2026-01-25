from datasets import load_dataset
import pandas as pd

print("Loading FreshRetailNet-50K dataset (4.5M rows)...")
print("This may take a few minutes on first run (downloads ~500MB)...\n")

# Load the dataset using HuggingFace datasets library
# This handles the download and caching automatically
dataset = load_dataset("Dingdong-Inc/FreshRetailNet-50K", split="train")

print(f"Loaded {len(dataset):,} rows")
print(f"Columns: {dataset.column_names}\n")

# Convert to pandas DataFrame
print("Converting to pandas DataFrame...")
df = dataset.to_pandas()

# Save to CSV
output_file = "freshretailnet_full.csv"
print(f"Saving to {output_file}...")
df.to_csv(output_file, index=False)

print(f"\nDone! Saved {len(df):,} rows to {output_file}")

import os
file_size_gb = os.path.getsize(output_file) / 1e9
print(f"File size: {file_size_gb:.2f} GB")

