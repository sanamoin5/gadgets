import os
import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(merged_csv, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                  seed=42, output_prefix="dataset"):
    """
    Split the merged dataset into train, validation, and test CSV files.
    Saves the split datasets to the specified output directory.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the merged dataset
    df = pd.read_csv(merged_csv)
    print(f"Loaded merged dataset: {df.shape}")

    # Split into train and temp (val + test)
    train_df, temp_df = train_test_split(df, test_size=(1 - train_ratio), random_state=seed)

    # Split temp into val and test
    val_df, test_df = train_test_split(temp_df, test_size=test_ratio / (val_ratio + test_ratio), random_state=seed)

    # File paths
    train_path = os.path.join(output_dir, f"{output_prefix}_train.csv")
    val_path = os.path.join(output_dir, f"{output_prefix}_val.csv")
    test_path = os.path.join(output_dir, f"{output_prefix}_test.csv")

    # Save to CSV
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Dataset split complete:")
    print(f"  Train → {train_df.shape} → {train_path}")
    print(f"  Validation → {val_df.shape} → {val_path}")
    print(f"  Test → {test_df.shape} → {test_path}")
