import os
import pandas as pd
from tqdm import tqdm


def merge_reviews_with_metadata(review_csv, meta_csv, output_csv, chunk_size=500000):
    """
    Merge cleaned review data with product metadata in chunks based on ASIN.
    Efficiently handles large datasets and saves the merged output to a CSV file.
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    print("Loading metadata into memory...")
    meta_df = pd.read_csv(meta_csv).drop_duplicates(subset=["parent_asin"])
    meta_df.set_index("parent_asin", inplace=True)

    if os.path.exists(output_csv):
        os.remove(output_csv)
    header_written = False

    print("Starting chunked merge...")
    for chunk in pd.read_csv(review_csv, chunksize=chunk_size):
        merged_chunk = chunk.merge(meta_df, left_on="asin", right_index=True, how="inner")
        merged_chunk = merged_chunk[["asin", "cleaned_review", "cleaned_metadata", "rating", "price"]]
        merged_chunk["price"] = pd.to_numeric(merged_chunk["price"], errors="coerce")
        merged_chunk["price"] = merged_chunk["price"].fillna(merged_chunk["price"].median())

        if not header_written:
            merged_chunk.to_csv(output_csv, index=False, mode="w")
            header_written = True
        else:
            merged_chunk.to_csv(output_csv, index=False, header=False, mode="a")
    print(f"Merged data saved to {output_csv}")
