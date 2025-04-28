from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
import os


def tokenize_and_save_tensors(csv_file, output_dir, tokenizer_name, max_length=128, batch_size=1024):
    """
    Tokenize review and metadata text and save as PyTorch tensors.
    """
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} rows from {csv_file}")

    review_input_ids, review_attention = [], []
    metadata_input_ids, metadata_attention = [], []

    # in batches
    for i in tqdm(range(0, len(df), batch_size), desc=f"Tokenizing {os.path.basename(csv_file)}"):
        batch = df.iloc[i:i + batch_size]

        #  view1: user review, view2: metadata + price + rating
        review_texts = batch["cleaned_review"].astype(str).tolist()
        metadata_texts = (
                batch["cleaned_metadata"].astype(str) +
                ". Price: " + batch["price"].astype(str) +
                ". Rating: " + batch["rating"].astype(str) + "."
        ).tolist()

        tokenized_view1 = tokenizer(review_texts, padding="max_length", truncation=True,
                                    max_length=max_length, return_tensors="pt")
        tokenized_view2 = tokenizer(metadata_texts, padding="max_length", truncation=True,
                                    max_length=max_length, return_tensors="pt")

        review_input_ids.append(tokenized_view1["input_ids"])
        review_attention.append(tokenized_view1["attention_mask"])
        metadata_input_ids.append(tokenized_view2["input_ids"])
        metadata_attention.append(tokenized_view2["attention_mask"])

    data = {
        "review_input_ids": torch.cat(review_input_ids, dim=0),
        "review_attention_mask": torch.cat(review_attention, dim=0),
        "metadata_input_ids": torch.cat(metadata_input_ids, dim=0),
        "metadata_attention_mask": torch.cat(metadata_attention, dim=0),
    }

    output_path = Path(output_dir) / Path(csv_file).with_suffix("_tokenized.pt").name

    torch.save(data, output_path)
    print(f"Saved tokenized tensors to {output_path}")
