import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


class MetaReviewScoreDataset(Dataset):
    def __init__(self, csv_file, tokenizer: BertTokenizer, max_length=256):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        review = str(row["cleaned_review"])
        metadata = str(row["cleaned_metadata"])
        price = float(row["price"])
        rating = float(row["rating"])

        review_enc = self.tokenizer(review, padding="max_length", truncation=True,
                                    max_length=self.max_length, return_tensors="pt")
        meta_enc = self.tokenizer(metadata, padding="max_length", truncation=True,
                                  max_length=self.max_length, return_tensors="pt")

        return {
            "review_input_ids": review_enc["input_ids"].squeeze(0),
            "review_attention_mask": review_enc["attention_mask"].squeeze(0),
            "meta_input_ids": meta_enc["input_ids"].squeeze(0),
            "meta_attention_mask": meta_enc["attention_mask"].squeeze(0),
            "price": torch.tensor(price, dtype=torch.float),
            "rating": torch.tensor(rating, dtype=torch.float)
        }
