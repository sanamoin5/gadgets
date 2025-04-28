import torch
from datasets import Dataset


class PreTokenizedProductDataset(Dataset):
    def __init__(self, tokenized_file):
        data = torch.load(tokenized_file)
        self.view1_input_ids = data["view1_input_ids"]
        self.view1_attention_mask = data["view1_attention_mask"]
        self.view2_input_ids = data["view2_input_ids"]
        self.view2_attention_mask = data["view2_attention_mask"]

    def __len__(self):
        return self.view1_input_ids.shape[0]

    def __getitem__(self, idx):
        return {
            "view1_input_ids": self.view1_input_ids[idx],
            "view1_attention_mask": self.view1_attention_mask[idx],
            "view2_input_ids": self.view2_input_ids[idx],
            "view2_attention_mask": self.view2_attention_mask[idx],
        }
