import pandas as pd
from torch.utils.data import Dataset


class ContrastiveProductDataset(Dataset):
    """
    Dataset for contrastive learning.
    Returns two views for each product: review text and product metadata+price+rating.
    """

    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        required_cols = ['asin', 'cleaned_review', 'cleaned_metadata', 'rating', 'price']
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Missing column: {col}")
        self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        view1 = str(row['cleaned_review']).strip()
        view2 = f"{str(row['cleaned_metadata']).strip()}. Price: {row['price']}. Rating: {row['rating']}."
        return {"view1": view1, "view2": view2}
