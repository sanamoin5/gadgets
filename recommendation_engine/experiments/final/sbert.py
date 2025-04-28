import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from tqdm import tqdm
import numpy as np
import random
import os

# Set device: use MPS if available on MacBook Pro, else CUDA, else CPU.
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
print(device)


# ---------------------------
# 1. Contrastive Dataset Definition
# ---------------------------
class ContrastiveProductDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        required_cols = ['asin', 'cleaned_review', 'cleaned_metadata', 'rating', 'price']
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"CSV file is missing required column: {col}")
        # Shuffle the data initially
        self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # View 1: cleaned review text (user perspective)
        view1 = str(row['cleaned_review']).strip()
        # View 2: combine metadata with price and rating (product details)
        view2 = f"{str(row['cleaned_metadata']).strip()}. Price: {row['price']}. Rating: {row['rating']}."
        return {"view1": view1, "view2": view2}


def my_collate_fn(batch, tokenizer, max_length=128):
    view1_texts = [item["view1"] for item in batch]
    view2_texts = [item["view2"] for item in batch]

    encoded_view1 = tokenizer(
        view1_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    encoded_view2 = tokenizer(
        view2_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    return {
        "view1_input_ids": encoded_view1["input_ids"],
        "view1_attention_mask": encoded_view1["attention_mask"],
        "view2_input_ids": encoded_view2["input_ids"],
        "view2_attention_mask": encoded_view2["attention_mask"],
    }


# ---------------------------
# 2. Model Architecture: Dual Encoder with SBERT
# ---------------------------
class DualEncoderSBERT(nn.Module):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L12-v2", embed_dim=128):
        """
        Uses SBERT as a base model. The chosen model outputs 384-dimensional embeddings.
        We then add a projection head mapping to a lower-dimensional space (e.g., 128 dims).
        """
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Load SBERT-style model
        self.encoder = AutoModel.from_pretrained(model_name)

        # Projection head: maps from 384 (MiniLM) to embed_dim.
        self.projection = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use the first token's embedding as the sentence representation (like [CLS])
        cls_emb = outputs.last_hidden_state[:, 0]  # shape: (batch, hidden_size)
        proj = self.projection(cls_emb)
        # Normalize embeddings for cosine similarity
        return F.normalize(proj, p=2, dim=1)



# ---------------------------
# 3. Contrastive Loss (InfoNCE)
# ---------------------------
def info_nce_loss(embeddings1, embeddings2, temperature=0.07):
    """
    Computes InfoNCE loss between two sets of embeddings.
    For each sample, the positive pair is at the same index,
    while other samples in the batch are treated as negatives.
    """
    logits = torch.mm(embeddings1, embeddings2.t()) / temperature
    labels = torch.arange(logits.shape[0]).to(logits.device)
    loss = F.cross_entropy(logits, labels)
    return loss


import numpy as np
import gc
from tqdm import tqdm
def evaluate_model_batchwise(model_view1, model_view2, dataloader, batch_size=512):
    """
    Memory-efficient and much faster evaluation.
    Computes Recall@1 and MRR in batch mode.
    """
    model_view1.eval()
    model_view2.eval()

    all_emb_view1 = []
    all_emb_view2 = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating embeddings"):
            view1_ids = batch["view1_input_ids"].to(device)
            view1_mask = batch["view1_attention_mask"].to(device)
            view2_ids = batch["view2_input_ids"].to(device)
            view2_mask = batch["view2_attention_mask"].to(device)

            emb1 = model_view1(view1_ids, view1_mask)
            emb2 = model_view2(view2_ids, view2_mask)

            all_emb_view1.append(emb1.cpu())
            all_emb_view2.append(emb2.cpu())

    all_emb_view1 = torch.cat(all_emb_view1, dim=0)
    all_emb_view2 = torch.cat(all_emb_view2, dim=0)

    # ---- Batch-wise similarity computation ----
    ranks = []
    num_samples = all_emb_view1.shape[0]

    for i in tqdm(range(0, num_samples, batch_size), desc="Computing metrics"):
        batch_emb = all_emb_view1[i:i+batch_size]  # (batch, dim)
        scores = torch.mm(batch_emb, all_emb_view2.t())  # (batch, N)

        sorted_indices = torch.argsort(scores, dim=1, descending=True)
        for j in range(batch_emb.shape[0]):
            idx = i + j
            rank = (sorted_indices[j] == idx).nonzero(as_tuple=True)[0].item() + 1
            ranks.append(rank)

    ranks = np.array(ranks)
    recall_at_1 = np.mean(ranks == 1)
    mrr = np.mean(1.0 / ranks)

    # Cleanup
    del all_emb_view1, all_emb_view2, batch_emb, scores, sorted_indices, ranks
    gc.collect()

    return recall_at_1, mrr



from functools import partial
from torch.optim import AdamW
import gc


def train_and_evaluate(train_csv, val_csv, epochs=3, batch_size_train=32, batch_size_val=16, lr=2e-5,
                       temperature=0.07, max_length=128, num_workers=4):
    # Create datasets and dataloaders for train and validation splits.
    train_dataset = ContrastiveProductDataset(train_csv)
    val_dataset = ContrastiveProductDataset(val_csv)

    # Initialize tokenizer for collate function.
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")

    collate = partial(my_collate_fn, tokenizer=tokenizer, max_length=max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate
    )

    # Initialize two models for the two views.
    model_view1 = DualEncoderSBERT(model_name="sentence-transformers/all-MiniLM-L12-v2").to(device)
    model_view2 = DualEncoderSBERT(model_name="sentence-transformers/all-MiniLM-L12-v2").to(device)

    optimizer = AdamW(list(model_view1.parameters()) + list(model_view2.parameters()), lr=lr)
    resume_checkpoint = True
    # ---- Resume from checkpoint ----
    if resume_checkpoint:
        model_view1.load_state_dict(torch.load(f"best_model_view1_1.pt"))
        model_view2.load_state_dict(torch.load(f"best_model_view2_1.pt"))
        print(f"Resumed models from epoch 1")

        # Optional: Resume optimizer (if you saved it)
        # optimizer.load_state_dict(torch.load(f"{config['checkpoint_dir']}/optimizer_{resume_epoch}.pt"))

        # Optional: Evaluate immediately
        recall_at_1, mrr = evaluate_model_batchwise(model_view1, model_view2, val_loader)
        print(f"Validation after resuming: Recall@1={recall_at_1:.4f}, MRR={mrr:.4f}")

    best_recall = 0.0
    best_epoch = 0

    for epoch in range(1, epochs):
        model_view1.train()
        model_view2.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
            view1_ids = batch["view1_input_ids"].to(device)
            view1_mask = batch["view1_attention_mask"].to(device)
            view2_ids = batch["view2_input_ids"].to(device)
            view2_mask = batch["view2_attention_mask"].to(device)

            emb1 = model_view1(view1_ids, view1_mask)
            emb2 = model_view2(view2_ids, view2_mask)

            loss = info_nce_loss(emb1, emb2, temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        torch.save(model_view1.state_dict(), f"best_model_view1_{epoch+1}.pt")
        torch.save(model_view2.state_dict(), f"best_model_view2_{epoch+1}.pt")
        print(f"Checkpoint saved at epoch {epoch + 1}")
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} - Training Loss: {avg_loss:.4f}")

        # Evaluate on validation set after each epoch
        recall_at_1, mrr = evaluate_model_batchwise(model_view1, model_view2, val_loader)
        print(f"Epoch {epoch + 1} - Validation Recall@1: {recall_at_1:.4f}, MRR: {mrr:.4f}")

        with open("best_metrics.txt", "w") as f:
            f.write(f"Epoch: {epoch}\nRecall@1: {recall_at_1:.4f}\nMRR: {mrr:.4f}\n")

        if recall_at_1 > best_recall:
            best_recall = recall_at_1
            best_epoch = epoch + 1
            torch.save(model_view1.state_dict(), "best_model_view1.pt")
            torch.save(model_view2.state_dict(), "best_model_view2.pt")
            print(f"New best model found at epoch {epoch + 1} with Recall@1: {best_recall:.4f}")

        gc.collect()
    print(f"Training complete. Best model at epoch {best_epoch} with Recall@1: {best_recall:.4f}")


def test_evaluate(test_csv, max_length=128, batch_size=16, num_workers=4):
    # Load test dataset
    test_dataset = ContrastiveProductDataset(test_csv)
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")
    collate = partial(my_collate_fn, tokenizer=tokenizer, max_length=max_length)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate
    )

    # Load the best models saved from training.
    model_view1 = DualEncoderSBERT(model_name="sentence-transformers/all-MiniLM-L12-v2").to(device)
    model_view2 = DualEncoderSBERT(model_name="sentence-transformers/all-MiniLM-L12-v2").to(device)
    model_view1.load_state_dict(torch.load("best_model_view1.pt", map_location=device))
    model_view2.load_state_dict(torch.load("best_model_view2.pt", map_location=device))

    recall_at_1, mrr = evaluate_model_batchwise(model_view1, model_view2, test_loader)
    print(f"Test Evaluation - Recall@1: {recall_at_1:.4f}, MRR: {mrr:.4f}")






from multiprocessing import freeze_support



if __name__ == '__main__':
    freeze_support()
    train_csv = '/Users/sanamoin/Documents/sites/gadgets/recommendation_engine/data/filtered_splits/electronics_train.csv'
    val_csv = '/Users/sanamoin/Documents/sites/gadgets/recommendation_engine/data/filtered_splits/electronics_val.csv'
    test_csv = '/Users/sanamoin/Documents/sites/gadgets/recommendation_engine/data/filtered_splits/electronics_test.csv'
    # Train and evaluate on validation set.
    train_and_evaluate(train_csv, val_csv, epochs=3, batch_size_train=32, batch_size_val=16, lr=2e-5,
                       temperature=0.07, max_length=128, num_workers=4)

    # Finally, evaluate on the test set using the best saved model.
    test_evaluate(test_csv, max_length=128, batch_size=16, num_workers=4)