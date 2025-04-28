import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np

from recommendation_engine.dataset.contrastive_product_dataset import ContrastiveProductDataset
from recommendation_engine.models.dual_encoder_sbert import DualEncoderSBERT
from recommendation_engine.utils.data_fetch import get_data_files
from recommendation_engine.utils.metrics import save_metrics


def info_nce_loss(emb1, emb2, temperature=0.07):
    logits = torch.mm(emb1, emb2.t()) / temperature
    labels = torch.arange(logits.shape[0]).to(logits.device)
    loss = nn.functional.cross_entropy(logits, labels)
    return loss


class SBERTContrastiveTrainer:
    """
    Trainer for DualEncoderSBERT model.
    Handles data loading, training, evaluation, and metric logging.
    Supports resume training and dynamic config setup.
    """

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config["device"])
        self.tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])

        # load data
        data_files = get_data_files(config["preprocessed_dir"] / "splits/")
        self.train_dataset = ContrastiveProductDataset(data_files["train"])
        self.val_dataset = ContrastiveProductDataset(data_files["val"])
        self.test_dataset = ContrastiveProductDataset(data_files["test"])

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=config["batch_size_train"], shuffle=True,
            num_workers=config["num_workers"], collate_fn=self.collate_fn,
            pin_memory=config["pin_memory"], persistent_workers=config["persistent_workers"]
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=config["batch_size_val"], shuffle=False,
            num_workers=config["num_workers"], collate_fn=self.collate_fn,
            pin_memory=config["pin_memory"], persistent_workers=config["persistent_workers"]
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=config["batch_size_test"], shuffle=False,
            num_workers=config["num_workers"], collate_fn=self.collate_fn,
            pin_memory=config["pin_memory"], persistent_workers=config["persistent_workers"]
        )

        # load models
        self.model_view1 = DualEncoderSBERT.from_config(config).to(self.device)  # for reviews
        self.model_view2 = DualEncoderSBERT.from_config(config).to(self.device)  # for metadata

        # Optimizer
        params = list(self.model_view1.parameters()) + list(self.model_view2.parameters())
        self.optimizer = optim.AdamW(params, lr=config["learning_rate"])

        # If resuming from a certain epoch
        resume_epoch = config["resume_epoch"]
        if resume_epoch > 1:
            prev_epoch = resume_epoch - 1
            self.model_view1.load_state_dict(
                torch.load(os.path.join(config["checkpoint_dir"], f"best_model_view1_{prev_epoch}.pt"),
                           map_location=self.device))
            self.model_view2.load_state_dict(
                torch.load(os.path.join(config["checkpoint_dir"], f"best_model_view2_{prev_epoch}.pt"),
                           map_location=self.device))
            print(f"Resumed models from epoch {prev_epoch}")

        self.metrics_log = []

        self.is_non_blocking = True if self.device == 'cuda' else False

    def collate_fn(self, batch):
        view1_texts = [item["view1"] for item in batch]
        view2_texts = [item["view2"] for item in batch]

        encoded_view1 = self.tokenizer(
            view1_texts, padding=True, truncation=True, max_length=self.config["max_length"], return_tensors="pt"
        )
        encoded_view2 = self.tokenizer(
            view2_texts, padding=True, truncation=True, max_length=self.config["max_length"], return_tensors="pt"
        )

        return {
            "view1_input_ids": encoded_view1["input_ids"],
            "view1_attention_mask": encoded_view1["attention_mask"],
            "view2_input_ids": encoded_view2["input_ids"],
            "view2_attention_mask": encoded_view2["attention_mask"],
        }

    def evaluate_batchwise(self, dataloader, batch_size=512):
        self.model_view1.eval()
        self.model_view2.eval()
        all_emb_view1 = []
        all_emb_view2 = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Generating embeddings"):
                v1_ids = batch["view1_input_ids"].to(self.device)
                v1_mask = batch["view1_attention_mask"].to(self.device)
                v2_ids = batch["view2_input_ids"].to(self.device)
                v2_mask = batch["view2_attention_mask"].to(self.device)

                emb1 = self.model_view1(v1_ids, v1_mask)
                emb2 = self.model_view2(v2_ids, v2_mask)

                all_emb_view1.append(emb1.cpu())
                all_emb_view2.append(emb2.cpu())

        all_emb_view1 = torch.cat(all_emb_view1, dim=0)
        all_emb_view2 = torch.cat(all_emb_view2, dim=0)

        ranks = []
        num_samples = all_emb_view1.shape[0]
        for i in tqdm(range(0, num_samples, batch_size), desc="Computing metrics"):
            batch_emb = all_emb_view1[i:i + batch_size]
            scores = torch.mm(batch_emb.to(self.device), all_emb_view2.to(self.device).t())

            sorted_indices = torch.argsort(scores, dim=1, descending=True)
            for j in range(batch_emb.shape[0]):
                idx = i + j
                rank = (sorted_indices[j] == idx).nonzero(as_tuple=True)[0].item() + 1
                ranks.append(rank)

        ranks = np.array(ranks)
        recall_at_1 = np.mean(ranks == 1)
        mrr = np.mean(1.0 / ranks)

        return recall_at_1, mrr

    def train_epoch(self, epoch):
        self.model_view1.train()
        self.model_view2.train()
        running_loss = 0.0
        for batch in tqdm(self.train_loader, desc=f"Training Epoch {epoch}/{self.config['num_epochs']}"):
            v1_ids = batch["view1_input_ids"].to(self.device, non_blocking=self.is_non_blocking)
            v1_mask = batch["view1_attention_mask"].to(self.device, non_blocking=self.is_non_blocking)
            v2_ids = batch["view2_input_ids"].to(self.device, non_blocking=self.is_non_blocking)
            v2_mask = batch["view2_attention_mask"].to(self.device, non_blocking=self.is_non_blocking)

            emb1 = self.model_view1(v1_ids, v1_mask)
            emb2 = self.model_view2(v2_ids, v2_mask)

            loss = info_nce_loss(emb1, emb2, self.config["temperature"])
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(self.train_loader)
        return avg_loss

    def train(self):
        best_recall = 0.0
        start_epoch = self.config["resume_epoch"]
        for epoch in range(start_epoch, self.config["num_epochs"] + 1):
            train_loss = self.train_epoch(epoch)
            val_recall, val_mrr = self.evaluate_batchwise(self.val_loader)

            print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}")
            print(f"Validation Recall@1: {val_recall:.4f} | MRR: {val_mrr:.4f}")

            self.metrics_log.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_recall@1": val_recall,
                "val_mrr": val_mrr
            })

            # Save checkpoint
            torch.save(self.model_view1.state_dict(),
                       os.path.join(self.config["checkpoint_dir"], f"best_model_view1_{epoch}.pt"))
            torch.save(self.model_view2.state_dict(),
                       os.path.join(self.config["checkpoint_dir"], f"best_model_view2_{epoch}.pt"))
            print(f"Checkpoint saved at epoch {epoch}")

            if val_recall > best_recall:
                best_recall = val_recall
                torch.save(self.model_view1.state_dict(),
                           os.path.join(self.config["checkpoint_dir"], "best_model_view1.pt"))
                torch.save(self.model_view2.state_dict(),
                           os.path.join(self.config["checkpoint_dir"], "best_model_view2.pt"))
                print(f"New best model found at epoch {epoch} with Recall@1: {best_recall:.4f}")

        # Test Evaluation
        test_recall, test_mrr = self.evaluate_batchwise(self.test_loader)
        print(f"\nTest Recall@1: {test_recall:.4f} | Test MRR: {test_mrr:.4f}")
        self.metrics_log.append({
            "epoch": "test",
            "test_recall@1": test_recall,
            "test_mrr": test_mrr
        })

        # Save metrics
        save_metrics(self.metrics_log, self.config["metrics_dir"])
        print(f"Training complete. Metrics saved at {self.config['metrics_dir']}/metrics.csv")


if __name__ == "__main__":

    # from recommendation_engine.configs.config_sbert_mps import config
    from recommendation_engine.configs.config_sbert_nv import config

    trainer = SBERTContrastiveTrainer(config)
    trainer.train()
