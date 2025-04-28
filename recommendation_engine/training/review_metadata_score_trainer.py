import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np

from recommendation_engine.configs.config_review_meta_score import config
from recommendation_engine.dataset.meta_review_score_dataset import MetaReviewScoreDataset
from recommendation_engine.models.review_metadata_score_model import ReviewMetadataScoreModel
from recommendation_engine.utils.data_fetch import get_data_files
from recommendation_engine.utils.metrics import save_metrics


class MetaReviewTrainer:
    """
    Trainer for MetaReviewScoreModel using custom PyTorch Dataset.
    """

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config["device"])
        self.tokenizer = BertTokenizer.from_pretrained(config["tokenizer_name"])

        self.checkpoint_dir = config["checkpoint_dir"]
        self.metrics_dir = config["metrics_dir"]

        # Dataset
        data_files = get_data_files(config["preprocessed_dir"] / "splits")
        self.train_dataset = MetaReviewScoreDataset(data_files["train"], self.tokenizer, config["max_length"])
        self.val_dataset = MetaReviewScoreDataset(data_files["val"], self.tokenizer, config["max_length"])
        self.test_dataset = MetaReviewScoreDataset(data_files["test"], self.tokenizer, config["max_length"])

        # DataLoaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=config["batch_size_train"],
                                       shuffle=True, num_workers=config["num_workers"])
        self.val_loader = DataLoader(self.val_dataset, batch_size=config["batch_size_val"],
                                     shuffle=False, num_workers=config["num_workers"])
        self.test_loader = DataLoader(self.test_dataset, batch_size=config["batch_size_test"],
                                      shuffle=False, num_workers=config["num_workers"])

        # Model
        self.model = ReviewMetadataScoreModel(
            bert_model_name=config["bert_model_name"],
            meta_vocab_size=config["meta_vocab_size"]
        ).to(self.device)

        # Loss & Optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config["learning_rate"],
                                     weight_decay=config["weight_decay"])
        total_steps = config["num_epochs"] * len(self.train_loader)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=int(0.1 * total_steps),
                                                         num_training_steps=total_steps)
        self.metrics_log = []

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        for batch in tqdm(self.train_loader, desc="Training"):
            self.optimizer.zero_grad()
            inputs = {k: v.to(self.device) for k, v in batch.items() if k != "rating"}
            target = batch["rating"].to(self.device)

            output, _ = self.model(**inputs)
            output = output.squeeze(-1)
            loss = self.criterion(output, target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["max_grad_norm"])
            self.optimizer.step()
            self.scheduler.step()

            running_loss += loss.item() * inputs["review_input_ids"].size(0)

        epoch_loss = running_loss / len(self.train_loader.dataset)
        return epoch_loss

    def evaluate(self, loader, split="val"):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Evaluating {split}"):
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != "rating"}
                target = batch["rating"].to(self.device)

                output, _ = self.model(**inputs)
                output = output.squeeze(-1)
                loss = self.criterion(output, target)
                running_loss += loss.item() * inputs["review_input_ids"].size(0)

                all_preds.extend(output.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        epoch_loss = running_loss / len(loader.dataset)
        rmse = np.sqrt(np.mean((np.array(all_preds) - np.array(all_targets)) ** 2))
        return epoch_loss, rmse

    def train(self):
        best_val_loss = float('inf')
        for epoch in range(self.config["num_epochs"]):
            print(f"\nEpoch {epoch + 1}/{self.config['num_epochs']}")
            train_loss = self.train_epoch()
            val_loss, val_rmse = self.evaluate(self.val_loader, split="val")

            print(f"Training Loss: {train_loss:.4f}")
            print(f"Validation Loss: {val_loss:.4f} | RMSE: {val_rmse:.4f}")

            self.metrics_log.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_rmse": val_rmse
            })

            # Save checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(self.checkpoint_dir, "best_model.pth")
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"Best model saved to {checkpoint_path}.")

        # Evaluate on test set
        test_loss, test_rmse = self.evaluate(self.test_loader, split="test")
        print(f"\nTest Loss: {test_loss:.4f} | Test RMSE: {test_rmse:.4f}")
        self.metrics_log.append({
            "epoch": "test",
            "test_loss": test_loss,
            "test_rmse": test_rmse
        })

        # Save metrics
        save_metrics(self.metrics_log, self.metrics_dir)
        print(f"Training complete. Metrics saved at {self.metrics_dir}/metrics.csv")


if __name__ == "__main__":
    trainer = MetaReviewTrainer(config)
    trainer.train()
