import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from torch import nn, optim

from recommendation_engine.configs.config_review_meta_score import config
from recommendation_engine.dataset.meta_review_score_dataset import ProductScoreDataset
from recommendation_engine.experiments.evaluation.product_score_evaluator import evaluate_regressor
from recommendation_engine.models.review_metadata_score_model import ReviewMetadataScoreModel
from recommendation_engine.training.review_metadata_score_trainer import train_regressor_epoch
from utils.logger import write_log
import os
import json

# Init
tokenizer = BertTokenizer.from_pretrained(config["bert_model"])

train_data = ProductScoreDataset(config["train_csv"], tokenizer, config["max_length"])
val_data = ProductScoreDataset(config["val_csv"], tokenizer, config["max_length"])
test_data = ProductScoreDataset(config["test_csv"], tokenizer, config["max_length"])

train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_data, batch_size=config["batch_size"], shuffle=False)
test_loader = DataLoader(test_data, batch_size=config["batch_size"], shuffle=False)

model = ReviewMetadataScoreModel(config["bert_model"], tokenizer.vocab_size)
model.to(config["device"])

criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
total_steps = config["num_epochs"] * len(train_loader)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

# ---- Training Loop ----
best_val_loss = float('inf')
metrics = {}

for epoch in range(config["num_epochs"]):
    train_loss = train_regressor_epoch(model, train_loader, optimizer, scheduler, criterion,
                                       config["device"], epoch)
    val_loss, val_rmse = evaluate_regressor(model, val_loader, criterion, config["device"], epoch, "Val")

    metrics[f"epoch_{epoch+1}"] = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_rmse": val_rmse
    }

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(config["output_dir"], "best_model.pt"))
        write_log(f"New best model at epoch {epoch+1} with Val Loss: {best_val_loss:.4f}")

# ---- Final Test Evaluation ----
test_loss, test_rmse = evaluate_regressor(model, test_loader, criterion, config["device"], epoch, "Test")
metrics["final_test"] = {"test_loss": test_loss, "test_rmse": test_rmse}

# Save metrics
metrics_path = os.path.join(config["metrics_dir"], "metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)

write_log(f"Training Complete. Best Val Loss: {best_val_loss:.4f}")
write_log(f"Test Loss: {test_loss:.4f} | Test RMSE: {test_rmse:.4f}")