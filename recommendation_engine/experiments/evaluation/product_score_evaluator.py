import torch
from tqdm import tqdm

from recommendation_engine.utils.logger import write_log
from recommendation_engine.utils.metrics import compute_rmse


def evaluate_regressor(model, dataloader, criterion, device, epoch, split, log_dir):
    model.eval()
    all_preds, all_targets = [], []
    running_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {split}"):
            inputs = {k: v.to(device) for k, v in batch.items() if k not in ["rating"]}
            targets = batch["rating"].to(device)
            outputs, _ = model(**inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * targets.size(0)
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    rmse = compute_rmse(all_preds, all_targets)

    write_log(f"Epoch {epoch+1} - {split} Loss: {epoch_loss:.4f} | RMSE: {rmse:.4f}", log_dir)
    return epoch_loss, rmse