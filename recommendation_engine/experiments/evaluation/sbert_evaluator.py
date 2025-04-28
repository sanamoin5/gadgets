import torch
import numpy as np
from tqdm import tqdm
import gc
from pathlib import Path
from torch.utils.data import DataLoader
from functools import partial
from transformers import AutoTokenizer

from recommendation_engine.experiments.config_sbert import config
from recommendation_engine.dataset.contrastive_product_dataset import ContrastiveProductDataset
from recommendation_engine.experiments.sbert_collate import sbert_collate_fn
from recommendation_engine.models.dual_encoder_sbert import DualEncoderSBERT


def evaluate_model_chunked(model_view1, model_view2, dataloader, temperature=0.07):
    device = next(model_view1.parameters()).device
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

    ranks = []
    for i in tqdm(range(all_emb_view1.shape[0]), desc="Computing metrics"):
        query_emb = all_emb_view1[i].unsqueeze(0)
        scores = torch.mm(query_emb, all_emb_view2.t()).squeeze(0)
        sorted_indices = torch.argsort(scores, descending=True)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)

    ranks = np.array(ranks)
    recall_at_1 = np.mean(ranks == 1)
    mrr = np.mean(1.0 / ranks)

    del all_emb_view1, all_emb_view2, query_emb, scores, sorted_indices, ranks
    gc.collect()

    return recall_at_1, mrr


def test_evaluate_sbert():
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = ContrastiveProductDataset(config["test_csv"])
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")
    collate = partial(sbert_collate_fn, tokenizer=tokenizer, max_length=config["max_length"])

    test_loader = DataLoader(test_dataset, batch_size=config["batch_size_test"], shuffle=False,
                             num_workers=config["num_workers"], collate_fn=collate)

    # Load best model
    model_view1 = DualEncoderSBERT().to(device)
    model_view2 = DualEncoderSBERT().to(device)
    model_view1.load_state_dict(torch.load(Path(config["checkpoint_dir"]) / "best_model_view1.pt", map_location=device))
    model_view2.load_state_dict(torch.load(Path(config["checkpoint_dir"]) / "best_model_view2.pt", map_location=device))

    recall_at_1, mrr = evaluate_model_chunked(model_view1, model_view2, test_loader, config["temperature"])
    print(f"\n[Test Evaluation] Recall@1: {recall_at_1:.4f}, MRR: {mrr:.4f}")
    gc.collect()
