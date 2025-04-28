# src/inference/model_loader.py

import torch
from pathlib import Path

sbert_model = None


def load_sbert_model():
    global sbert_model
    model_path = Path("trained_checkpoints/dual_encoder_sbert_view1_ts.pt")
    sbert_model = torch.jit.load(str(model_path), map_location="cpu")
    sbert_model.eval()
    print("SBERT view1 model loaded")
