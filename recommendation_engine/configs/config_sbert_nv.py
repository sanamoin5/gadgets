import datetime
from pathlib import Path
import torch
import os

PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_NAME = "sbert_nvidia"
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

BASE_OUTPUT_DIR = PROJECT_ROOT / "outputs" / MODEL_NAME / f"run_{TIMESTAMP}"
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = BASE_OUTPUT_DIR / "checkpoints"
METRICS_DIR = BASE_OUTPUT_DIR / "metrics"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"
RAW_DATA_DIR = DATA_DIR / "raw/amazon_2023"
TOKENIZED_DIR = DATA_DIR / "tokenized"

for dir_path in [CHECKPOINT_DIR, METRICS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

config = {
    "model_name": MODEL_NAME,
    "num_epochs": 7,
    "batch_size_train": 128,
    "batch_size_val": 256,
    "batch_size_test": 256,
    "learning_rate": 2e-5,
    "temperature": 0.07,
    "max_length": 128,
    "num_workers": 12,
    "embed_dim": 128,
    "resume_epoch": 4,
    "pin_memory": True,
    "persistent_workers": True,

    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # Directories
    "base_output_dir": BASE_OUTPUT_DIR,
    "raw_data_dir": RAW_DATA_DIR,
    "checkpoint_dir": CHECKPOINT_DIR,
    "metrics_dir": METRICS_DIR,
    "preprocessed_dir": PREPROCESSED_DIR,

    # Tokenizer
    "tokenizer_name": "sentence-transformers/all-MiniLM-L12-v2"
}
