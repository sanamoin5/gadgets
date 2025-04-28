import datetime
from pathlib import Path
import torch
import os

PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_NAME = "sbert_mps"
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
    "num_epochs": 3,
    "batch_size_train": 32,
    "batch_size_val": 16,
    "batch_size_test": 16,
    "learning_rate": 2e-5,
    "temperature": 0.07,
    "max_length": 128,
    "num_workers": 4,
    "embed_dim": 128,
    "resume_epoch": 1,
    "pin_memory": False,
    "persistent_workers": False,

    "device": "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu",

    # Directories
    "base_output_dir": BASE_OUTPUT_DIR,
    "data_dir": DATA_DIR,
    "checkpoint_dir": CHECKPOINT_DIR,
    "metrics_dir": METRICS_DIR,
    "preprocessed_dir": PREPROCESSED_DIR,
    "raw_data_dir": RAW_DATA_DIR,
    "tokenized_dir": TOKENIZED_DIR,

    # Tokenizer
    "tokenizer_name": "sentence-transformers/all-MiniLM-L12-v2",
    "tokenizer_max_length": 128,
    "tokenizer_batch_size": 1024,
}
