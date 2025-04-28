# import datetime
# from pathlib import Path
#
# MODEL_NAME = "sbert"
# TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#
# BASE_OUTPUT_DIR = Path("outputs") / MODEL_NAME / f"run_{TIMESTAMP}"
# RAW_DATA_DIR = Path("data")
# CHECKPOINT_DIR = BASE_OUTPUT_DIR / "checkpoints"
# METRICS_DIR = BASE_OUTPUT_DIR / "metrics"
# PREPROCESSED_DIR = BASE_OUTPUT_DIR / "preprocessed"
# TOKENIZED_DIR = BASE_OUTPUT_DIR / "tokenized"
#
# for dir_path in [CHECKPOINT_DIR, METRICS_DIR, PREPROCESSED_DIR, TOKENIZED_DIR]:
#     dir_path.mkdir(parents=True, exist_ok=True)
#
# config = {
#     "model_name": MODEL_NAME,
#     "num_epochs": 3,
#     "batch_size_train": 32,
#     "batch_size_val": 16,
#     "batch_size_test": 16,
#     "learning_rate": 2e-5,
#     "temperature": 0.07,
#     "max_length": 128,
#     "num_workers": 4,
#     "embed_dim": 128,
#
#
#     # directories
#     "base_output_dir": BASE_OUTPUT_DIR,
#     "raw_data_dir": RAW_DATA_DIR,
#     "checkpoint_dir": CHECKPOINT_DIR,
#     "metrics_dir": METRICS_DIR,
#     "preprocessed_dir": PREPROCESSED_DIR,
#     "tokenized_dir": TOKENIZED_DIR,
#
#     # Tokenization settings
#     "tokenizer_name": "sentence-transformers/all-MiniLM-L12-v2",
#     "tokenizer_max_length": 128,
#     "tokenizer_batch_size": 1024
# }
