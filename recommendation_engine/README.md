# Amazon Product Recommendation Engine

This repository contains the full training pipeline for two deep learning models used in product recommendation and scoring:

- **DualEncoderSBERT** – A dual encoder model using SBERT for product metadata and reviews (contrastive learning setup).
- **ReviewMetadataScoreModel** – A metadata and review fusion model to predict product scores.

#  Component Flow

---

## **Data Collection & Preparation**
- **Source:** Amazon Product Data (scraped & public metadata)
- **Categories:** Tech products, gaming consoles, webcams, etc.
- **Cleaning:** Filter relevant categories, preprocess metadata & reviews

---

## **Model Training (Dual Encoder)**
- **Model:** Dual SBERT Encoder with Contrastive Learning
- **Training Flow:**
  1. Product metadata and reviews → Encoded using two parallel SBERT models.
  2. InfoNCE loss used to align product data & user preferences in the embedding space.
  3. Model checkpoints saved.
  4. Final model converted to TorchScript (.jit file) for optimized inference.

---

##  **Serving & Inference Flow**
- **Product Embeddings:**
  - Precomputed product embeddings using one encoder.
  - Stored in Qdrant Vector Database for fast similarity search.

- **User Embedding:**
  - At runtime, user preferences are encoded using the second encoder.

- **Similarity Search:**
  - User embedding is matched against product embeddings in Qdrant DB.
  - Top-N similar products are recommended based on **cosine similarity**.

---

## **API Layer**
- **Function:**  
  Accepts user query → Encodes preferences → Fetches similar products from Qdrant → Returns recommendations

---

 **End-to-End Flow:**  
**User preferences → API request → Encoding → Similarity Search → Product Recommendations**

---

---

## Project Structure

```
recommendation_engine/
├── configs/               # Model and training configurations
├── data/                  # Raw and processed datasets
├── dataset/               # Dataset classes
├── experiments/           # Any experiment-specific files
├── models/                # Model architectures
├── outputs/               # All training outputs (checkpoints, metrics)
├── preprocessing/         # Data preprocessing logic
├── scripts/               # Analysis & visualization scripts
├── training/              # Model training scripts
├── utils/                 # Utility functions
├── run_data_preprocessing.py # Entry-point to run preprocessing
├── requirements.txt
└── README.md
```

---

## Workflow & Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 2. Data Preprocessing

Preprocessing includes:
- Cleaning raw data
- Splitting into train/val/test sets
- Saving filtered and preprocessed datasets

To preprocess the data:

```bash
python run_data_preprocessing.py
```

Preprocessed data will be saved in the `data/preprocessed` directory.

---

### 3. Configuration

Before training, **update the respective config file**:

For **SBERT Dual Encoder**:
```
configs/config_sbert_mps.py
configs/config_sbert_nv.py
```
For **Review-Metadata Score Model**:
```
configs/config_review_meta_score.py
```

Set your hyperparameters, directories, and device in the config files.

---

### 4. Model Training

Ensure preprocessing is complete and config is correctly updated.

**To train SBERT model:**
```bash
python training/sbert_trainer.py
```

**To train Review-Metadata Score model:**
```bash
python training/review_metadata_score_trainer.py
```

Training will also evaluate on validation and test sets after each epoch.
The following will be automatically saved in `outputs/`:
- Best model checkpoints
- Training & validation metrics

---

### 5. Data & Output Visualization

For **data analysis** and **training curves visualization**, use the following notebooks:

- `scripts/plot_metrics.ipynb` → Visualize model loss, RMSE, Recall@1, etc.
- `scripts/visualize_data.ipynb` → For raw data inspection and distributions

---

## Outputs

All model artifacts, metrics, and logs will be stored under:

```
outputs/
├── sbert/
├── review_meta_score/
```
Each run will have its own timestamped directory inside these folders, containing:
- Model checkpoints
- Metrics (CSV)
- Training logs

---

##  Notes

- This repo does not include inference code.