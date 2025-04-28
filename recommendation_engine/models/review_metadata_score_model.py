import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class ReviewMetadataScoreModel(nn.Module):
    """
    Combines review text (via BERT) and product metadata (via embeddings + CNN) to predict product score.
    Fuses both modalities and passes through feedforward layers for final regression output.
    Used for ranking or scoring tasks with mixed data sources.
    """

    def __init__(self, bert_model_name, meta_vocab_size):
        super().__init__()
        self.review_encoder = BertModel.from_pretrained(bert_model_name)  # Extract [CLS] token embedding
        self.review_proj = nn.Linear(self.review_encoder.config.hidden_size, 256)  # transform CLS to 256-dim embedding
        self.review_dropout = nn.Dropout(0.1)

        self.meta_embedding = nn.Embedding(meta_vocab_size, 128, padding_idx=0)  # Categorical embedding
        self.meta_conv1d = nn.Conv1d(128, 128, kernel_size=3, padding=1)  # 1D conv for local feature patterns
        self.meta_pool = nn.AdaptiveMaxPool1d(1)

        self.price_proj = nn.Sequential(nn.Linear(1, 64), nn.ReLU())
        self.meta_price_fusion = nn.Sequential(nn.Linear(128 + 64, 256), nn.ReLU(),
                                               nn.Dropout(0.1))  # Fuse metadata tokens + price

        self.review_meta_fusion = nn.Sequential(nn.Linear(256+256, 256), nn.ReLU(), nn.Dropout(0.1))
        self.score_output = nn.Linear(256, 1)

    def forward(self, review_input_ids, review_attention_mask, meta_input_ids, meta_attention_mask, price):
        review_output = self.review_encoder(review_input_ids, attention_mask=review_attention_mask)
        cls_embedding = review_output.last_hidden_state[:, 0, :]  # [CLS] token embedding
        review_emb = self.review_dropout(self.review_proj(cls_embedding))

        meta_embedded = self.meta_embedding(meta_input_ids).transpose(1, 2)
        meta_conv = F.relu(self.meta_conv1d(meta_embedded))
        meta_pooled = self.meta_pool(meta_conv).squeeze(-1)

        price_vector = self.price_proj(price.view(-1, 1))
        meta_concat = torch.cat([meta_pooled, price_vector], dim=1)
        meta_price_emb = self.meta_price_fusion(meta_concat)

        fused_vector = torch.cat([review_emb, meta_price_emb], dim=1)
        fused_output = self.review_meta_fusion(fused_vector)
        score = self.score_output(fused_output)

        return score.squeeze(-1), fused_output

    @classmethod
    def from_config(cls, config):
        """
        Instantiate model from configuration dictionary.

        Required keys in config:
            - "bert_model_name": Pre-trained BERT model name.
            - "meta_vocab_size": Vocabulary size of metadata tokens.
        """
        return cls(
            bert_model_name=config["bert_model_name"],
            meta_vocab_size=config["meta_vocab_size"]
        )
