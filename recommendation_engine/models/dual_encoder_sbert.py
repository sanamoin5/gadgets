import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class DualEncoderSBERT(nn.Module):
    """
    Dual Encoder model using SBERT backbone.
    It encodes user reviews and product metadata into a shared embedding space and projects them to a lower dimension.
    Useful for recommendation, similarity search, contrastive learning.
    Keeps it lightweight and fast for retrieval tasks, with projection + normalization for cosine similarity training.


    Args:
        model_name (str): HuggingFace model name.
        embed_dim (int): Dimension of the final projected embeddings.
    """

    def __init__(self, model_name: str, embed_dim: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.projection = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, input_ids, attention_mask):
        """
        Forward pass for encoding input and projecting embeddings.

        Args:
            input_ids (torch.Tensor): Token IDs.
            attention_mask (torch.Tensor): Attention mask.

        Returns:
            torch.Tensor: Normalized projected embeddings.
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0]  # [CLS] token embedding
        proj = self.projection(cls_emb)
        return F.normalize(proj, p=2, dim=1)

    @classmethod
    def from_config(cls, config):
        """
        Instantiate model from config dict
        """
        return cls(
            model_name=config["tokenizer_name"],
            embed_dim=config["embed_dim"]
        )
