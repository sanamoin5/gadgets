import torch
from fastapi import APIRouter, HTTPException
import logging

from src.inference.model_loader import sbert_model
from src.utils.qdrant_loader import qdrant_manager

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/sbert/recommend")
async def recommend_products(inputs: dict):
    try:
        input_ids = torch.tensor(inputs["input_ids"]).unsqueeze(0)
        attention_mask = torch.tensor(inputs["attention_mask"]).unsqueeze(0)

        with torch.no_grad():
            embedding = sbert_model(input_ids, attention_mask).squeeze(0).tolist()

        results = await qdrant_manager.search_similar_products(query_vector=embedding, limit=5)

        return {
            "results": [{
                "product_id": hit.payload.get("product_id"),
                "score": hit.score
            } for hit in results]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
