from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import PointStruct, Filter, FieldCondition, MatchValue

from src.configs.vars import VarsConfig


class QdrantManager:
    """
    Manager for interacting with Qdrant DB - storing & searching product embeddings.
    """

    def __init__(self):
        qdrant_url = f"http://{VarsConfig.QDRANT_HOST}:{VarsConfig.QDRANT_HTTP_PORT}"
        self.client = AsyncQdrantClient(url=qdrant_url, api_key=VarsConfig.QDRANT_API_KEY)
        self.collection_name = "product_embeddings"

    async def upsert_embeddings(self, product_embeddings: list):
        """
        Store product embeddings in Qdrant collection.
        Args:
            product_embeddings (list): List of dicts with 'id', 'vector', 'payload'.
        """
        points = [
            PointStruct(
                id=item["id"],
                vector=item["vector"],
                payload=item.get("payload", {})
            ) for item in product_embeddings
        ]
        await self.client.upsert(collection_name=self.collection_name, points=points)

    async def search_similar_products(self, query_vector: list, limit: int = 5):
        """
        Search for similar products given an embedding.
        """
        results = await self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit
        )
        return results

    async def delete_product(self, product_id: int):
        """
        Delete a product vector from Qdrant.
        """
        await self.client.delete(collection_name=self.collection_name, points_selector=[product_id])

    async def filter_search(self, query_vector: list, category: str, limit: int = 5):
        """
        Search with additional filtering (example: category filter).
        """
        filter_condition = Filter(
            must=[
                FieldCondition(key="category", match=MatchValue(value=category))
            ]
        )
        results = await self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=filter_condition,
            limit=limit
        )
        return results
