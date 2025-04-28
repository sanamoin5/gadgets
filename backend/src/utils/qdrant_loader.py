from src.database.qdrant_manager import QdrantManager

qdrant_manager = None


async def load_qdrant_manager():
    global qdrant_manager
    qdrant_manager = QdrantManager()
    print("Qdrant Manager loaded and ready")
