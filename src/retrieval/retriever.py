from qdrant_client import QdrantClient
from src.ingestion.embedder import JinaEmbedder
from typing import List, Dict

class QdrantRetriever:
    def __init__(self, collection_name: str = "rag-challenge"):
        """
        Khởi tạo kết nối tới DB và Embedding Model
        """
        self.client = QdrantClient("localhost", port=6333)
        self.collection_name = collection_name
        self.embedder = JinaEmbedder()

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Tìm kiếm Vecto Search cơ bản và trả về  top_k kết quả tương đồng nhất
        """
        if not query.strip():
            return []
        
        # Convert a question's user to vecto (get it to a list)
        try:
            query_vecto = self.embedder.embed_batch([query])[0]
        except Exception as e:
            print(f"[Error] Have an error while embed a question")
            return []
        
        # Search in Qdrant
        try:
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vecto,
                limit=top_k
            )
        except Exception as e:
            print(f"[Error] Have an error while query Qdrant")
            return []
        
        # Format a result back to LLM for easy to read
        results = []
        for hit in search_result:
            results.append({
                "score": hit.score, #cosine similarity (càng gần 1 càng tốt)
                "content": hit.payload.get("text", ""),
                "source": hit.payload.get("source", "unknow"),
                "chunk_id": hit.payload.get("chunk_id", -1)
            })
        
        return results