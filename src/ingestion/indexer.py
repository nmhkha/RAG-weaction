from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from typing import List, Dict
import uuid

class QdrantIndexer:
    def __init__(self, collection_name: str = "rag_challenge"):
        # Connect to Qdrant while running to Docker
        self.client = QdrantClient("localhost", port=6333)
        self.collection_name = collection_name
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """
        Tạo collection, set dimension = 1024 (do Jina v3 trả về 1024)
        """
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
            )
            print(f"Have created new collection: {self.collection_name}")

    def index_batch(self, chunks: List[Dict], embeddings: List[List[float]]):
        if len(chunks) != len(embeddings):    
            print("[Error] The number of text and vecto don't match")
            return
        
        points = []
        for i in range(len(chunks)):
            # Qdrant requires the ID to be a UUID or a large integer
            point_id = str(uuid.uuid4())

            payload = {
                "text": chunks[i]["content"],
                "source": chunks[i]["metadata"].get("source", "unknow"),
                "chunk_id": chunks[i]["metadata"].get("chunk_id", -1) 
                }
            
            points.append(
                PointStruct(id=point_id, vector=embeddings[i], payload=payload)
            )

        # Batch upload
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print(f"Have saved success {len(points)} vectors to Qdrant.")