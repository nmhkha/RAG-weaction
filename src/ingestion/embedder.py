from src.core.config import settings
import requests
from typing import List

class JinaEmbedder:
    def __init__(self, api_key: str = None):
        # Get key from .env
        self.api_key = settings.jina_api_key
        self.url = "https://api.jina.ai/v1/embeddings"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Gọi API để chuyển danh sách các chuỗi text thành vecto
        """
        if not texts:
            return []
        
        payload = {
            "model": "jina-embeddings-v3",
            "task": "retrieval.passage", # Jina v3 yêu cầu xác định task (document hay query)
            "dimensions": 1024,
            "late_chunking": False,
            "input": texts
        }

        try:
            response = requests.post(self.url, headers=self.headers, json=payload)
            response.raise_for_status()
            data = response.json()

            # Extract vecto array from response
            embeddings = [item["embedding"] for item in data["data"]]
            return embeddings
        
        except requests.exceptions.RequestException:
            print(f"[Error] Fail to call Jina API")
            return []