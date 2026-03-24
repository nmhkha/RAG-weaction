import sys
import os
# Đảm bảo Python hiểu thư mục gốc của project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.document_loader import DocumentLoader
from src.ingestion.splitter import TextSplitter
from src.ingestion.embedder import JinaEmbedder
from src.ingestion.indexer import QdrantIndexer
from pathlib import Path

def run_ingestion():
    # 1. Quét toàn bộ file PDF trong thư mục data/raw/
    data_dir = Path("data/raw")
    pdf_files = list(data_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("Không tìm thấy file PDF nào trong data/raw/")
        return
        
    # Khởi tạo các module
    splitter = TextSplitter(chunk_size=512, chunk_overlap=50)
    embedder = JinaEmbedder()
    indexer = QdrantIndexer(collection_name="rag_challenge")

    # 2. Vòng lặp xử lý từng file
    for pdf_path in pdf_files:
        print(f"Đang xử lý: {pdf_path.name}")
        
        # Bước A: Load & Parse
        loader = DocumentLoader(str(pdf_path))
        raw_text = loader.load_and_parse()
        
        # Bước B: Chunking
        chunks = splitter.split_text(raw_text, source_file=pdf_path.name)
        print(f" -> Cắt được {len(chunks)} chunks.")
        
        # Bước C: Embedding (Tách list text ra từ list dict)
        texts_to_embed = [chunk["content"] for chunk in chunks]
        embeddings = embedder.embed_batch(texts_to_embed)
        
        # Bước D: Indexing vào Qdrant
        if embeddings:
            indexer.index_batch(chunks, embeddings)
        
    print("HOÀN TẤT INGESTION!")

if __name__ == "__main__":
    run_ingestion()