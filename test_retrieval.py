from src.retrieval.retriever import QdrantRetriever

def test_query():
    retriever = QdrantRetriever(collection_name="rag_challenge")
    
    # Đặt một câu hỏi liên quan đến YOLOv10
    user_query = "YOLOv10 là gì?"
    
    print(f"Câu hỏi: {user_query}")
    print("Đang tìm kiếm trong cơ sở dữ liệu...")
    print("-" * 50)
    
    results = retriever.search(query=user_query, top_k=3)
    
    if not results:
        print("Không tìm thấy kết quả nào.")
        return
        
    for i, res in enumerate(results):
        print(f"\n[Kết quả {i+1}] - File: {res['source']} - Độ chính xác (Score): {res['score']:.4f}")
        # Chỉ in 200 ký tự đầu tiên cho đỡ rối mắt
        print(f"Nội dung: {res['content'][:200]}...")

if __name__ == "__main__":
    test_query()