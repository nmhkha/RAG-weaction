from fastapi import APIRouter, HTTPException
from src.api.schemas.models import QueryRequest, QueryResponse
from src.retrieval.retriever import QdrantRetriever
from src.generation.llm_client import OllamaClient

router = APIRouter()

# Khởi tạo các core modules
retriever = QdrantRetriever(collection_name="rag_challenge")

# Lưu ý: Thay "qwen2.5:1.5b" bằng đúng tên model bạn đang có trong Ollama local nhé!
llm_client = OllamaClient(model_name="qwen2.5:1.5b") 

@router.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    try:
        # 1. Lấy thông tin từ Qdrant
        results = retriever.search(query=request.query, top_k=request.top_k)
        
        # 2. Ràng buộc An toàn: Không có ngữ cảnh -> Từ chối trả lời
        if not results:
            return QueryResponse(
                answer="Xin lỗi, tôi không tìm thấy thông tin nào trong tài liệu cung cấp để trả lời câu hỏi này.",
                sources=[]
            )
        
        # 3. Đưa cho Ollama tổng hợp câu trả lời
        answer = llm_client.generate_answer(query=request.query, context_chunks=results)
        
        # 4. Trích xuất nguồn tài liệu (Sources)
        sources = [
            {
                "source": res["source"],
                "chunk_id": res["chunk_id"],
                "score": res["score"]
            } for res in results
        ]
        
        return QueryResponse(answer=answer, sources=sources)
        
    except Exception as e:
        print(f"[API Error] {e}")
        raise HTTPException(status_code=500, detail="Lỗi xử lý hệ thống RAG.")