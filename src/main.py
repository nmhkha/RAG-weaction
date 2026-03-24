from fastapi import FastAPI
from src.api.routes import query

# Khởi tạo ứng dụng FastAPI
app = FastAPI(
    title="RAG Pipeline API - WeAction",
    description="Hệ thống RAG Production-grade cho Challenge 1 - Week 2",
    version="1.0.0"
)

# Đăng ký các API Routes
app.include_router(query.router, tags=["RAG Generation"])

# Health check endpoint (Bắt buộc phải có khi deploy Production)
@app.get("/health", tags=["System"])
def health_check():
    return {"status": "healthy", "message": "Hệ thống RAG đang hoạt động tốt!"}

if __name__ == "__main__":
    import uvicorn
    # Chạy server ở port 8000
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)