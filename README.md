# RAG Pipeline - Challenge 1 (Week 2) - WeAction to Fresher AI Engineer

Dự án này xây dựng một hệ thống Retrieval-Augmented Generation (RAG) hoàn chỉnh đạt chuẩn Production. Hệ thống được thiết kế theo kiến trúc 3 pha (Offline Ingestion, Online Query, Evaluation) với cấu trúc thư mục module hóa, sẵn sàng cho việc mở rộng và bảo trì.

Tài liệu được sử dụng trong dự án là `YOLOv10_Tutorials.pdf` (20 trang) nói về kiến trúc và cách huấn luyện mô hình YOLOv10.

---

##  1. Architecture Diagram (Kiến trúc hệ thống)

Dưới đây là luồng dữ liệu (Data Flow) xuyên suốt 3 pha của hệ thống:

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                       RAG PIPELINE ARCHITECTURE                         │
└─────────────────────────────────────────────────────────────────────────┘

 ╔═══════════════════════════════════════════════════════════════════════╗
 ║ PHASE 1: OFFLINE INGESTION PIPELINE                                   ║
 ║ Raw PDF ──> PyMuPDF4LLM ──> RecursiveChunking ──> Embedding ──> Qdrant║
 ╚═══════════════════════════════════════════════════════════════════════╝
                                    │ (Vectors + Metadata)
                                    ▼
 ╔═══════════════════════════════════════════════════════════════════════╗
 ║ PHASE 2: ONLINE QUERY PIPELINE (FastAPI)                              ║
 ║ User Query ──> Embed Query ──> Vector Search (Qdrant) ──> Top 3 Chunks║
 ║                                                                       ║
 ║ Top 3 Chunks + Prompt Template ──> Ollama (Qwen 1.5B) ──> JSON Output ║
 ╚═══════════════════════════════════════════════════════════════════════╝
                                    │
                                    ▼
 ╔═══════════════════════════════════════════════════════════════════════╗
 ║ PHASE 3: EVALUATION LOOP                                              ║
 ║ Dataset (20 QA) ──> Ragas Framework ──> Faithfulness Score            ║
 ╚═══════════════════════════════════════════════════════════════════════╝
````

-----

## 2. Tech Stack & Lý do chọn (Rationale)

Dự án được tối ưu hóa để chạy hoàn toàn trên môi trường Local (CPU-only / RAM hạn chế) nhưng vẫn giữ form kiến trúc của hệ thống lớn.

| Thành phần | Công nghệ sử dụng | Lý do lựa chọn (Trade-off) |
| :--- | :--- | :--- |
| **Document Parsing** | `PyMuPDF4LLM` | Tốc độ trích xuất cực nhanh (\~0.12s/doc), output ra Markdown sạch sẽ, giữ được cấu trúc heading. Phù hợp làm baseline. |
| **Chunking Strategy**| `RecursiveCharacter` | Cấu hình `512 tokens`, `overlap 50`. Cách chia này bảo toàn ngữ nghĩa tốt hơn cắt cứng (fixed-size), dễ setup. |
| **Embedding Model** | `Jina Embeddings v3` | Hỗ trợ tiếng Việt xuất sắc (MTEB 65.5). Sử dụng qua API (Free tier) để tiết kiệm tài nguyên GPU nội bộ. |
| **Vector Database** | `Qdrant` | Viết bằng Rust nên tốc độ truy xuất cực nhanh. Chạy gọn nhẹ qua Docker. Hỗ trợ Payload Filtering và Hybrid Search cho hướng phát triển sau này. |
| **LLM Engine** | `Ollama` + `Qwen2.5:1.5b`| Cho phép chạy LLM nội bộ (Self-host) không tốn phí API. Qwen 1.5B tuy nhỏ nhưng xử lý tiếng Việt rất mượt. API tương thích 100% chuẩn OpenAI. |
| **API Framework** | `FastAPI` | Chuẩn công nghiệp hiện tại cho Backend AI. Định nghĩa input/output rõ ràng qua Pydantic schemas. Tích hợp sẵn Swagger UI. |
| **Evaluation** | `Ragas` | Framework đo lường RAG tiêu chuẩn. Đo đạc các chỉ số tự động để thay thế việc "test bằng mắt" (eyeballing). |

-----

##  3. Setup Instructions (Hướng dẫn cài đặt)

Hệ thống yêu cầu cài đặt sẵn `Docker` và `Ollama` trên máy host.

**Bước 1: Clone dự án và cài đặt thư viện**

```bash
git clone <your-repo-link>
cd rag-project-weaction
pip install -r requirements.txt
```

**Bước 2: Cấu hình biến môi trường**
Tạo file `.env` ở thư mục gốc và điền thông tin (tham khảo `.env.example`):

```env
JINA_API_KEY=your_jina_api_key_here
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

**Bước 3: Khởi động Hạ tầng (Infrastructure)**

```bash
# 1. Khởi động Qdrant Vector DB
docker compose -f docker/docker-compose.yml up -d

# 2. Đảm bảo Ollama đang chạy ở port 11434 (Mở terminal riêng)
ollama serve
ollama pull qwen2.5:1.5b
```

**Bước 4: Chạy Pipeline Ingestion (Nạp dữ liệu)**

```bash
# Chạy script bóc tách PDF và nạp vào Qdrant
python scripts/ingest.py
```

**Bước 5: Khởi chạy API Server**

```bash
# Bật FastAPI server
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

👉 Truy cập **Swagger UI** để test API: `http://localhost:8000/docs`

-----

## 4. RAGAS Evaluation Results

Để đảm bảo hệ thống không bị ảo giác (hallucination), pipeline được đánh giá qua bộ đề 20 câu hỏi (Ground truth) về mô hình YOLOv10.

  * **Metric đánh giá:** `Faithfulness` (Độ trung thực - LLM có tự bịa thông tin ngoài context hay không).
  * **Kết quả:** **0.7500 (75%)**
  * *Ghi chú về phần cứng:* Do hệ thống deploy Local sử dụng mô hình SLM (`Qwen2.5:1.5B`), mô hình gặp giới hạn trong việc sinh định dạng JSON chuẩn cho Ragas ở hàm đo `Answer Relevancy`. Do đó, luồng đánh giá được tinh chỉnh để tối ưu cho chỉ số quan trọng nhất là `Faithfulness`.

*(Xem chi tiết báo cáo tại file: `eval/results/ragas_score.json`)*

-----

## 📸 5. Screenshots Demo

*(Các hình ảnh minh chứng quá trình hoạt động của hệ thống)*

### 1\. Data Ingestion thành công (Logs)
<img width="1430" height="892" alt="2-query-response" src="https://github.com/user-attachments/assets/00b7d3bf-65fc-4c31-b070-c24dc34412d2" />

### 2\. Giao diện FastAPI & Kết quả RAG (Trả về Answer + Sources)
<img width="900" height="135" alt="1_ingestion_success" src="https://github.com/user-attachments/assets/eaa0b085-d7da-4787-b363-8af0315ece42" />

### 3\. Health Check
<img width="1438" height="506" alt="3-health-check" src="https://github.com/user-attachments/assets/9d8dc497-c908-4e9f-9cca-fc79df2de008" />

### 4\. Kết quả Dashboard Qdrant (Đã nạp Vectors)
<img width="1612" height="385" alt="4-qdrant-dashboard" src="https://github.com/user-attachments/assets/dd394d35-8a44-4b64-92fd-67e7ee0fca7d" />

### 5\. RAGAS evaluation output
<img width="517" height="103" alt="5-ragas-scores" src="https://github.com/user-attachments/assets/76ac77dd-d80d-4028-8d80-8584ccd2218d" />

### 6\. Hạ tầng Qdrant chạy trên Docker
<img width="1239" height="90" alt="6_docker_qdrant" src="https://github.com/user-attachments/assets/4cf052df-2810-43a4-838b-6144a4756fbf" />

```
***
