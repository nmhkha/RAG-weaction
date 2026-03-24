from openai import OpenAI
from typing import List, Dict

class OllamaClient:
    def __init__(self, model_name: str = "qwen2.5:1.5b"):
        """
        Khởi tạo kết nối tới Ollama local thông qua OpenAI Client.
        Ollama mặc định chạy ở port 11434.
        """
        self.client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama" # API key ảo bắt buộc phải có cho thư viện openai
        )
        self.model_name = model_name

    def build_prompt(self, query: str, context_chunks: List[Dict]) -> str:
        """
        Gộp các chunks văn bản lấy từ DB vào Prompt Template.
        """
        # Trích xuất phần text từ các chunks
        context_texts = [f"--- Trích đoạn {i+1} (Nguồn: {chunk['source']}) ---\n{chunk['content']}" 
                         for i, chunk in enumerate(context_chunks)]
        
        context_block = "\n\n".join(context_texts)

        prompt = f"""Bạn là trợ lý AI chuyên nghiệp. Hãy trả lời câu hỏi CHỈ DỰA VÀO ngữ cảnh bên dưới.
Nếu ngữ cảnh không chứa đáp án, hãy nói "Tài liệu hiện tại không đề cập đến thông tin này". Không bịa đặt.

Ngữ cảnh:
{context_block}

Câu hỏi: {query}
"""
        return prompt

    def generate_answer(self, query: str, context_chunks: List[Dict]) -> str:
        """
        Gửi prompt tới Ollama và lấy câu trả lời.
        """
        if not context_chunks:
            return "Tôi không tìm thấy tài liệu nào liên quan đến câu hỏi của bạn."

        full_prompt = self.build_prompt(query, context_chunks)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Bạn là trợ lý AI trả lời câu hỏi dựa trên tài liệu."},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.1 # Giữ temperature thấp để LLM không "sáng tạo" bậy bạ
            )
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"[Error] Lỗi khi gọi Ollama: {e}")
            return "Đã có lỗi xảy ra khi kết nối tới LLM."