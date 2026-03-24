from langchain_text_splitters import RecursiveCharacterTextSplitter

class TextSplitter:
    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 50):
        """
        Khởi tạo 
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Use RecursiveCharacterTextSplitter of Langchain
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlap = self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""] # Order of priority
        )

    def split_text(self, text: str, source_file: str = "unknow") -> list[dict]:
        """
        Chia nhỏ text và đính kèm Metadata
        Trả về danh sách các dictionary chứa chunk_text và metadata
        """
        if not text:
            return []
        
        chunks = self.splitter.split_text(text)

        # We wrap it in a structure containing metadata
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            processed_chunks.append({
                "content": chunk,
                "metadata": {
                    "source": source_file,
                    "chunk_id": i
                }
            })
        return processed_chunks
