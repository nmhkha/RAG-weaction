import re
from pathlib import Path
import pymupdf4llm

class DocumentLoader:
    def __init__(self, file_path:str):
        """
        Hàm khởi tạo
        """
        self.file_path = Path(file_path)
    
    def clean_text(self, text: str):
        """
        Hàm dọn "rác" trong text
        """
        if not text:
            return ""
        
        #1. Remove extra spaces and tabs (convert multiple spaces into a single space)
        text = re.sub(r'[ \t]+', ' ', text)

        #2. Delete any extra blank lines (if there are 3 or more consecutive blank lines, merge them into 2 lines)
        text = re.sub(r'n{3,}', '\n\n', text)

        #Logic to remove URL, char Unicoe

        return text.strip()
    
    def load_and_parse(self) -> str:
        """
        Đọc file PDF, parse thành Markdown và dọn dẹp text
        """
        # Error: File doesn't exist
        if not self.file_path.exists():
            print(f"[Warning] The file doesn't exist: {self.file_path}")
            return ""
        try:
            # Parse PDF -> Markdown
            md_text = pymupdf4llm.to_markdown(str(self.file_path))

            # Clean text before return
            clean_md = self.clean_text(md_text)
            return clean_md
        
        except Exception as e:
            # The file is encrypted or has a corrupted format
            # Log the error and skip this file for pipeline running 
            print(f"[Error] Have the error while processing file{self.file_path}. Details: {str(e)}")
            return ""