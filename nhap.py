from src.ingestion.document_loader import DocumentLoader

def test_pdf_parsing():
    pdf_path = "data/raw/YOLOv10_Tutorials.pdf"
    loader = DocumentLoader(pdf_path)
    markdown_text = loader.load_and_parse()

    if not markdown_text:
        print("Fail")
        return
    
    print(f"Tong ki tu: {len(markdown_text)}")
    print("-" * 50)
    print(markdown_text[:500])

if __name__ == "__main__":
    test_pdf_parsing()
