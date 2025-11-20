import os
import fitz
from docx import Document


def extract_text_from_pdf(file_path):
    """Extract and return text from a PDF file."""
    text = ""
    try:
        with fitz.open(file_path) as pdf:
            for page in pdf:
                page_text = page.get_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    except Exception as e:
        return f"[ERROR] Could not extract PDF: {str(e)}"



def extract_text_from_docx(docx_path):
    """Extract and return text from a DOCX file."""
    text = ""
    try:
        doc = Document(docx_path) 
        for para in doc.paragraphs:
            if para.text.strip():
                text += para.text.strip() + "\n"
        return text.strip()
    except Exception as e:
        return f"[ERROR] Could not extract DOCX: {str(e)}"



def extract_text(file_path):
    """Detect file type and extract text accordingly."""
    if not os.path.exists(file_path):
        return "[ERROR] File not found."
 
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif ext == '.docx':
        return extract_text_from_docx(file_path)
    else:
        return "[ERROR] Unsupported file type. Only PDF and DOCX are supported."





if __name__ == "__main__":
    contract_file = r"D:/Internships/Infosys SpringBoard/ClauseEase/employment_agreement.pdf"
    extracted = extract_text(contract_file)
    
    print("=== Extracted Contract Text ===\n")
    print(extracted[:2000])
