from pathlib import Path
from backend.api import extract_text_from_pdf  # adjust import

pdf_path = r"C:\Users\Stalin Rajkumar\Downloads\resume.pdf"
with open(pdf_path, "rb") as f:
    pdf_bytes = f.read()

text = extract_text_from_pdf(pdf_bytes)
print("Extracted chars:", len(text))
print("Preview:\n", text[:500])