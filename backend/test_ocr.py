from pathlib import Path
from backend.api import extract_text_from_pdf, POPPLER_PATH  # adapt import
p = r"C:\Users\Stalin Rajkumar\Downloads\Anushri Rajkumar Resume _compressed.pdf"
with open(p, "rb") as f:
    text = extract_text_from_pdf(f, ocr_min_chars=30, poppler_path=POPPLER_PATH)
    print("TOTAL CHARS:", len(text))
    print(text[:1000])