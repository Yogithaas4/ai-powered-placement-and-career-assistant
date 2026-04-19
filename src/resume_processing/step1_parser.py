"""
step1_parser.py
---------------
Extracts raw text from resume files.
Supports: .docx, .pdf, .txt
Output  : list of dicts → {filename, raw_text, file_type}
"""

import os
from docx import Document
from pdfminer.high_level import extract_text as pdf_extract_text
from tqdm import tqdm


SUPPORTED_EXTENSIONS = {".docx", ".pdf", ".txt"}


def parse_docx(filepath: str) -> str:
    """Extract text from a .docx file."""
    doc = Document(filepath)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def parse_pdf(filepath: str) -> str:
    """Extract text from a .pdf file using pdfminer."""
    text = pdf_extract_text(filepath)
    if not text:
        return ""
    # Clean up excessive whitespace from PDF extraction
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def parse_txt(filepath: str) -> str:
    """Read plain text file."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()


def parse_file(filepath: str) -> str:
    """Route to the correct parser based on file extension."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".docx":
        return parse_docx(filepath)
    elif ext == ".pdf":
        return parse_pdf(filepath)
    elif ext == ".txt":
        return parse_txt(filepath)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def parse_all_resumes(folder_path: str) -> list:
    """
    Parse all supported resume files in a folder.
    Returns list of dicts: {filename, raw_text, file_type}
    """
    all_files = [
        f for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ]

    if not all_files:
        print(f"[!] No supported files found in '{folder_path}'")
        print(f"    Supported formats: {SUPPORTED_EXTENSIONS}")
        return []

    print(f"[+] Found {len(all_files)} resume file(s) in '{folder_path}'")

    results = []
    for filename in tqdm(all_files, desc="Parsing"):
        filepath = os.path.join(folder_path, filename)
        ext = os.path.splitext(filename)[1].lower()
        try:
            raw_text = parse_file(filepath)
            if raw_text:
                results.append({
                    "filename" : filename,
                    "raw_text" : raw_text,
                    "file_type": ext.lstrip(".")
                })
            else:
                print(f"  [SKIP] Empty content: {filename}")
        except Exception as e:
            print(f"  [ERROR] {filename}: {e}")

    print(f"[+] Successfully parsed: {len(results)}/{len(all_files)}")
    return results


# ── quick test ──────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else "data/resumes"
    parsed = parse_all_resumes(folder)
    if parsed:
        r = parsed[0]
        print(f"\n--- Preview: {r['filename']} ({r['file_type']}) ---")
        print(r["raw_text"][:600])
