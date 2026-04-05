from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

USELESS_TOPICS = [
    "table of contents index sections",
    "definitions glossary meaning of terms interpretations",
    "about the company our history who we are",
    "registered office address company details cin",
    "irdai registration regulatory information license",
    "grievance redressal complaints escalation",
    "ombudsman contact insurance ombudsman",
    "customer care toll free number website email",
    "please read this document carefully before signing",
    "privacy policy data protection personal information",
    "how to contact us reach support helpline",
    "schedule of benefits summary table",
]

# Build TF-IDF vectorizer on useless topics once
_vectorizer = TfidfVectorizer().fit(USELESS_TOPICS)
_useless_vecs = _vectorizer.transform(USELESS_TOPICS).toarray()

def is_useful_node(node, threshold=0.25) -> bool:  # lower threshold for TF-IDF
    text = node.text.strip()
    role = node.metadata.get("role", "")

    # Fast checks
    if role in {"pageHeader", "pageFooter", "pageNumber"}:
        return False
    if len(text) < 40:
        return False

    # TF-IDF similarity — pure CPU, no downloads
    vec = _vectorizer.transform([text.lower()]).toarray()
    sims = cosine_similarity(vec, _useless_vecs)[0]
    return sims.max() < threshold

import fitz  # pymupdf
import json
import io

def highlight_clause_in_pdf(pdf_bytes: bytes, page_number: int, polygon_json: str) -> bytes:
    """
    Returns PDF bytes with the clause highlighted on the given page.
    polygon_json is the JSON string stored in node metadata.
    """
    try:
        polygon = json.loads(polygon_json)
    except:
        return pdf_bytes

    if not polygon:
        return pdf_bytes

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    # page_number is 1-based
    page = doc[page_number - 1]

    # Azure polygon format: [x1,y1, x2,y2, x3,y3, x4,y4] in inches
    # PyMuPDF uses points (1 inch = 72 points)
    pts = [(polygon[i] * 72, polygon[i+1] * 72) for i in range(0, len(polygon), 2)]

    # Draw highlight rectangle around the polygon
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    rect = fitz.Rect(min(xs), min(ys), max(xs), max(ys))

    # Yellow highlight
    highlight = page.add_highlight_annot(rect)
    highlight.set_colors(stroke=[1, 1, 0])  # yellow
    highlight.update()

    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def jump_to_page_pdf(pdf_bytes: bytes, page_number: int, polygon_json: str) -> tuple[bytes, int]:
    """Returns highlighted PDF and the 0-based page index for display."""
    highlighted = highlight_clause_in_pdf(pdf_bytes, page_number, polygon_json)
    return highlighted, page_number - 1