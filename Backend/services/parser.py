import re
import fitz  # PyMuPDF
from utils.logger import get_logger

logger = get_logger(__name__)


def extract_text_from_pdf(file_bytes: bytes, filename: str) -> str:
    """Extract raw text from a PDF file given its bytes."""
    logger.info(f"Parsing PDF: {filename}")
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        raw_text = ""
        for page_num, page in enumerate(doc):
            raw_text += page.get_text("text")
            logger.debug(f"  Page {page_num + 1} extracted.")
        doc.close()
        cleaned = _clean_text(raw_text)
        logger.info(f"PDF parsed successfully – {len(cleaned)} characters extracted.")
        return cleaned
    except Exception as exc:
        logger.error(f"PDF parsing failed for {filename}: {exc}")
        raise RuntimeError(f"Could not parse PDF: {exc}") from exc


def _clean_text(text: str) -> str:
    """Remove noise characters and normalise whitespace."""
    # Replace non-breaking spaces and other Unicode whitespace with regular space
    text = text.replace("\xa0", " ").replace("\u200b", "")
    # Keep alphanumeric, common punctuation, and whitespace; strip the rest
    text = re.sub(r"[^\w\s,./()\-@+#&:;'\"•]", " ", text, flags=re.UNICODE)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    return text.strip()