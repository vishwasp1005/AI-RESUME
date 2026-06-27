import re

def clean_text(text: str) -> str:
    text = re.sub(r'[^\x00-\x7f]', r' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text