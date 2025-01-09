
import json

def combine_page(doc_obj):
    """
    If you want to combine the entire doc as a single string, you can use this.
    Currently unused in the indexing pipeline, but provided if needed.
    """
    # Combine doc-level title tokens
    doc_text = doc_obj.get("title", [])
    # Combine each section title + body
    for sec in doc_obj.get("sections", []):
        title_tokens = sec.get("section_title", [])
        body_tokens  = sec.get("section_body", [])
        doc_text += " " + title_tokens
        doc_text += " " + body_tokens
    return doc_text

def read_original_docs(docs_path):
    # Read line by line
    with open(docs_path, "r", encoding="utf-8") as f:
        docs_raw = f.readlines()

    # Parse each line as JSON
    docs_map = {}
    for line in docs_raw:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        doc_id = obj['doc_id']
        docs_map[doc_id] = obj

    return docs_map