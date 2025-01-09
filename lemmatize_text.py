import re
import math
import json
import torch
from transformers import AutoModel, AutoTokenizer
import tqdm
from bidi.algorithm import get_display
import os
import chardet

INPUT_JSONL = "indexed_content.jsonl"       # Where we read from
OUTPUT_JSONL = "lemmatized_content.jsonl"   # Where we write final output
MODEL_NAME = "dicta-il/dictabert-lex"       # DictaBERT-lex model name


########################################
# LOAD MODEL
########################################

device = "cuda" if torch.cuda.is_available() else "cpu"
print("CUDA available?", torch.cuda.is_available())
print("Using device:", device)


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
model.to(device)
model.eval()

########################################
# HELPER FUNCTIONS
########################################

def filter_text_for_indexing(text: str) -> str:
    """
    Removes unwanted characters (like punctuation, '#'). 
    Keeps Hebrew letters, English letters, digits, and whitespace.
    Adjust as needed for your scenario.
    """
    pattern = r"[^א-תA-Za-z0-9\s\u0590-\u05FF]"
    cleaned = re.sub(pattern, "", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

def chunk_text(text: str, chunk_size: int = 510) -> list:
    """
    Splits text that might exceed 512 tokens into multiple 'chunks'.
    We'll do a naive approach by splitting on whitespace for now,
    then rejoin until we approximate chunk_size tokens. 
    Alternatively, you can do a more precise approach with tokenize().
    """
    tokens = text.split()
    chunks = []
    
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = " ".join(chunk_tokens)
        chunks.append(chunk_text)
        start = end
    
    return chunks

def lemmatize_with_predict(text: str) -> list:
    """
    1) Filter out punctuation / '#'.
    2) Chunk the text if it might exceed 512 tokens.
    3) Use `model.predict([chunk], tokenizer)` to get lemmas for each chunk.
    4) Return combined list of lemmas for the entire text.
    """
    # Filter text
    cleaned_text = filter_text_for_indexing(text)
    
    # Tokenize once to see if > 512 tokens
    initial_tokens = cleaned_text.split()
    if len(initial_tokens) <= 512:
        # Single chunk is enough
        result = model.predict([cleaned_text], tokenizer)
        # result is e.g. [[[original, lemma], [original, lemma], ...]]
        # We only have 1 chunk, so result[0] is the list of pairs
        pairs = result[0] if result else []
        
        # Extract only lemma from each pair
        lemmas = [pair[1] for pair in pairs]
        # text_for_display = " ".join(lemmas)   # Convert list to a single string
        # print(get_display(text_for_display))
        return lemmas
    else:
        # We do multiple chunks
        chunks = chunk_text(cleaned_text, chunk_size=510)
        
        all_lemmas = []
        for chunk in chunks:
            chunk_result = model.predict([chunk], tokenizer)
            # chunk_result[0] -> list of pairs for this chunk
            pairs = chunk_result[0] if chunk_result else []
            chunk_lemmas = [pair[1] for pair in pairs]
            all_lemmas.extend(chunk_lemmas)
        
        return all_lemmas

########################################
# MAIN
########################################

def main():

    total_lines = 0
    if os.path.exists(INPUT_JSONL):
        with open(INPUT_JSONL, "r", encoding="utf-8") as f_tmp:
            for _ in f_tmp:
                total_lines += 1
    else:
        print(f"File {INPUT_JSONL} not found.")
        return
    
    # We’ll create a new JSONL with the lemmatized text
    output_data = []
    
    with open(INPUT_JSONL, "r", encoding="utf-8",errors = "ignore") as f_in:
        for line in tqdm.tqdm(f_in,total=total_lines,desc="Lemmatizing", unit="docs"):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            doc_id = obj.get("doc_id", "")
            doc_title = obj.get("title", "")
            sections = obj.get("sections", [])

            # Lemmatize title
            title_lemmas = lemmatize_with_predict(doc_title)

            # Lemmatize each section
            new_sections = []
            for sec in sections:
                sec_title = sec.get("section_title", "")
                sec_body = sec.get("section_body", "")
                
                lem_title = lemmatize_with_predict(sec_title)
                lem_body  = lemmatize_with_predict(sec_body)
                
                new_sections.append({
                    "lemmatized_section_title": lem_title,
                    "lemmatized_section_body": lem_body
                })

            new_record = {
                "doc_id": doc_id,
                "lemmatized_title": title_lemmas,
                "sections": new_sections
            }
            output_data.append(new_record)

    # Decide encoding
    encoding_mode = "utf-8"

    with open(OUTPUT_JSONL, "w", encoding=encoding_mode) as f_out:
        for item in output_data:
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"Done! Wrote {len(output_data)} documents to {OUTPUT_JSONL} using {encoding_mode} encoding.")

if __name__ == "__main__":
    main()
