import gensim.downloader as dl
import json
import numpy as np
import os
import torch
import math
import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer
from transformers import BertTokenizer, BertModel
from bidi.algorithm import get_display

DOCS_PATH = "indexed_content.jsonl"

# Load the Hebrew BERT model (DictaBERT)
tokenizer = BertTokenizer.from_pretrained('dicta-il/dictabert')
model = BertModel.from_pretrained("dicta-il/dictabert", output_hidden_states=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")
model.to(device)
model.eval()


def separate_units(docs):
    """
    Splits each doc into multiple units (doc_id, combined_text).
    Each unit is the doc's title + one section's title & body.
    """
    all_units = []
    for doc in docs:
        doc_id = doc["doc_id"]

        # Possibly the doc's title is stored as a list of tokens
        if isinstance(doc["title"], list):
            doc_title = " ".join(doc["title"])
        else:
            doc_title = doc["title"]

        for sec in doc.get("sections", []):
            # If they are lists of tokens, join them
            if isinstance(sec.get("section_title", []), list):
                sec_title = " ".join(sec["section_title"])
            else:
                sec_title = sec.get("section_title", "")

            if isinstance(sec.get("section_body", []), list):
                sec_body = " ".join(sec["section_body"])
            else:
                sec_body = sec.get("section_body", "")

            combined_text = f"{doc_title} {sec_title} {sec_body}"
            all_units.append((doc_id, combined_text))

    return all_units

def get_text_vector(text, layers=None):
    """
    Converts text to a 768-dim vector by:
      1) Tokenizing
      2) Handling chunking if >512 tokens
      3) Summing last 4 layers
      4) Averaging all tokens => final vector
    """
    if layers is None:
        layers = [-4, -3, -2, -1]
    
    encoded = tokenizer.encode_plus(text, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    
    outputArr = []
    
    # If more than 512 tokens, chunk
    if input_ids.shape[1] > 512:
        n = math.ceil(input_ids.shape[1]/510)
        for i in range(n):
            start_idx = i*510
            end_idx   = min((i+1)*510, input_ids.shape[1])

            chunk_ids = input_ids[0, start_idx:end_idx].unsqueeze(0)
            chunk_mask = attention_mask[0, start_idx:end_idx].unsqueeze(0)

            with torch.no_grad():
                chunk_output = model(
                    input_ids=chunk_ids,
                    attention_mask=chunk_mask,
                    output_hidden_states=True
                )
            outputArr.append(chunk_output)
    else:
        # Single chunk
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        outputArr.append(output)

    # Merge chunk outputs
    vectorArr = []
    for out in outputArr:
        states = out.hidden_states
        sum_layers = torch.stack([states[i] for i in layers]).sum(0).squeeze(0)
        # Average all tokens
        avg_chunk_vec = sum_layers.mean(dim=0)  # shape [768]
        vectorArr.append(avg_chunk_vec.cpu())

    # Now average across chunks
    if len(vectorArr) == 1:
        text_vector = vectorArr[0]
    else:
        text_vector = torch.stack(vectorArr, dim=0).mean(dim=0)
    
    return text_vector.numpy()

def text_to_vec(text):
    vec = get_text_vector(text)
    return vec

def index(docs):
    """
    Creates vecs-bert.npy and doc_ids-bert.npy for all 'units' (doc sections).
    """
    units = separate_units(docs)

    vecs = []
    doc_ids = []

    for doc_id, unit_text in tqdm.tqdm(units, total=len(units), desc="Indexing"):
        embedding = text_to_vec(unit_text)
        vecs.append(embedding)
        doc_ids.append(doc_id)
    
    vecs = np.array(vecs)
    np.save("vecs-bert.npy", vecs)
    
    doc_ids = np.array(doc_ids)
    np.save("doc_ids-bert.npy", doc_ids)

def most_similar_all(vecs, doc_ids, queryVec):
    """
    Returns ALL sections, sorted by descending similarity score.
    """
    scores = cosine_similarity(vecs, queryVec.reshape(1, -1)).squeeze()
    # Sort all sections by descending score
    section_indices = np.argsort(-scores)
    section_scores = scores[section_indices]
    return section_indices, section_scores

def retrieve_and_deduplicate(vecs, doc_ids, query, k=5):
    """
    1) Encode the query.
    2) Sort all sections by similarity, descending.
    3) Keep the BEST (highest-scoring) section for each doc_id.
    4) Stop once we have k unique docs.
    """
    queryVec = text_to_vec(query)
    section_indices, section_scores = most_similar_all(vecs, doc_ids, queryVec)

    docid_to_best_score = {}
    
    # Collect top k doc_ids, deduplicated
    for idx, section_idx in enumerate(section_indices):
        this_doc_id = doc_ids[section_idx]
        score = section_scores[idx]
        # If this doc hasn't been added or we found a better-scoring section
        if (this_doc_id not in docid_to_best_score) or (score > docid_to_best_score[this_doc_id]):
            docid_to_best_score[this_doc_id] = score
        if len(docid_to_best_score) >= k:
            break

    # Sort doc_id => best_score by descending score
    deduped_sorted = sorted(docid_to_best_score.items(), key=lambda x: x[1], reverse=True)
    return deduped_sorted[:k]

def find_docs_bert(queries,k=20):

    # Load docs
    with open(DOCS_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    docs = [json.loads(doc.strip()) for doc in lines]


    # Build index if we haven't
    if not os.path.exists("vecs-bert.npy"):
        index(docs)

    # Load the vectors
    loadedVecs = np.load("vecs-bert.npy")
    loadedDocIds = np.load("doc_ids-bert.npy")

    results =[]
    for query in queries: 
        top_docs = retrieve_and_deduplicate(loadedVecs, loadedDocIds, query, k=k)
        doc_ids = [doc[0] for doc in top_docs]
        results.append(doc_ids)
    return results

if __name__ == '__main__':
    # Load docs
    with open(DOCS_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    docs = [json.loads(doc.strip()) for doc in lines]

    docs_map = {d['doc_id']: d for d in docs}

    # Build index if we haven't
    if not os.path.exists("vecs-bert.npy"):
        index(docs)

    # Load the vectors
    loadedVecs = np.load("vecs-bert.npy")
    loadedDocIds = np.load("doc_ids-bert.npy")

    # Example query
    query =  "מהן הזכויות שלי לאחר שפיטרו אותי לא כהוגן במהלך חופשת הלידה שלי"

    top_k = 5
    top_docs = retrieve_and_deduplicate(loadedVecs, loadedDocIds, query, k=top_k)

    print(get_display(f"Top {top_k} doc IDs for query: {query}"))
    for rank, (doc_id, score) in enumerate(top_docs, start=1):
        print(f"Rank {rank} => doc_id={doc_id}, score={score}")
        doc_json = docs_map[doc_id]
        # Show entire doc or partial
        print(get_display(json.dumps(doc_json, ensure_ascii=False)))
