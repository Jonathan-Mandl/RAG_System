import gensim.downloader as dl
import json
import numpy as np
import os
import torch
from sentence_transformers import SentenceTransformer
import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from bidi.algorithm import get_display

DOCS_PATH = "indexed_content.jsonl"

model = SentenceTransformer("sentence-transformers/LaBSE")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")
model.to(device)
model.eval()


def index(docs):
    
    units = separate_units(docs)

    vecs = []
    doc_ids = []

    for doc_id, unit in tqdm.tqdm(units, total=len(units), desc="Indexing"):
        avgVec = text_to_vec(unit)
        vecs.append(avgVec)
        doc_ids.append(doc_id)
    
    vecs = np.array(vecs)
    np.save(open("vecs-sent-transformers.npy", "wb"), vecs)
    
    doc_ids = np.array(doc_ids)
    np.save(open("doc_ids-sent-transformers.npy", "wb"), doc_ids)


def separate_units(docs):
    """
    Splits each doc into (doc_id, combined_text)
    combining doc's title & each section's title/body.
    """
    all_units = []
    for doc in docs:
        doc_id = doc["doc_id"]

        if isinstance(doc["title"], list):
            doc_title = " ".join(doc["title"])
        else:
            doc_title = doc["title"]
        
        for sec in doc.get("sections", []):
            if isinstance(sec["section_title"], list):
                sec_title = " ".join(sec["section_title"])
            else:
                sec_title = sec["section_title"]

            if isinstance(sec["section_body"], list):
                sec_body = " ".join(sec["section_body"])
            else:
                sec_body = sec["section_body"]
            
            combined_text = f"{doc_title} {sec_title} {sec_body}"
            all_units.append((doc_id, combined_text))

    return all_units


def text_to_vec(text):
    embedding = model.encode(text)
    vecText = np.array(embedding)
    return vecText


def most_similar(vecs, queryVec):
    """
    Returns (section_indices, section_scores) for *all* sections 
    in descending order of similarity.
    """
    scores = cosine_similarity(vecs, queryVec.reshape(1, -1)).squeeze()
    # Sort all sections by descending score
    section_indices = np.argsort(-scores)
    section_scores = scores[section_indices]
    return section_indices, section_scores


def retrieve_and_deduplicate(vecs, doc_ids, queryVec, k=5):
    """
    Retrieve top sections, then deduplicate so you only get one doc_id.
    Returns a list of (doc_id, best_score).
    """
    section_indices, section_scores = most_similar(vecs,queryVec)
    
    docid_to_score = {}
    
    # Collect up to k unique doc_ids
    for idx, section_idx in enumerate(section_indices):
        doc_id = doc_ids[section_idx]
        score = section_scores[idx]
        
        # If this doc is not seen yet, or we found a better section
        if doc_id not in docid_to_score or score > docid_to_score[doc_id]:
            docid_to_score[doc_id] = score
        
        # Stop if we have at least k unique doc_ids
        if len(docid_to_score) >= k:
            break
    
    # Sort the doc_id -> score map by descending score
    deduped_sorted = sorted(docid_to_score.items(), key=lambda x: x[1], reverse=True)
    
    # Take top-k from the deduplicated list
    final_docs = deduped_sorted[:k]
    return final_docs


def find_docs_sentence_transformer(queries,k=20):

    # Load docs
    with open(DOCS_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    docs = [json.loads(doc.strip()) for doc in lines]

    docs_map = {d['doc_id']: d for d in docs}

    # Build index if we haven't
    if not os.path.exists("vecs-sent-transformers.npy"):
        index(docs)
    
    # Load embeddings
    loadedVecs = np.load(open("vecs-sent-transformers.npy", "rb"))
    loadedDocIds = np.load(open("doc_ids-sent-transformers.npy", "rb"))

    results =[]
    for query in queries: 
        queryVec = text_to_vec(query)
        top_docs = retrieve_and_deduplicate(loadedVecs, loadedDocIds, queryVec, k=k)
        top_doc_ids = [doc[0] for doc in top_docs]
        results.append(top_doc_ids)

    return results


if __name__ == '__main__':

    with open(DOCS_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    docs = [json.loads(doc.strip()) for doc in lines]
    docs_map = {d['doc_id']: d for d in docs}
    
    # Build index if not done already
    if not os.path.exists("vecs-sent-transformers.npy"):
        index(docs)
    
    # Load embeddings
    loadedVecs = np.load(open("vecs-sent-transformers.npy", "rb"))
    loadedDocIds = np.load(open("doc_ids-sent-transformers.npy", "rb"))
    
    query = "מהן הזכויות שלי לאחר שפיטרו אותי לא כהוגן במהלך חופשת הלידה שלי"
    queryVec = text_to_vec(query)

    # Retrieve top 5 doc_ids with deduplication
    top_docs = retrieve_and_deduplicate(loadedVecs, loadedDocIds, queryVec, k=5)

    print(get_display(f"Top 5 doc IDs for query: {query}"))
    for rank, (doc_id, score) in enumerate(top_docs, start=1):
        print(f"Rank {rank}: doc_id={doc_id}, score={score}")
        doc_str = json.dumps(docs_map[doc_id], ensure_ascii=False)
        print(get_display(doc_str))
