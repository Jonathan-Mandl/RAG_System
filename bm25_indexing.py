import json
import bm25s  # Your BM25 library
import numpy as np
from lemmatize_text import *
from bidi.algorithm import get_display
from utils import *

# JSONL file produced by your lemmatization script

def combine_lemmatized_page(doc_obj):
    """
    doc_obj is a dict with:
      doc_id: str
      lemmatized_title: list of strings
      sections: list of {lemmatized_section_title: [...], lemmatized_section_body: [...]}
    Returns a single string containing the entire doc's lemmas.
    """
    # 1) Combine the doc-level title tokens
    doc_text = " ".join(doc_obj.get("lemmatized_title", []))
    
    # 2) Combine each section title + body
    for sec in doc_obj.get("sections", []):
        title_tokens = sec.get("lemmatized_section_title", [])
        body_tokens  = sec.get("lemmatized_section_body", [])
        
        # Add to the doc text
        doc_text += " " + " ".join(title_tokens)
        doc_text += " " + " ".join(body_tokens)
    
    # doc_text is now one giant string of lemmas for the entire doc
    return doc_text


def section_lemmatized_page(doc_obj):
    """
    doc_obj is a dict with:
      doc_id: str
      lemmatized_title: list of strings
      sections: list of {lemmatized_section_title: [...], lemmatized_section_body: [...]}
    Returns a list of tuples, each containing (section_title, section_text_with_titles).
    """
    sections = []
    # 1) Combine the doc-level title tokens
    doc_title_text = " ".join(doc_obj.get("lemmatized_title", []))
    
    # 2) Combine each section title + body
    for sec_id, sec in enumerate(doc_obj.get("sections", [])):
        title_tokens = sec.get("lemmatized_section_title", [])
        body_tokens  = sec.get("lemmatized_section_body", [])
        
        # Add to the doc text
        sec_title_text = " ".join(title_tokens)
        sec_body_text = doc_title_text + " " + sec_title_text + " " + " ".join(body_tokens)
        
        sections.append((sec_id, sec_body_text))
    
    # sections is a list with all the sections, to each section the doc title and section title was added for improved indexing
    return sections

def separate_units(lammatized_json):

    all_units = []
    
    all_sectioned_units = {}

    with open(lammatized_json, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)  # e.g. {"doc_id": "...", "lemmatized_title": [...], "sections": [...]}

            doc_id = obj["doc_id"]
            
            full_page = combine_lemmatized_page(obj)
            
            # we also return sectioned units
            sectioned_doc = section_lemmatized_page(obj)
            all_sectioned_units.update({doc_id: sectioned_doc})

            all_units.append((doc_id,full_page))

    return all_units, all_sectioned_units

class BM25():
    @classmethod
    def from_file(cls, lemmatized_json):
        new_index = BM25()
        index_units, sectioned_units = separate_units(lemmatized_json)

        new_index.index_units = index_units
        new_index.sectioned_units = sectioned_units
        
        # We only need the text portion for BM25 indexing
        bm25_texts = [u[1] for u in index_units]  # the second element is the combined text

        index = bm25s.BM25()                     # Initialize BM25
        bm25_tokens = bm25s.tokenize(bm25_texts) # Tokenize each text (BM25 style)
        index.index(bm25_tokens)  
        
        print(f"Built BM25 index with {len(index_units)} 'section-documents'!")
        
        new_index.index = index
        return new_index
    
    @classmethod
    def from_param(cls, texts):
        new_index = BM25()
        
        index = bm25s.BM25()                     # Initialize BM25
        bm25_tokens = bm25s.tokenize(texts) # Tokenize each text (BM25 style)
        index.index(bm25_tokens)  
        
        print(f"Built BM25 index with {len(texts)} 'section-documents'!")
        
        new_index.index = index
        return new_index
        
    def retrieve_sections(self, query_text, k=5):
        """
        Retrieves top-k BM25 results for the query_text,
        returns a list of (section_index_in_section_units, score).
        """
        ## lemmatize query first
        query_text=lemmatize_with_predict(query_text)

        # Convert query_text to tokens
        query_tokens = bm25s.tokenize(" ".join(query_text))
        
        # Retrieve top k from BM25
        doc_indices, doc_scores = self.index.retrieve(query_tokens, k=k)

        # doc_indices, doc_scores are typically lists-of-lists, 
        # e.g. doc_indices[0] is the top k doc indices for the single query

        return doc_indices[0], doc_scores[0]
    
# we should have to index the documents only once
full_bm25_index = BM25.from_file("lemmatized_content.jsonl")

def find_docs_bm25(queries,k=20):              # Build the BM25 index

    results = []
    for query in queries:
        doc_indices, doc_scores = full_bm25_index.retrieve_sections(query, k)
        
        doc_ids = [full_bm25_index.index_units[idx][0] for idx in doc_indices]

        results.append(doc_ids)

    return results
    
def find_docs_bm25_with_sections(queries, k=20):
    """
    This function finds the most relevant documents, and returns their IDs.
    It also indexes the sections of each of the relevant documents, and returns a sorted list ("ranked list") of these sections, based on the queries.
    """

    top_doc_ids = []
    top_sec_ids = []
    for query in queries:
        doc_indices, doc_scores = full_bm25_index.retrieve_sections(query, k)
        
        doc_ids = [full_bm25_index.index_units[idx][0] for idx in doc_indices]

        top_doc_ids.append(doc_ids)
        
        # index the sections by relevance to query
        bm25_sections = []
        for id in doc_ids:
            sections = full_bm25_index.sectioned_units[id]
            sections = [(id,)+section for section in sections]
            bm25_sections.extend(sections)
                
        bm25_section_texts = [section[2] for section in bm25_sections]
            
        section_index = BM25.from_param(bm25_section_texts)
        
        sec_indices, sec_scores = section_index.retrieve_sections(query, len(bm25_sections))
        
        ranked_sections_ids = [bm25_sections[sec_id][0:2] for sec_id in sec_indices]
        
        top_sec_ids.append(ranked_sections_ids)

    return top_doc_ids, top_sec_ids
    

if __name__ == "__main__":

    docs_path = "indexed_content.jsonl"

    docs_map = read_original_docs(docs_path)

    # Example usage
    user_query = "מהן הזכויות שלי לאחר שפיטרו אותי לא כהוגן במהלך חופשת הלידה שלי"  # A Hebrew query

    results = full_bm25_index.retrieve_sections(user_query, k=5)
    
    print(get_display(f"Top 5 results for query:{user_query}"))
    for doc_idx, score in results:
        # doc_idx corresponds to section_units[doc_idx]
        doc_id = full_bm25_index.index_units[doc_idx][0]
        docs_data = docs_map[doc_id]
        doc_str  = json.dumps(docs_data, ensure_ascii=False) 
        print(get_display(f"Match score: {str(score)} \n {doc_str}"))

