# rag_system.py
from openai import OpenAI
import tiktoken
import openai
from bm25_indexing import *
import os
from utils import *
from bidi.algorithm import get_display

## initiailize OpenAI API
client = OpenAI()

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

client.api_key = API_KEY

class RAGSystem:

    def __init__(self, corpus_file, index_function, constraint=None, k=5, model='gpt-4o'):

        self.index_function = index_function
        self.k = k
        self.docs_map = read_original_docs(corpus_file)
        self.constraint = constraint
        self.model = model
        self.encoding = tiktoken.encoding_for_model(model)

    def num_tokens_from_string(self, string):
        """Returns the number of tokens in a text string."""
        num_tokens = len(self.encoding.encode(string))
        return num_tokens
    
    def get_top_docs(self,query):
        if self.constraint==None:
            return self.get_top_docs_no_constraint(query)
        else:
            return self.get_top_docs_with_constraint(query)

    def get_top_docs_no_constraint(self,query):

        query = self.enrich_with_llm(query)

        top_doc_ids = self.index_function([query],k=self.k)[0]
        top_docs =[]
        for id in top_doc_ids:
            doc = self.docs_map[id]
            full_page = combine_page(doc)
            top_docs.append(full_page)

        return top_docs, top_doc_ids
    
    def get_top_docs_with_constraint(self, query):
        """
        This func gets the ranked sections of the matched documents and gets as most of them as can fit in the constraint we have.
        The tokenization is based on the OpenAI tokenizer (as we're using OpenAI's LLM) 'tiktoken'.
        """

        query = self.enrich_with_llm(query)

        top_doc_ids, ranked_sections_ids = self.index_function([query], k=self.k)
        top_sections = []
        total_token_length = 0  # section is of the form (section_title, section_body)
        for doc_id, sec_id in ranked_sections_ids[0]: # this is a sorted passtrough over the sections, based on bm25s and the query, it contains a doc_id and sec_id within the doc
            if total_token_length == self.constraint:
                break
            # get the unlemmatized section text
            doc = self.docs_map[doc_id]
            section = doc['sections'][sec_id]
            section_text = section['section_title'] + ' ' + section['section_body']
            # check for constraint
            num_tokens = self.num_tokens_from_string(section_text)
            if total_token_length + num_tokens <= self.constraint and section_text not in top_sections:
                top_sections.append(section_text)
                total_token_length += num_tokens
                
        return top_sections, top_doc_ids
    
    def enrich_with_llm(self,query):

        num_queries = 5

        # Make the API call to generate the response
        completion = client.chat.completions.create(
        model=self.model,
        max_tokens=300,
        messages=[
            {"role": "developer", "content": "אתה עוזר מקצועי המנסח שאלות דומות לשאלה שקיבלת מהשמשתמש"},
            {
                "role": "user",
                "content": f"המר את השאילתה הבאה ל-{num_queries} שאילתות תיאוריות טובות: {query}"
            }
        ]
        )

        # Extract and print the generated answer
        llm_answer = completion.choices[0].message.content.strip()

        llm_answer = query + llm_answer

        return llm_answer

        
    def generate_answer(self, query):

        # Fetch the content of the top documents (Assuming you have access to the documents)
        # For this example, we'll mock document contents. Replace with actual retrieval.
        
        if self.constraint==None:
            documents, ids = self.get_top_docs(query)
        else:
            documents, ids = self.get_top_docs_with_constraint(query)
        
        # Construct the prompt
        prompt = self.construct_prompt(query, documents)
        
        # Make the API call to generate the response
        completion = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "אתה עוזר מקצועי העונה על שאלה בהתבסס על מסמכים שאתה מקבל מהמשתמש"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500 # Adjust based on desired length  
        )
        
        # Extract the response content
        answer = completion.choices[0].message.content.strip()
        answer = answer + f"The documents ids used in this answer are: {ids}"
        
        return answer
        
    
    def construct_prompt(self, query, documents):

        # Combine documents into a single context
        context = "\n\n".join(documents)

        # print("Context")
        # print(get_display(context))
        
        # Construct the prompt
        prompt = (
        f"הנה מספר מסמכים רלוונטיים:\n{context}\n\n"
        f"בהתבסס על המסמכים הללו, ענה על השאלה הבאה בצורה ברורה, תמציתית ומדויקת:\n{query}"
        )
        return prompt


def Query(query, constraint=None):

    if constraint==None:
        RAG = RAGSystem(corpus_file = 'indexed_content.jsonl', index_function=find_docs_bm25, k= 5)
    else:
        RAG = RAGSystem(corpus_file = 'indexed_content.jsonl', index_function=find_docs_bm25_with_sections, constraint=constraint, k= 5)

    answer = RAG.generate_answer(query=query)

    print("LLM answer: ")
    print(get_display(answer))
    
    return answer

if __name__ == '__main__':
    
    query = 'אפשר טלפון למוקד של נפגעי חרדה?'

    Query(query, 1000)

