from pprint import pprint
from FlagEmbedding import BGEM3FlagModel
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
from FlagEmbedding import FlagReranker
import numpy as np


def get_text_chunks(text, chunk_size=800, chunk_overlap=400):
    # Initialize the RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    # Split the text into chunks
    chunks = text_splitter.split_text(text)
    return chunks

def bge_retrieve(qs, source, corpus_dict, chunk_size, chunk_overlap):
    # Filter and chunk the corpus
    filtered_corpus = [corpus_dict[int(file)] for file in source]
    chunked_corpus = []
    chunk_to_doc_map = {}  # Map chunks to their original document
    
    for index, doc in zip(source, filtered_corpus):
        doc_chunks = get_text_chunks(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunked_corpus.extend(doc_chunks)
        # Map each chunk to its document index in filtered_corpus
        for chunk in doc_chunks:
            chunk_to_doc_map[chunk] = index
    

    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    embeddings_1 = model.encode([qs], batch_size=12, max_length=1024)['dense_vecs']
    embeddings_2 = model.encode(chunked_corpus, batch_size=12, max_length=1024)["dense_vecs"]


    # Use FAISS for similarity search
    faiss_index = faiss.IndexFlatIP(embeddings_2.shape[1])
    faiss_index.add(embeddings_2)

    top_n = 5
    score, rank = faiss_index.search(embeddings_1, top_n)

    # Get the index of the top matching chunk
    top_chunk_idx = rank[0]
    top_chunk = [chunked_corpus[top_idx] for top_idx in top_chunk_idx]

    # Use reranker to find the best match among the top 2
    reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
    rerank_scores = reranker.compute_score([[qs, passage] for passage in top_chunk], normalize=True)

    best_match_idx = np.argmax(rerank_scores)
    best_chunk = top_chunk[best_match_idx]
    
    # Retrieve the original document key based on the chunk mapping
    doc_index = chunk_to_doc_map[best_chunk]
    return doc_index