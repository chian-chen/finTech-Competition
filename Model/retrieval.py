# Import required libraries
from FlagEmbedding import BGEM3FlagModel  # Import the BGEM3 model for encoding
import faiss  # FAISS library for efficient similarity search
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Import text splitting tool
from FlagEmbedding import FlagReranker  # Import reranking model for reranking passages
import numpy as np  # Import numpy for numerical operations

# Function to split text into manageable chunks
def get_text_chunks(text, chunk_size=800, chunk_overlap=400):
    """
    Splits text into chunks using RecursiveCharacterTextSplitter.
    
    Parameters:
        text (str): The input text to split.
        chunk_size (int): Maximum length of each chunk.
        chunk_overlap (int): Overlap between adjacent chunks.
    
    Returns:
        list: A list of text chunks.
    """
    # Initialize the RecursiveCharacterTextSplitter with specified chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    # Split the text into chunks
    chunks = text_splitter.split_text(text)
    return chunks

# Function for retrieving the most relevant chunk for a query
def bge_retrieve(qs, source, corpus_dict, chunk_size, chunk_overlap):
    """
    Retrieves the most relevant document chunk for a given query using a combination of BGEM3 encoding and FAISS.
    
    Parameters:
        qs (str): Query string.
        source (list): List of document keys to filter the corpus.
        corpus_dict (dict): Dictionary mapping document keys to document text.
        chunk_size (int): Maximum length of each chunk.
        chunk_overlap (int): Overlap between adjacent chunks.

    Returns:
        str: The key of the document containing the best matching chunk.
    """
    # Filter and chunk the corpus based on the provided source keys
    filtered_corpus = [corpus_dict[int(file)] for file in source]
    chunked_corpus = []
    chunk_to_doc_map = {}  # Dictionary to map each chunk to its original document
    
    # Split each document in filtered_corpus into chunks and store mappings
    for index, doc in zip(source, filtered_corpus):
        doc_chunks = get_text_chunks(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunked_corpus.extend(doc_chunks)
        # Map each chunk to its original document index
        for chunk in doc_chunks:
            chunk_to_doc_map[chunk] = index
    
    # Initialize the BGEM3 model for encoding
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    # Encode the query and document chunks to dense vectors
    embeddings_1 = model.encode([qs], batch_size=12, max_length=1024)['dense_vecs']
    embeddings_2 = model.encode(chunked_corpus, batch_size=12, max_length=1024)["dense_vecs"]

    # Initialize FAISS index for similarity search on encoded document chunks
    faiss_index = faiss.IndexFlatIP(embeddings_2.shape[1])
    faiss_index.add(embeddings_2)

    # Perform similarity search to retrieve the top-n similar chunks
    top_n = 5
    score, rank = faiss_index.search(embeddings_1, top_n)

    # Retrieve the top matching chunks based on FAISS results
    top_chunk_idx = rank[0]
    top_chunk = [chunked_corpus[top_idx] for top_idx in top_chunk_idx]

    # Use FlagReranker to rerank the top chunks and select the best match
    reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
    rerank_scores = reranker.compute_score([[qs, passage] for passage in top_chunk], normalize=True)

    # Find the chunk with the highest rerank score
    best_match_idx = np.argmax(rerank_scores)
    best_chunk = top_chunk[best_match_idx]
    
    # Retrieve the original document key associated with the best matching chunk
    doc_index = chunk_to_doc_map[best_chunk]
    return doc_index
