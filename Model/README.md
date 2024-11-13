# Text Chunk Retrieval with BGEM3 and FAISS

---
This code retrieves the most relevant document chunk for a given query, using a chunking strategy with the BGEM3 model, FAISS similarity search, and reranking. It is designed for handling large texts by dividing them into manageable chunks, encoding them with a pre-trained model, and efficiently searching for the most relevant pieces of text.

## Requirements

Python 3.10+

Dependencies:

pprint
faiss
numpy
FlagEmbedding (includes BGEM3FlagModel and FlagReranker)
langchain_text_splitters


---

## Code Overview

The code consists of two main functions:

1. get_text_chunks():

This function splits a text document into smaller, overlapping chunks using RecursiveCharacterTextSplitter.
chunk_size and chunk_overlap parameters allow control over chunk length and the overlap between chunks.

2. bge_retrieve():

This function retrieves the most relevant document chunk for a given query using BGEM3 embeddings and FAISS similarity search.

Steps:
1. Filters the documents specified by the source parameter.
2. Splits each document into chunks and maps each chunk back to its original document key.
3. Encodes the query and document chunks using the BGEM3 model.
4. Uses FAISS to retrieve the top 5 similar chunks based on cosine similarity.
5. Reranks the top chunks using FlagReranker to find the best match.
6. Returns the key of the document containing the best matching chunk.

---
## Usage

1. Split Text: Use get_text_chunks to divide long documents into chunks for easier processing.
2. Retrieve Relevant Chunk: Pass a query to bge_retrieve to find the most relevant chunk in the provided documents.
