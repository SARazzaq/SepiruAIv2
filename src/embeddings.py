"""
RAG pipeline using ChromaDB + Ollama nomic-embed-text embeddings.
Converts CSV rows into vectors and retrieves relevant context for chat.
"""

import chromadb
import requests
import pandas as pd
import hashlib
import os
from typing import List


EMBED_MODEL  = "nomic-embed-text"
OLLAMA_HOST  = os.getenv("OLLAMA_HOST", "http://localhost:11434")
COLLECTION   = "csv_data"


def get_embedding(text: str) -> List[float]:
    """Get embedding vector from Ollama nomic-embed-text."""
    response = requests.post(
        f"{OLLAMA_HOST}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=30
    )
    response.raise_for_status()
    return response.json()["embedding"]


def build_vector_store(df: pd.DataFrame, file_hash: str) -> chromadb.Collection:
    """
    Convert each CSV row into a text chunk, embed it,
    and store in ChromaDB. Skips rebuild if same file.
    """
    client = chromadb.Client()

    # Delete old collection if exists
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )

    # Convert rows to text chunks (batch of 100 rows each)
    chunks      = []
    chunk_ids   = []
    metadatas   = []

    columns = list(df.columns)
    batch_size = 5  # rows per chunk

    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i + batch_size]
        lines = []
        for _, row in batch.iterrows():
            line = ", ".join([f"{col}: {row[col]}" for col in columns])
            lines.append(line)
        chunk_text = f"Rows {i+1} to {min(i+batch_size, len(df))}:\n" + "\n".join(lines)
        chunks.append(chunk_text)
        chunk_ids.append(f"chunk_{i}")
        metadatas.append({"rows": f"{i}-{i+batch_size}"})

    # Also add column summary as a special chunk
    summary_chunk = "Dataset columns and types:\n"
    for col in columns:
        dtype = str(df[col].dtype)
        sample = str(df[col].dropna().head(3).tolist())
        summary_chunk += f"- {col} ({dtype}): sample values = {sample}\n"
    chunks.append(summary_chunk)
    chunk_ids.append("summary_chunk")
    metadatas.append({"rows": "summary"})

    # Embed and store in batches
    print(f"Building vector store: {len(chunks)} chunks...")
    embeddings = []
    for chunk in chunks:
        emb = get_embedding(chunk)
        embeddings.append(emb)

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=chunk_ids,
        metadatas=metadatas
    )

    print(f"Vector store ready: {collection.count()} chunks indexed")
    return collection


def retrieve_context(collection: chromadb.Collection,
                     query: str,
                     top_k: int = 5) -> str:
    """
    Embed the user query and retrieve top_k most relevant chunks.
    Returns them as a single context string.
    """
    query_embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count())
    )

    docs = results["documents"][0]
    context = "\n\n".join(docs)
    return context


def get_file_hash(df: pd.DataFrame) -> str:
    """Generate a hash to detect if file has changed."""
    return hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()