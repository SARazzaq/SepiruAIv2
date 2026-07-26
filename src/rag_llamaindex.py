"""
RAG pipeline using LlamaIndex — indexes CSV data and answers questions
using vector search + LLM synthesis. 100% free & open-source.

Embedding: sentence-transformers (local, no API key)
LLM: routes through existing AIClient (Groq free tier / Ollama)
Vector store: ChromaDB (in-memory)
"""

import pandas as pd
import os
from typing import Optional


def _build_documents(df: pd.DataFrame):
    """Convert DataFrame into LlamaIndex Document objects."""
    from llama_index.core import Document

    docs = []
    columns = list(df.columns)

    # Schema document
    schema_lines = ["Dataset Schema:"]
    for col in columns:
        dtype = str(df[col].dtype)
        n_unique = df[col].nunique()
        sample = df[col].dropna().head(3).tolist()
        schema_lines.append(f"  - {col} ({dtype}) | unique={n_unique} | sample={sample}")
    docs.append(Document(
        text="\n".join(schema_lines),
        metadata={"type": "schema", "chunk_id": "schema"}
    ))

    # Statistics document for numeric columns
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        stats_text = "Numeric Statistics:\n" + df[num_cols].describe().round(3).to_string()
        docs.append(Document(
            text=stats_text,
            metadata={"type": "stats", "chunk_id": "stats"}
        ))

    # Row chunks (5 rows per chunk)
    batch_size = 5
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i + batch_size]
        lines = []
        for _, row in batch.iterrows():
            line = ", ".join(f"{col}: {row[col]}" for col in columns)
            lines.append(line)
        chunk_text = f"Rows {i+1}-{min(i+batch_size, len(df))}:\n" + "\n".join(lines)
        docs.append(Document(
            text=chunk_text,
            metadata={"type": "rows", "chunk_id": f"rows_{i}"}
        ))

    return docs


def build_llamaindex_engine(df: pd.DataFrame, ai_client=None):
    """
    Build a LlamaIndex VectorStoreIndex from the DataFrame.
    Uses HuggingFace sentence-transformers for embedding (free, local).
    Returns a query engine.
    """
    try:
        from llama_index.core import VectorStoreIndex, Settings
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.llms.openai_like import OpenAILike
    except ImportError as e:
        raise ImportError(
            f"Missing dependency: {e}\n"
            "Run: pip install llama-index llama-index-embeddings-huggingface "
            "llama-index-llms-openai-like sentence-transformers"
        )

    # Local sentence-transformer embedding (no API key needed)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.embed_model = embed_model
    Settings.chunk_size = 512

    # Wire LLM: use vLLM / Ollama OpenAI-compatible endpoint if available,
    # otherwise use a dummy LLM and return raw context for the AIClient to handle.
    vllm_host = os.getenv("VLLM_HOST", "")
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    try:
        if vllm_host:
            llm = OpenAILike(
                model=os.getenv("VLLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.2"),
                api_base=f"{vllm_host}/v1",
                api_key="EMPTY",
                is_chat_model=True,
            )
        else:
            llm = OpenAILike(
                model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
                api_base=f"{ollama_host}/v1",
                api_key="ollama",
                is_chat_model=True,
            )
        Settings.llm = llm
    except Exception:
        # Fall back to no LLM — return raw retrieved context
        from llama_index.core.llms import MockLLM
        Settings.llm = MockLLM()

    docs = _build_documents(df)
    index = VectorStoreIndex.from_documents(docs, show_progress=False)
    return index.as_query_engine(similarity_top_k=5)


def llamaindex_query(engine, question: str) -> tuple[str, list[str]]:
    """
    Query the LlamaIndex engine.
    Returns (answer_text, list_of_source_chunks).
    """
    response = engine.query(question)
    answer = str(response)
    sources = []
    if hasattr(response, "source_nodes"):
        for node in response.source_nodes:
            sources.append(node.node.text[:300])
    return answer, sources
