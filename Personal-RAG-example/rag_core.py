"""
rag_core.py — Shared RAG logic: Chroma retrieval + LLM calls via Ollama's
OpenAI-compatible API. Imported by rag_cli.py, streamlit_app.py, and
chainlit_app.py so the retrieval/prompting logic lives in exactly one place.
"""

import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

# >>> Chroma DB settings — must match your ingestion script <<<
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # routes through SentenceTransformerEmbeddingFunction

# >>> Ollama connection settings (served via its OpenAI-compatible API) <<<
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "llama3.2:latest"
OLLAMA_TEMPERATURE = 0.0

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions using only the provided "
    "context. If the context does not contain the answer, say you don't know "
    "based on the given documents. Be concise and cite which part of the "
    "context you relied on when helpful."
)


def get_collection(db_path: str = CHROMA_DB_PATH, collection_name: str = COLLECTION_NAME,
                    embedding_model: str = EMBEDDING_MODEL):
    """Open a Chroma collection with a query-side embedder matching ingestion."""
    client = chromadb.PersistentClient(path=db_path)
    embed_fn = (
        embedding_functions.DefaultEmbeddingFunction()
        if embedding_model == "default"
        else embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model)
    )
    return client.get_collection(name=collection_name, embedding_function=embed_fn)


def get_client(base_url: str = OLLAMA_BASE_URL) -> OpenAI:
    """OpenAI-compatible client pointed at a local Ollama server."""
    # Ollama doesn't check the API key, but the OpenAI client requires a non-empty string.
    return OpenAI(base_url=base_url, api_key="ollama")


def retrieve(collection, question: str, k: int = 4):
    results = collection.query(query_texts=[question], n_results=k)
    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    return docs, metadatas, distances


def build_prompt(question: str, chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(
        f"[Chunk {i + 1}]\n{chunk}" for i, chunk in enumerate(chunks)
    )
    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer the question using only the context above."
    )


def ask_ollama(client: OpenAI, model: str, temperature: float, system_prompt: str, prompt: str) -> str:
    """Single-shot (non-streaming) completion."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    response = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
    return response.choices[0].message.content


def ask_ollama_stream(client: OpenAI, model: str, temperature: float, system_prompt: str, prompt: str):
    """Generator that yields text pieces as they stream in."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    response = client.chat.completions.create(model=model, messages=messages, temperature=temperature, stream=True)
    for chunk in response:
        piece = chunk.choices[0].delta.content or ""
        if piece:
            yield piece
