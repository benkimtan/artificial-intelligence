#!/usr/bin/env python3
"""
rag_cli.py — Query a Chroma vector DB and answer questions with a local Ollama LLM.

Stateless / single-shot: one question in, one answer out. No conversation memory.

Usage:
    python rag_cli.py "What is the termination clause?"
    python rag_cli.py --db-path ./chroma_db --collection my_docs --k 5 "your question"
    python rag_cli.py --model llama3.1 --show-chunks "your question"
    python rag_cli.py                     # no question arg -> interactive prompt

Requirements:
    pip install chromadb openai
    Ollama must be running locally and serving its OpenAI-compatible API
    (default: http://localhost:11434/v1) with the chosen model already pulled,
    e.g.:  ollama pull llama3.2
"""

import argparse
import sys
import textwrap

import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI


# >>> Set the path to your Chroma DB folder here <<<
CHROMA_DB_PATH = "./chroma_db"

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


def build_prompt(question: str, chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(
        f"[Chunk {i+1}]\n{chunk}" for i, chunk in enumerate(chunks)
    )
    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer the question using only the context above."
    )


def get_collection(db_path: str, collection_name: str, embedding_model: str):
    client = chromadb.PersistentClient(path=db_path)

    embed_fn = embedding_functions.DefaultEmbeddingFunction() if embedding_model == "default" \
        else embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model)

    try:
        collection = client.get_collection(name=collection_name, embedding_function=embed_fn)
    except Exception as e:
        available = [c.name for c in client.list_collections()]
        print(f"Error: could not open collection '{collection_name}': {e}", file=sys.stderr)
        if available:
            print(f"Available collections in '{db_path}': {available}", file=sys.stderr)
        sys.exit(1)

    return collection


def retrieve(collection, question: str, k: int):
    results = collection.query(query_texts=[question], n_results=k)
    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    return docs, metadatas, distances


def ask_ollama(client: OpenAI, model: str, temperature: float, system_prompt: str, prompt: str, stream: bool = True) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    if not stream:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content

    full_text = []
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=True,
    )
    for chunk in response:
        piece = chunk.choices[0].delta.content or ""
        if piece:
            print(piece, end="", flush=True)
            full_text.append(piece)
    print()  # trailing newline
    return "".join(full_text)


def main():
    parser = argparse.ArgumentParser(
        description="Query a Chroma vector DB and answer with a local Ollama LLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("question", nargs="?", help="Question to ask. If omitted, enters interactive mode.")
    parser.add_argument("--db-path", default=CHROMA_DB_PATH,
                         help=f"Path to the Chroma persistent DB (default: {CHROMA_DB_PATH})")
    parser.add_argument("--collection", default="documents", help="Chroma collection name (default: documents)")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2",
                         help="Embedding model for the QUERY side. Must match what was used during ingestion. "
                              "Use 'default' for Chroma's built-in ONNX all-MiniLM-L6-v2, or a sentence-transformers "
                              "model name to use SentenceTransformerEmbeddingFunction directly (default: all-MiniLM-L6-v2)")
    parser.add_argument("--model", default=OLLAMA_MODEL, help=f"Ollama model name (default: {OLLAMA_MODEL})")
    parser.add_argument("--base-url", default=OLLAMA_BASE_URL,
                         help=f"Ollama OpenAI-compatible base URL (default: {OLLAMA_BASE_URL})")
    parser.add_argument("--temperature", type=float, default=OLLAMA_TEMPERATURE,
                         help=f"Sampling temperature (default: {OLLAMA_TEMPERATURE})")
    parser.add_argument("--k", type=int, default=4, help="Number of chunks to retrieve (default: 4)")
    parser.add_argument("--show-chunks", action="store_true", help="Print retrieved chunks before the answer")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT, help="Override the system prompt")

    # parse_known_args (not parse_args) so this also works inside Jupyter,
    # which injects its own hidden argv (e.g. `-f kernel-xxxx.json`) that
    # would otherwise cause an "unrecognized arguments" SystemExit.
    args, _unknown = parser.parse_known_args()

    collection = get_collection(args.db_path, args.collection, args.embedding_model)

    # Ollama's OpenAI-compatible endpoint doesn't validate the API key, but
    # the OpenAI client requires a non-empty string.
    client = OpenAI(base_url=args.base_url, api_key="ollama")

    if args.question:
        # Single-shot mode: one question, one answer, exit.
        answer_question(collection, client, args, args.question)
        return

    # Interactive mode: keep prompting for questions until the user exits.
    # Each question is still answered independently — no memory/history is
    # carried between turns.
    print(f"Interactive RAG session (collection: '{args.collection}', model: '{args.model}').")
    print("Type your question and press Enter. Type 'exit' or 'quit' (or Ctrl+D) to leave.\n")

    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            break

        answer_question(collection, client, args, question)
        print()  # spacer between turns


def answer_question(collection, client: OpenAI, args, question: str):
    docs, metadatas, distances = retrieve(collection, question, args.k)

    if not docs:
        print("No relevant chunks found in the vector DB.")
        return

    if args.show_chunks:
        print("\n=== Retrieved chunks ===")
        for i, (doc, meta, dist) in enumerate(zip(docs, metadatas, distances)):
            src = (meta or {}).get("source", "unknown")
            print(f"\n--- Chunk {i+1} (source: {src}, distance: {dist:.4f}) ---")
            print(textwrap.shorten(doc, width=400, placeholder=" [...]"))
        print("\n=== Answer ===\n")

    prompt = build_prompt(question, docs)

    try:
        ask_ollama(client, args.model, args.temperature, args.system_prompt, prompt, stream=not args.no_stream)
    except Exception as e:
        print(f"\nError calling Ollama: {e}", file=sys.stderr)
        print("Is Ollama running? Try: ollama serve  (and ollama pull <model>)", file=sys.stderr)


if __name__ == "__main__":
    main()