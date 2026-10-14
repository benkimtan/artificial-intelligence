#!/usr/bin/env python3
"""
rag_cli.py — Terminal RAG app: query a Chroma vector DB, answer with a local
Ollama LLM (via its OpenAI-compatible API). Single-shot per question — no
conversation memory is sent to the model.

Usage:
    python rag_cli.py "What is the termination clause?"
    python rag_cli.py --collection policy --db-path ./chroma_policy_db "your question"
    python rag_cli.py --show-chunks "your question"
    python rag_cli.py                     # no question arg -> interactive mode

"""

import argparse
import sys
import textwrap

from rag_core import (
    CHROMA_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL,
    OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TEMPERATURE, DEFAULT_SYSTEM_PROMPT,
    get_collection, get_client, retrieve, build_prompt, ask_ollama_stream, ask_ollama,
)


def answer_question(collection, client, args, question: str):
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
        if args.no_stream:
            print(ask_ollama(client, args.model, args.temperature, args.system_prompt, prompt))
        else:
            for piece in ask_ollama_stream(client, args.model, args.temperature, args.system_prompt, prompt):
                print(piece, end="", flush=True)
            print()
    except Exception as e:
        print(f"\nError calling Ollama: {e}", file=sys.stderr)
        print("Is Ollama running? Try: ollama serve  (and ollama pull <model>)", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Query a Chroma vector DB and answer with a local Ollama LLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("question", nargs="?", help="Question to ask. If omitted, enters interactive mode.")
    parser.add_argument("--db-path", default=CHROMA_DB_PATH, help=f"Path to the Chroma DB (default: {CHROMA_DB_PATH})")
    parser.add_argument("--collection", default=COLLECTION_NAME, help=f"Chroma collection name (default: {COLLECTION_NAME})")
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL,
                         help=f"Query-side embedding model, must match ingestion (default: {EMBEDDING_MODEL})")
    parser.add_argument("--model", default=OLLAMA_MODEL, help=f"Ollama model name (default: {OLLAMA_MODEL})")
    parser.add_argument("--base-url", default=OLLAMA_BASE_URL, help=f"Ollama OpenAI-compatible base URL (default: {OLLAMA_BASE_URL})")
    parser.add_argument("--temperature", type=float, default=OLLAMA_TEMPERATURE, help=f"Sampling temperature (default: {OLLAMA_TEMPERATURE})")
    parser.add_argument("--k", type=int, default=4, help="Number of chunks to retrieve (default: 4)")
    parser.add_argument("--show-chunks", action="store_true", help="Print retrieved chunks before the answer")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT, help="Override the system prompt")

    # parse_known_args so this also works inside Jupyter, which injects its
    # own hidden argv (e.g. `-f kernel-xxxx.json`).
    args, _unknown = parser.parse_known_args()

    try:
        collection = get_collection(args.db_path, args.collection, args.embedding_model)
    except Exception as e:
        print(f"Error: could not open collection '{args.collection}': {e}", file=sys.stderr)
        sys.exit(1)

    client = get_client(args.base_url)

    if args.question:
        answer_question(collection, client, args, args.question)
        return

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
        print()


if __name__ == "__main__":
    main()
