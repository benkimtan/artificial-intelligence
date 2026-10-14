"""
Document ingestion & semantic search using ChromaDB.

Flow:
  1. ingest_document() reads a .txt or .pdf file, chunks it, embeds each
     chunk, and stores it in a persistent Chroma collection.
  2. query() takes an incoming question, embeds it, and returns the top-k
     most semantically similar chunks + their similarity scores.

Key idea (vectorisation / embedding):
  An embedding model turns a piece of text into a list of numbers (a
  "vector"). Texts with similar MEANING get vectors that are close together,
  even if they share no keywords. A vector database (ChromaDB) stores these
  vectors and can quickly find the ones closest to a query vector.

Install:
    pip install chromadb 
    pip install sentence-transformers 
    pip install pypdf
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import chromadb
from chromadb.utils import embedding_functions

# =============================================================================
# CONFIGURATION - change these values to tune the pipeline
# =============================================================================

# The file to chunk and embed when you run this script. Supports .txt and .pdf.
# Example: INPUT_FILE = "./docs/employee_handbook.pdf"
# Leave as None to use the small built-in sample text instead.
INPUT_FILE = "./Chatbot_Usage_Policy.pdf"

# Questions to try against the ingested file when running this script.
# Replace these with questions that make sense for YOUR document.
SAMPLE_QUESTIONS = [
    "What am I allowed to ask the chatbot?",
    "I want to ask about the return policy?",
    "Can I get a refund for a clearance item?",
    "What's the weather like today?",  # clearly out-of-scope test
]

# Maximum size of one chunk, in CHARACTERS (roughly 4 characters = 1 token).
#   Smaller chunks -> more precise matches, but less context per chunk.
#   Larger chunks  -> more context, but matching gets "blurrier".
CHUNK_SIZE = 500

# Number of characters repeated from the end of one chunk at the start of the
# next. This stops an idea from being lost when it straddles a chunk boundary.
# A common rule of thumb is 10-20% of CHUNK_SIZE.
CHUNK_OVERLAP = 50

# Embedding model (any model name supported by the sentence-transformers lib).
# "all-MiniLM-L6-v2" is small, fast and runs locally on CPU.
# NOTE: use the SAME model for ingesting and querying, and re-ingest if you
# change it (vectors from different models cannot be compared).
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Name of the collection (think of it as a table) inside ChromaDB.
COLLECTION_NAME = "documents"

# Folder where ChromaDB saves its data so it survives restarts.
PERSIST_DIRECTORY = "./chroma_db"

# How many chunks to return from a search by default.
TOP_K = 3


@dataclass
class RetrievedChunk:
    text: str
    similarity: float  # 0.0-1.0, higher = more similar (cosine)
    chunk_id: str


def read_document(file_path: str) -> str:
    """Return the plain text of a .txt or .pdf file.

    PDFs are read with pypdf, which extracts the text layer of each page.
    Caveat: a *scanned* PDF (pictures of pages) has no text layer, so it
    would need OCR first.
    """
    if file_path.lower().endswith(".pdf"):
        # Imported here so .txt users don't need pypdf installed.
        from pypdf import PdfReader

        reader = PdfReader(file_path)
        # Extract each page's text and join the pages with a newline.
        # (extract_text() can return None for empty pages, hence "or ''".)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    # Anything else is treated as a plain text file.
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Simple sentence-aware chunker: splits on sentence boundaries, then
    groups sentences into ~chunk_size-character windows with overlap.

    A naive fixed-width character split can cut a sentence (and therefore
    an idea) in half. This keeps sentences whole.
    """
    # Split after '.', '!' or '?' when followed by whitespace.
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    current: list[str] = []  # sentences of the chunk currently being built
    current_len = 0

    for sentence in sentences:
        # If adding this sentence would exceed the size limit, close the chunk.
        if current_len + len(sentence) > chunk_size and current:
            chunks.append(" ".join(current))
            # keep the last sentence(s) for overlap, roughly by character budget
            overlap_sentences: list[str] = []
            overlap_len = 0
            for s in reversed(current):
                if overlap_len + len(s) > chunk_overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_len += len(s)
            # The next chunk starts with those overlapping sentences.
            current = overlap_sentences
            current_len = overlap_len

        current.append(sentence)
        current_len += len(sentence)

    # Don't forget the final, partially filled chunk.
    if current:
        chunks.append(" ".join(current))

    return chunks


class DocumentStore:
    def __init__(
        self,
        collection_name: str = COLLECTION_NAME,
        persist_directory: str = PERSIST_DIRECTORY,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
    ):
        # PersistentClient saves everything to disk in persist_directory.
        self.client = chromadb.PersistentClient(path=persist_directory)
        # The embedding function converts text -> vectors. Chroma calls it
        # automatically whenever we add documents or run a query.
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model_name
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
        )

    def ingest_document(
        self,
        file_path: str,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        reset: bool = False,
    ) -> int:
        """Read a .txt or .pdf file, chunk it, and add it to the collection.

        reset=True clears any existing chunks first — use this when
        re-ingesting an updated version of the document, otherwise you'll
        accumulate duplicate/stale chunks across runs.
        Returns the number of chunks ingested.
        """
        # 1. LOAD: file -> plain text
        text = read_document(file_path)

        # 2. CHUNK: plain text -> list of small pieces
        chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        if reset:
            # Chroma has no "clear collection" call; delete and recreate.
            self.client.delete_collection(self.collection.name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection.name,
                embedding_function=self.embedding_fn,
            )

        # 3. EMBED + STORE: every chunk needs a unique ID. Chroma embeds the
        # chunks for us inside add() and stores text + vector together.
        ids = [f"{file_path}::chunk_{i}" for i in range(len(chunks))]
        self.collection.add(documents=chunks, ids=ids)
        return len(chunks)

    def query(self, question: str, top_k: int = TOP_K) -> list[RetrievedChunk]:
        """Semantic search: return the top_k chunks most similar
        to the question, with similarity scores."""
        # Chroma embeds the question with the same model, then finds the
        # stored vectors closest to it.
        results = self.collection.query(query_texts=[question], n_results=top_k)

        retrieved: list[RetrievedChunk] = []
        # Results are lists-of-lists (one inner list per query text). We only
        # sent one question, so we take element [0].
        documents = results["documents"][0]
        distances = results["distances"][0]  # Chroma returns distance, not similarity
        ids = results["ids"][0]

        for doc, distance, chunk_id in zip(documents, distances, ids):
            # Chroma's default distance is squared L2 on normalized embeddings,
            # which relates to cosine similarity as: similarity = 1 - distance/2
            similarity = 1 - (distance / 2)
            retrieved.append(RetrievedChunk(text=doc, similarity=similarity, chunk_id=chunk_id))

        return retrieved

    def top_similarity(self, question: str, top_k: int = TOP_K) -> float:
        """Convenience method: just the best similarity score, handy as a
        cheap "is this question related to my documents at all?" check."""
        results = self.query(question, top_k=top_k)
        return max((r.similarity for r in results), default=0.0)


if __name__ == "__main__":
    import os
    import tempfile

    if INPUT_FILE:
        # Use the file named in the configuration section at the top.
        path = INPUT_FILE
        cleanup = False
    else:
        # No file configured: write a small sample text to a temp file.
        sample_text = (
            "Our return policy allows returns within 30 days of purchase with a valid receipt. "
            "Refunds are issued to the original payment method within 5-7 business days. "
            "Items must be unused and in original packaging. "
            "We do not accept returns on final-sale or clearance items. "
            "Shipping costs are non-refundable unless the item was defective."
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(sample_text)
            path = f.name
        cleanup = True

    try:
        store = DocumentStore()  # persists to PERSIST_DIRECTORY (see config)
        n_chunks = store.ingest_document(path, reset=True)  # uses CHUNK_SIZE / CHUNK_OVERLAP from config
        print(f"Ingested {n_chunks} chunks")

        for q in SAMPLE_QUESTIONS:
            results = store.query(q, top_k=3)
            print(f"\nQuestion: {q}")
            for r in results:
                print(f"  similarity={r.similarity:.3f} | {r.text[:200]}...")
    finally:
        if cleanup:
            os.unlink(path)