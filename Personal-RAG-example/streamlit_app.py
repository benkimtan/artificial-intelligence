"""
streamlit_app.py — Streamlit GUI for the RAG app.

Run with:
    streamlit run streamlit_app.py

Each question is answered independently — no conversation history is sent to
the LLM, even though the chat log is displayed on screen for readability.

"""

import streamlit as st

from rag_core import (
    CHROMA_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL,
    OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TEMPERATURE, DEFAULT_SYSTEM_PROMPT,
    get_collection, get_client, retrieve, build_prompt, ask_ollama_stream,
)

st.set_page_config(page_title="Document Q&A", page_icon="📄")
st.title("📄 Document Q&A (RAG)")

with st.sidebar:
    st.header("Settings")
    db_path = st.text_input("Chroma DB path", value=CHROMA_DB_PATH)
    collection_name = st.text_input("Collection name", value=COLLECTION_NAME)
    embedding_model = st.text_input("Embedding model", value=EMBEDDING_MODEL,
                                     help="Must match what your ingestion script used.")
    model = st.text_input("Ollama model", value=OLLAMA_MODEL)
    base_url = st.text_input("Ollama base URL", value=OLLAMA_BASE_URL)
    temperature = st.slider("Temperature", 0.0, 1.0, OLLAMA_TEMPERATURE, 0.1)
    k = st.slider("Chunks to retrieve (k)", 1, 10, 4)
    show_chunks = st.checkbox("Show retrieved chunks", value=False)
    system_prompt = st.text_area("System prompt", value=DEFAULT_SYSTEM_PROMPT, height=120)

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()


@st.cache_resource(show_spinner="Connecting to Chroma...")
def load_collection(db_path: str, collection_name: str, embedding_model: str):
    return get_collection(db_path, collection_name, embedding_model)


@st.cache_resource
def load_client(base_url: str):
    return get_client(base_url)


try:
    collection = load_collection(db_path, collection_name, embedding_model)
except Exception as e:
    st.error(f"Could not open collection '{collection_name}' at '{db_path}': {e}")
    st.stop()

client = load_client(base_url)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a question about your documents...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        docs, metadatas, distances = retrieve(collection, question, k)

        if not docs:
            answer = "No relevant chunks found in the vector DB."
            st.markdown(answer)
        else:
            if show_chunks:
                with st.expander(f"Retrieved {len(docs)} chunks"):
                    for i, (doc, meta, dist) in enumerate(zip(docs, metadatas, distances)):
                        src = (meta or {}).get("source", "unknown")
                        st.markdown(f"**Chunk {i+1}** — source: `{src}`, distance: {dist:.4f}")
                        st.text(doc[:800])

            prompt = build_prompt(question, docs)
            try:
                answer = st.write_stream(
                    ask_ollama_stream(client, model, temperature, system_prompt, prompt)
                )
            except Exception as e:
                answer = f"Error calling Ollama: {e}\n\nIs Ollama running? Try: `ollama serve`"
                st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
