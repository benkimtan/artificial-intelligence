"""
chainlit_app.py — Chainlit GUI for the RAG app.

Run with:
    chainlit run chainlit_app.py -w

`-w` enables auto-reload while you edit. Each question is answered independently — no conversation history is sent to
the LLM, even though Chainlit displays the running chat log on screen.

"""

import chainlit as cl

from rag_core import (
    CHROMA_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL,
    OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TEMPERATURE, DEFAULT_SYSTEM_PROMPT,
    get_collection, get_client, retrieve, build_prompt,
)

K = 4  # number of chunks to retrieve; edit here or wire up cl.ChatSettings if you want it configurable in the UI


@cl.on_chat_start
async def start():
    try:
        collection = get_collection(CHROMA_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL)
    except Exception as e:
        await cl.Message(
            content=f"⚠️ Could not open collection '{COLLECTION_NAME}' at '{CHROMA_DB_PATH}': {e}"
        ).send()
        return

    client = get_client(OLLAMA_BASE_URL)
    cl.user_session.set("collection", collection)
    cl.user_session.set("client", client)

    await cl.Message(
        content=f"Connected to collection **{COLLECTION_NAME}**. Ask me anything about your documents."
    ).send()


@cl.on_message
async def main(message: cl.Message):
    collection = cl.user_session.get("collection")
    client = cl.user_session.get("client")

    if collection is None or client is None:
        await cl.Message(content="Not connected to the vector DB — restart the chat.").send()
        return

    question = message.content
    docs, metadatas, distances = retrieve(collection, question, K)

    if not docs:
        await cl.Message(content="No relevant chunks found in the vector DB.").send()
        return

    elements = [
        cl.Text(
            name=f"Chunk {i + 1} — {(meta or {}).get('source', 'unknown')} (distance {dist:.4f})",
            content=doc,
            display="side",
        )
        for i, (doc, meta, dist) in enumerate(zip(docs, metadatas, distances))
    ]

    prompt = build_prompt(question, docs)
    msg = cl.Message(content="", elements=elements)
    await msg.send()

    try:
        stream = client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=OLLAMA_TEMPERATURE,
            stream=True,
        )
        for chunk in stream:
            piece = chunk.choices[0].delta.content or ""
            if piece:
                await msg.stream_token(piece)
    except Exception as e:
        msg.content = f"Error calling Ollama: {e}\n\nIs Ollama running? Try: `ollama serve`"

    await msg.update()
