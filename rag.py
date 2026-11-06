# =============================================================================
# rag.py  —  RAG (Retrieval-Augmented Generation) in ~60 lines
#
# Problem: you can't paste ALL your notes into the model every time; it would
# overflow the context window. RAG finds only the RELEVANT pieces and gives
# just those to the model.
#
# How:
#   1. Split each note into chunks (here: one paragraph = one chunk).
#   2. An EMBEDDING MODEL turns each chunk into a vector (a list of ~768
#      numbers). Texts with similar MEANING get vectors pointing in similar
#      directions.
#   3. When a question comes in, turn the question into a vector too.
#   4. Compare the question vector with every chunk vector (cosine similarity)
#      and return the closest ones.
#
# No vector database on purpose. A vector DB (Chroma, Qdrant...) does exactly
# this, just faster and for millions of chunks.
# =============================================================================

import os
import math
import ollama


# -----------------------------------------------------------------------------
# SETTINGS
# EMBED_MODEL is a different, much smaller model than qwen. It doesn't chat;
# it only converts text into vectors.
# MIN_SCORE: chunks scoring below this are thrown away as "not relevant".
# -----------------------------------------------------------------------------
EMBED_MODEL = "nomic-embed-text"
# The notes folder next to THIS file (not "wherever the program was started"),
# so search works no matter which app or folder launches the server.
NOTES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "notes")
MIN_SCORE = 0.6
TOP_K = 3   # at most this many chunks are returned

# The index: lives in memory (RAM) while the program runs.
# Each item looks like: {"source": "project.md", "text": "...", "vector": [0.01, -0.3, ...]}
_index = []


# -----------------------------------------------------------------------------
# CHUNKING
# Splits a note on blank lines, so every paragraph becomes its own chunk.
# Smaller chunks = more precise matches, but less context in each one.
# Real systems tune this a lot (fixed sizes, overlap between chunks, etc.).
# -----------------------------------------------------------------------------
def _chunks(text: str):
    return [p.strip() for p in text.split("\n\n") if p.strip()]


# -----------------------------------------------------------------------------
# COSINE SIMILARITY
# Measures how much two vectors point in the same direction.
#   1.0 = same meaning,  ~0 = unrelated.
# Formula: (a · b) / (|a| * |b|)
# -----------------------------------------------------------------------------
def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


# -----------------------------------------------------------------------------
# BUILD THE INDEX
# Reads every .md/.txt file in notes/, chunks it, embeds every chunk, and
# stores the result in _index. Called at startup and after every write
# (from agent.py), so the search always matches what's on disk.
# -----------------------------------------------------------------------------
def build_index():
    _index.clear()
    if not os.path.isdir(NOTES_DIR):
        return
    for name in sorted(os.listdir(NOTES_DIR)):
        if not name.endswith((".md", ".txt")):
            continue
        with open(os.path.join(NOTES_DIR, name), encoding="utf-8") as f:
            pieces = _chunks(f.read())
        if not pieces:
            continue
        # nomic-embed-text was trained with prefixes: "search_document: "
        # for stored text and "search_query: " for questions. Using them
        # noticeably improves matching. (Other embedding models may not need them.)
        vectors = ollama.embed(model=EMBED_MODEL, input=[f"search_document: {p}" for p in pieces]).embeddings
        for text, vec in zip(pieces, vectors):
            _index.append({"source": name, "text": text, "vector": vec})
    print(f"📚 indexed {len(_index)} chunks from {NOTES_DIR}")


# -----------------------------------------------------------------------------
# SEARCH (this is the TOOL the agent uses)
# 1. Build the index if it's empty (first call).
# 2. Embed the question.
# 3. Score every chunk, sort from best to worst.
# 4. Print the top scores so you can watch retrieval happen.
# 5. Keep only chunks above MIN_SCORE and return them as one text block,
#    each tagged with its source file so the model can say where it came from.
# -----------------------------------------------------------------------------
def search_notes(query: str) -> str:
    """Search the user's personal notes by meaning and return the most relevant passages.

    Args:
        query: What to look for, e.g. "when is the project due".
    """
    if not _index:
        build_index()
    if not _index:
        return "No notes found."

    q_vec = ollama.embed(model=EMBED_MODEL, input=f"search_query: {query}").embeddings[0]

    ranked = sorted(((_cosine(q_vec, c["vector"]), c) for c in _index),
                    key=lambda x: x[0], reverse=True)

    for s, c in ranked[:TOP_K]:
        print(f"   🔎 {s:.3f}  {c['source']}  {'✅' if s >= MIN_SCORE else '❌ dropped'}")

    scored = [c for s, c in ranked[:TOP_K] if s >= MIN_SCORE]
    if not scored:
        return "No relevant notes found."
    return "\n\n".join(f"[{c['source']}] {c['text']}" for c in scored)


# =============================================================================
# STANDALONE TEST MODE  —  `python rag.py` lets you try searches directly,
# without the agent. Useful for tuning MIN_SCORE and testing queries.
# =============================================================================
if __name__ == "__main__":
    build_index()
    while True:
        q = input("\nSearch: ")
        if q.lower() in ("exit", "quit"):
            break
        print(search_notes(q))