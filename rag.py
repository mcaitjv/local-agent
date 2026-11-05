"""Tiny RAG: embed note chunks, then find the ones closest in meaning to a query.

No vector database on purpose -- just lists of numbers and cosine similarity,
so you can see exactly what a vector DB does under the hood.
"""
import os
import math
import ollama

# ---------- SETTINGS ----------
# Which embedding model turns text into vectors, and where your notes live.
# _index is the in-memory "vector database": every paragraph of every note,
# stored together with its vector so it can be compared to a query later.
EMBED_MODEL = "nomic-embed-text"
NOTES_DIR = os.path.abspath("notes")

_index = []  # list of {"source": filename, "text": chunk, "vector": [...]}


# ---------- _chunks ----------
# Splits a note into paragraphs (separated by blank lines).
# Each paragraph becomes one searchable piece, so a search returns the
# relevant paragraph instead of the whole file.
def _chunks(text: str):
    """Split a note into paragraphs (one chunk per paragraph)."""
    return [p.strip() for p in text.split("\n\n") if p.strip()]


# ---------- _cosine ----------
# Measures how similar two vectors are, giving a score near 1 when two
# texts mean nearly the same thing and near 0 when they are unrelated.
# This is the comparison a vector database does for you.
def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


# ---------- build_index ----------
# Prepares the notes for searching:
#   1. read every .md / .txt file in notes/
#   2. split each file into paragraphs
#   3. send the paragraphs to the embedding model to get one vector each
#   4. store (file name, paragraph, vector) in _index
# Runs once; after that searches only need to embed the query.
def build_index():
    """Read every .md/.txt file in notes/ and embed each chunk."""
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
        # nomic-embed-text expects a task prefix
        vectors = ollama.embed(model=EMBED_MODEL, input=[f"search_document: {p}" for p in pieces]).embeddings
        for text, vec in zip(pieces, vectors):
            _index.append({"source": name, "text": text, "vector": vec})
    print(f"📚 indexed {len(_index)} chunks from {NOTES_DIR}")


# ---------- search_notes ----------
# Finds the paragraphs that best match a question:
#   1. build the index if it hasn't been built yet
#   2. turn the question into a vector
#   3. score every stored paragraph against it with _cosine and sort best-first
#   4. keep at most the top 3, and only those scoring 0.6 or higher
#   5. return them as text, each labeled with its file name
# agent.py gives this function to the model as a tool, so the docstring
# below is what tells the model when and how to use it.
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
    for s, c in ranked[:3]:
        print(f"   🔎 {s:.3f}  {c['source']}  {'✅' if s >= 0.6 else '❌ dropped'}")
    scored = [c for s, c in ranked[:3] if s >= 0.6]
    if not scored:
        return "No relevant notes found."
    return "\n\n".join(f"[{c['source']}] {c['text']}" for c in scored)



# ---------- MAIN ----------
# Only runs when you start this file directly (python rag.py), not when
# agent.py imports it. Lets you try the search on its own: type a query,
# see the scores and results; type exit or quit to stop.
if __name__ == "__main__":
    build_index()
    while True:
        q = input("\nSearch: ")
        if q.lower() in ("exit", "quit"):
            break
        print(search_notes(q))
