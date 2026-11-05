# =============================================================================
# agent.py  —  a small offline AI agent that manages your notes
#
# Big picture:
#   1. You type a question.
#   2. We send the whole conversation + a list of "tools" (Python functions)
#      to the local model running in Ollama.
#   3. The model either answers with text, OR asks us to run a tool.
#   4. If it asks for a tool, WE run the Python function and send the result
#      back to the model. Then we repeat from step 2.
#   5. When the model answers with text and no tool request, we're done.
#
# The model never runs code itself. It only *asks*. Your code decides.
# =============================================================================

import os                      # file paths, listing folders, deleting files
import json                    # reading/writing memory.json
from datetime import date      # to tell the model today's date
import ollama                  # talks to the Ollama server on localhost:11434
import rag                     # our own rag.py (we need rag.build_index after writes)
from rag import search_notes   # the RAG search tool, imported so the model can use it


# -----------------------------------------------------------------------------
# SETTINGS
# MODEL: which local model to use.
# THINK: True = the model reasons before answering (slower, usually smarter).
# WORKDIR: the folder you ran `python agent.py` from = the project folder.
# -----------------------------------------------------------------------------
MODEL = "qwen3.5:9b"
THINK = True
WORKDIR = os.path.abspath(".")
NOTES_DIR = os.path.join(WORKDIR, "notes")
MEMORY_FILE = os.path.join(WORKDIR, "memory.json")


# -----------------------------------------------------------------------------
# PATH SAFETY CHECK
# The model chooses file names, and a model can be tricked or simply wrong
# (e.g. "../../etc/passwd"). This turns the name into a full path and refuses
# anything that escapes the allowed folder. Returns the path, or None.
# -----------------------------------------------------------------------------
def _safe_path(base: str, relative: str):
    path = os.path.abspath(os.path.join(base, relative))
    # Checking base + "/" prevents "local-agent-evil" from passing as "local-agent".
    return path if path == base or path.startswith(base + os.sep) else None


# -----------------------------------------------------------------------------
# NOTE PATH HELPER  (the fix for the "notes/notes/tasks.md" bug)
# The model may write "tasks.md" OR "notes/tasks.md" for the same note.
# Instead of making the model remember a rule, we accept BOTH forms:
# strip a leading "notes/" and always resolve inside the notes folder.
# Lesson: make tools forgiving and consistent; don't make the model guess.
# -----------------------------------------------------------------------------
def _note_path(filename: str):
    name = filename.strip().removeprefix("./").removeprefix("notes/")
    return _safe_path(NOTES_DIR, name), name


# -- Lists existing notes; used inside error messages so the model knows
#    exactly which names are valid and can fix its mistake on the next step.
def _existing_notes() -> str:
    if not os.path.isdir(NOTES_DIR):
        return "(no notes folder)"
    return ", ".join(sorted(f for f in os.listdir(NOTES_DIR) if not f.startswith("."))) or "(empty)"


# =============================================================================
# READ TOOLS  —  safe, they only look at things, so they run automatically.
#
# IMPORTANT: the docstring ("""...""") of every tool is SENT TO THE MODEL.
# Ollama reads the function name, the parameters and the docstring and turns
# them into a description the model can understand. A clear docstring = the
# model uses the tool correctly. A vague one = it guesses.
# =============================================================================

# -- list_files: shows what's in a folder (hidden files like .venv are skipped)
def list_files(folder: str = ".") -> str:
    """List the files in a folder inside the project directory.

    Args:
        folder: Relative path of the folder to list. Use "." for the project root, "notes" for notes.
    """
    path = _safe_path(WORKDIR, folder)
    if not path or not os.path.isdir(path):
        return f"Error: '{folder}' is not a folder inside the project. Try '.' or 'notes'."
    return "\n".join(f for f in os.listdir(path) if not f.startswith(".") and f != "__pycache__")


# -- read_file: returns a file's text (cut at 4000 characters so one big file
#    can't fill the context window). If the file isn't found in the project
#    root, it also tries the notes folder, so "tasks.md" and "notes/tasks.md"
#    both work. On failure, the error says which notes exist.
def read_file(filename: str) -> str:
    """Read a text file. Works for notes ("tasks.md" or "notes/tasks.md") and project files ("agent.py").

    Args:
        filename: Name or relative path of the file to read.
    """
    path = _safe_path(WORKDIR, filename)
    if path and not os.path.isfile(path):       # not in project root → try notes/
        path, _ = _note_path(filename)
    if not path:
        return "Error: access outside the project folder is not allowed."
    if not os.path.isfile(path):
        return f"Error: '{filename}' not found. Existing notes: {_existing_notes()}"
    with open(path, encoding="utf-8") as f:
        text = f.read()
    return text[:4000] if text.strip() else f"'{filename}' exists but is empty."


# -- _load_facts: helper (not a tool, the model can't see it because it's not
#    in TOOLS). Reads memory.json and returns a Python list of facts.
def _load_facts() -> list:
    if not os.path.exists(MEMORY_FILE):
        return []
    with open(MEMORY_FILE, encoding="utf-8") as f:
        return json.load(f)


# -- remember: LONG-TERM MEMORY (write side). Adds one fact to memory.json.
#    Saved on disk, so it survives when you close the program.
def remember(fact: str) -> str:
    """Save an important fact about the user for future conversations.

    Args:
        fact: A short fact to remember, e.g. "User's name is Mucahit".
    """
    facts = _load_facts()
    facts.append(fact)
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(facts, f, ensure_ascii=False, indent=2)  # keeps Turkish letters readable
    return f"Saved. I now remember {len(facts)} facts."


# -- recall: LONG-TERM MEMORY (read side). Returns every saved fact.
#    Fine for a few facts; for thousands you'd use RAG instead (see rag.py).
def recall() -> str:
    """Return all facts saved about the user from previous conversations."""
    facts = _load_facts()
    return "\n".join(facts) if facts else "No saved facts yet."


# =============================================================================
# WRITE TOOLS  —  these CHANGE things on disk, so they need your approval.
# They only touch .md / .txt files inside notes/ (via _note_path).
# After each change we call rag.build_index() so search_notes immediately
# knows about the new/edited/deleted note (fixes the "stale index" problem).
# =============================================================================

# -- write_note: creates a note, or REPLACES an existing one completely.
def write_note(filename: str, content: str) -> str:
    """Create a new note or overwrite an existing note in the notes folder.

    Args:
        filename: Note name, e.g. "tasks.md" (a leading "notes/" is also fine). Must end with .md or .txt.
        content: The full text to write into the note.
    """
    path, name = _note_path(filename)
    if not path or not name.endswith((".md", ".txt")) or "/" in name:
        return "Error: use a simple name ending in .md or .txt, like 'tasks.md'."
    with open(path, "w", encoding="utf-8") as f:   # "w" = write (overwrite)
        f.write(content)
    rag.build_index()
    return f"Wrote {len(content)} characters to notes/{name}. You can read it back with read_file('{name}')."


# -- append_to_note: adds text to the END of a note, keeps the old content.
def append_to_note(filename: str, content: str) -> str:
    """Add text to the end of an existing note without deleting what is already there.

    Args:
        filename: Existing note name, e.g. "tasks.md".
        content: The text to add at the end.
    """
    path, name = _note_path(filename)
    if not path or not os.path.isfile(path):
        return f"Error: note '{name}' does not exist. Existing notes: {_existing_notes()}. Use write_note to create a new one."
    with open(path, "a", encoding="utf-8") as f:   # "a" = append
        f.write("\n\n" + content)                  # blank line = new chunk for RAG
    rag.build_index()
    return f"Appended {len(content)} characters to notes/{name}."


# -- delete_note: removes a note permanently (no trash bin!).
def delete_note(filename: str) -> str:
    """Permanently delete a note from the notes folder.

    Args:
        filename: Note name to delete, e.g. "old.md".
    """
    path, name = _note_path(filename)
    if not path or not os.path.isfile(path):
        return f"Error: note '{name}' does not exist. Existing notes: {_existing_notes()}"
    os.remove(path)
    rag.build_index()
    return f"Deleted notes/{name}."


# -----------------------------------------------------------------------------
# TOOL REGISTRY
# TOOLS maps a name ("read_file") to the real function (read_file).
#   - We send TOOLS.values() to the model so it knows what it can call.
#   - When the model says "call read_file", we look it up here by name.
# NEEDS_APPROVAL lists the tools that must ask you before running.
# To add a new tool: write the function, then add it to this list.
# -----------------------------------------------------------------------------
TOOLS = {f.__name__: f for f in [
    list_files, read_file, remember, recall, search_notes,
    write_note, append_to_note, delete_note,
]}
NEEDS_APPROVAL = {"write_note", "append_to_note", "delete_note"}


# -----------------------------------------------------------------------------
# SYSTEM PROMPT
# The first message in every conversation. The user never sees it, but it
# shapes the model's behaviour. It starts with today's date because the model
# cannot know the date by itself (that's what broke the first weekly plan).
# -----------------------------------------------------------------------------
SYSTEM = {"role": "system", "content": (
    f"Today is {date.today().strftime('%A, %d %B %Y')}. "
    "You are a helpful personal assistant that manages the user's notes.\n"
    "- Use search_notes or read_file to look at notes before answering questions about them.\n"
    "- Use remember / recall for facts about the user.\n"
    "- You can create and edit notes with write_note and append_to_note. "
    "The user must approve every change; if they deny it, do not retry the same action, "
    "ask what they would like instead.\n"
    "- For tasks with several steps, first write a short numbered plan, then carry it out step by step.\n"
    "- Respect deadlines: never schedule anything after a deadline."
)}


# -----------------------------------------------------------------------------
# HUMAN-IN-THE-LOOP
# Shows you exactly what the agent wants to do (long text is shortened)
# and waits for you to type y or n. Returns True only for "y".
# -----------------------------------------------------------------------------
def ask_approval(name: str, args: dict) -> bool:
    print(f"\n⚠️  The agent wants to run: {name}")
    for key, value in args.items():
        text = str(value)
        print(f"   {key}: {text if len(text) < 300 else text[:300] + '...'}")
    return input("   Allow? (y/n): ").strip().lower() == "y"


# -----------------------------------------------------------------------------
# RUN ONE TOOL SAFELY
# Every tool call goes through here. Rule: ALWAYS return a string, NEVER crash.
#   1. Unknown tool name (model made it up)      → error message
#   2. Risky tool and you said "n"               → "User denied" message
#   3. The tool itself throws an exception       → error message
#   4. The tool returned None or a non-string    → converted to a string
# -----------------------------------------------------------------------------
def run_tool(name: str, args: dict) -> str:
    if name not in TOOLS:
        return f"Error: unknown tool '{name}'. Available tools: {', '.join(TOOLS)}"
    if name in NEEDS_APPROVAL and not ask_approval(name, args):
        return "User denied this action. Do not retry it; ask the user what they want instead."
    try:
        result = TOOLS[name](**args)   # **args turns {"filename": "a.md"} into filename="a.md"
    except Exception as e:
        return f"Error: {e}"
    return str(result) if result is not None else "Error: tool returned nothing."


# =============================================================================
# THE AGENT LOOP  —  the heart of the agent.
# One call to run_agent() = one answer to one user question, which may take
# several "steps" (model → tool → model → tool → ... → final answer).
# max_steps stops it if the model gets stuck calling tools forever.
# =============================================================================
def run_agent(messages: list, max_steps: int = 12):
    for step in range(1, max_steps + 1):
        print(f"\n--- step {step} ---")

        # -- Ask the model: full conversation + tool list + settings.
        #    num_ctx = context window size (how many tokens it can "see").
        response = ollama.chat(
            model=MODEL,
            messages=messages,
            tools=list(TOOLS.values()),
            think=THINK,
            options={"num_ctx": 16384},
        )
        msg = response.message
        messages.append(msg)   # the model's reply becomes part of the history

        # -- With THINK on, the model's private reasoning arrives separately
        #    in msg.thinking. We print a shortened version so you can watch it.
        if getattr(msg, "thinking", None):
            t = msg.thinking.strip().replace("\n", " ")
            print(f"🧠 {t[:400]}{'...' if len(t) > 400 else ''}")

        # -- Text written together with a tool request is usually its plan.
        if msg.content and msg.tool_calls:
            print("💭", msg.content)

        # -- No tool requested → this is the final answer. Done.
        if not msg.tool_calls:
            print("🤖 ANSWER:", msg.content)
            return

        # -- Run each requested tool and add its result to the conversation
        #    as a "tool" message, so the model can read it on the next step.
        for call in msg.tool_calls:
            name, args = call.function.name, call.function.arguments
            short_args = {k: (str(v)[:60] + '...' if len(str(v)) > 60 else v) for k, v in args.items()}
            print(f"🔧 model wants: {name}({short_args})")
            result = run_tool(name, args)
            print(f"📄 result: {result[:200]}")
            messages.append({"role": "tool", "content": result, "tool_name": name})

    # -- Only reached if all steps were used without a final answer.
    print("⚠️ Stopped: reached max steps.")


# =============================================================================
# MAIN PROGRAM  —  runs only when you do `python agent.py`.
# `messages` is created ONCE here, outside the loop, so it keeps growing
# with every question. That list IS the agent's short-term memory.
# =============================================================================
if __name__ == "__main__":
    messages = [SYSTEM]
    while True:
        q = input("\nYou: ")
        if q.lower() in ("exit", "quit"):
            break
        messages.append({"role": "user", "content": q})
        run_agent(messages)
        print(f"   (conversation size: {len(messages)} messages)")