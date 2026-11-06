# =============================================================================
# notes_tools.py  —  ALL the tool code, in one place
#
# This is the "real" code behind your MCP server: plain Python functions that
# read notes, search them, remember facts, calculate weeks, and write files.
#
# This file knows NOTHING about:
#   - models (no ollama.chat here)
#   - agent loops
#   - MCP
# That's on purpose. Pure functions are easy to test, and the same file can be
# used in different ways:
#   notes_server.py  → publishes these tools over MCP (the real-world way)
#   agent.py         → imports them directly (your original "version 1")
#   eval.py          → tests them through agent.py
# One source of truth: fix a bug here and every user of the tools gets the fix.
# =============================================================================

import os                      # file paths, listing folders, deleting files
import json                    # reading/writing memory.json
from datetime import date, datetime, timedelta   # calendar math for week_ranges
import rag                     # our own rag.py (search + re-indexing after writes)
from rag import search_notes   # the RAG search tool (lives in rag.py, re-exported here)


# -----------------------------------------------------------------------------
# WHERE THE DATA LIVES
# BASE_DIR = the folder THIS FILE is in, not the folder you started from.
# Before, we used os.path.abspath("."), which means "wherever the program was
# started". That broke as soon as another app (Claude Code, Claude Desktop)
# started the server from a different folder: it would look for notes/ in the
# wrong place. __file__ always points to this file, so it works from anywhere.
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKDIR = BASE_DIR
NOTES_DIR = os.path.join(BASE_DIR, "notes")
MEMORY_FILE = os.path.join(BASE_DIR, "memory.json")


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
# Both agent.py (via Ollama) and notes_server.py (via the MCP SDK) read the
# function name, the type hints and the docstring and turn them into a tool
# description + JSON schema. A clear docstring = the model uses the tool
# correctly. A vague one = it guesses.
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
    """Read a text file. Works for notes ("tasks.md" or "notes/tasks.md") and project files.

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


# -- _parse_date: helper. Models write dates in many forms ("2026-12-15",
#    "December 15", "Dec 15 2026"...). We try each format; if there's no year,
#    we use this year (or next year if that date has already passed).
def _parse_date(text: str):
    text = text.strip().replace(",", "")
    for fmt in ("%Y-%m-%d", "%B %d %Y", "%b %d %Y", "%d %B %Y", "%d %b %Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            pass
    for fmt in ("%B %d", "%b %d", "%d %B", "%d %b"):
        try:
            d = datetime.strptime(text, fmt).date().replace(year=date.today().year)
            return d if d >= date.today() else d.replace(year=d.year + 1)
        except ValueError:
            pass
    return None


# -- week_ranges: CALENDAR MATH DONE BY CODE, NOT BY THE MODEL.
#    Added after eval showed the model skipping Nov 30 – Dec 5 again and again.
#    Models are bad at date arithmetic; Python never makes this mistake.
#    The model calls this tool, gets exact weeks back, and only has to fill in
#    the tasks. The last week is cut off at the end date (no week past a deadline).
def week_ranges(end_date: str, start_date: str = "") -> str:
    """Return the exact list of weeks between two dates. ALWAYS use this for weekly plans or schedules instead of calculating dates yourself.

    Args:
        end_date: Last day, e.g. "2026-12-15" or "December 15".
        start_date: First day. Leave empty to start today.
    """
    start = _parse_date(start_date) if start_date else date.today()
    end = _parse_date(end_date)
    if not start or not end:
        return "Error: could not read the dates. Use the format YYYY-MM-DD, e.g. 2026-12-15."
    if end < start:
        return f"Error: end date {end} is before start date {start}."
    lines, week, cur = [], 1, start
    while cur <= end:
        last = min(cur + timedelta(days=6), end)
        lines.append(f"Week {week}: {cur:%b} {cur.day} - {last:%b} {last.day}, {last.year}")
        cur, week = last + timedelta(days=1), week + 1
    return f"{len(lines)} weeks from {start} to {end}:\n" + "\n".join(lines)


# =============================================================================
# WRITE TOOLS  —  these CHANGE things on disk, so they need your approval.
# They only touch .md / .txt files inside notes/ (via _note_path).
# After each change we call rag.build_index() so search_notes immediately
# knows about the new/edited/deleted note (fixes the "stale index" problem).
# =============================================================================

# -- _reindex: rebuild the search index WITHOUT letting a failure hide the
#    fact that the file change already happened. (Found in the MCP test: the
#    note was written, then re-indexing failed, and the tool reported "Error",
#    so the model believed the write had failed. That's a lie by the tool.)
#    Now the change is reported as done, plus a warning if the index failed.
def _reindex() -> str:
    try:
        rag.build_index()
        return ""
    except Exception as e:
        return f" (Warning: the file change succeeded, but the search index could not be updated: {e})"


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
    return f"Wrote {len(content)} characters to notes/{name}. You can read it back with read_file('{name}')." + _reindex()


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
    return f"Appended {len(content)} characters to notes/{name}." + _reindex()


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
    return f"Deleted notes/{name}." + _reindex()


# =============================================================================
# TOOL CATALOG  —  the list of tools + how risky each one is.
# This is the single place that decides which functions are tools and which
# need approval. Both clients read it:
#   notes_server.py turns it into MCP annotations (read_only / destructive)
#   agent.py turns it into its NEEDS_APPROVAL set
#   "read"        → only looks at things; runs automatically
#   "write"       → adds data, never removes any
#   "destructive" → can overwrite or delete data
# To add a tool: write the function above, then add one line here.
# =============================================================================
TOOLS = {
    list_files:     "read",
    read_file:      "read",
    recall:         "read",
    search_notes:   "read",
    week_ranges:    "read",
    remember:       "write",
    append_to_note: "write",
    write_note:     "destructive",
    delete_note:    "destructive",
}


# -----------------------------------------------------------------------------
# HOW TO USE THESE TOOLS  —  instructions that travel WITH the tools.
# Rules about specific tools belong next to the tools, not in each agent.
# The MCP server sends this text to every client on connect; agent.py adds
# it to its system prompt. Rules about general behavior (assume vs ask, how
# to plan) stay in the agents, because they're about the agent, not the tools.
# -----------------------------------------------------------------------------
TOOL_INSTRUCTIONS = (
    "Tools for managing the user's personal notes and memory.\n"
    "- Use search_notes or read_file to look at notes before answering questions about them.\n"
    "- Use remember / recall for facts about the user.\n"
    "- For any weekly plan or schedule, call week_ranges to get the exact weeks and use "
    "those dates exactly. Never calculate dates yourself. Never schedule anything after a deadline.\n"
    "- write_note, append_to_note and delete_note change files and may need the user's approval. "
    "If an action is denied, do not retry it; ask the user what they want instead.\n"
    "- Be honest about results: if a tool failed or was denied, never say the action happened."
)