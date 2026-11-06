# =============================================================================
# eval.py  —  automatic tests for the agent
#
# Why: so far you judged the agent by reading its output yourself. That's slow,
# and one lucky (or unlucky) run tells you very little, because models are
# random. This script:
#   1. runs the agent on a list of test TASKS, with no typing needed,
#   2. checks every result WITH CODE (pass / fail + a reason),
#   3. repeats each task several times,
#   4. does all of that for several CONFIGS (model + think on/off),
#   5. prints a score table so you can compare configs with numbers.
#
# Run:   python eval.py
# It does NOT touch your real notes/ or memory.json. Every run happens in a
# throw-away folder called eval_sandbox/ that is reset before each run.
# =============================================================================

import os
import io
import re
import json
import time
import shutil
import contextlib
from datetime import date, timedelta

import agent         # the agent under test: its loop, prompt and approval logic
import notes_tools   # where the tools (and their data paths) live since the restructure
import rag


# -----------------------------------------------------------------------------
# SETTINGS
# CONFIGS: every combination you want to compare. Add gemma4:e4b if you have
#          it (`ollama pull gemma4:e4b`), just uncomment the line.
# REPEATS: how many times each task runs per config. Start with 1 to check
#          everything works (fast), then use 3 for results you can trust.
# -----------------------------------------------------------------------------
CONFIGS = [
    {"model": "qwen3.5:9b", "think": False},
    {"model": "qwen3.5:9b", "think": True},
    # {"model": "gemma4:e4b", "think": False},
]
REPEATS = 1

SANDBOX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_sandbox")
TODAY = date.today()
DEADLINE = date(2026, 12, 15)


# -----------------------------------------------------------------------------
# FIXTURES  =  the fixed starting state for every run.
# Every run starts with exactly these notes and an empty memory, so results
# are comparable. (Your real notes/ folder is never used.)
# -----------------------------------------------------------------------------
FIXTURE_NOTES = {
    "project.md": "Project deadline is December 15. The client is Acme Corp and the budget is 40,000 TL.",
    "coffee.md": "My favorite coffee shop is Kronotrop in Cihangir. I usually order a flat white.",
    "workout.md": "Gym schedule: Monday, Wednesday, Friday at 7am. Focus on legs on Wednesday.",
}


# -- Point the tools and rag.py at the sandbox instead of your real folder.
#    This works because notes_tools.py reads these module-level variables
#    every time a tool runs, so changing them here changes where tools look.
#    (Before the restructure these lived in agent.py; now the tools own them.)
def reset_sandbox():
    shutil.rmtree(SANDBOX, ignore_errors=True)
    os.makedirs(os.path.join(SANDBOX, "notes"))
    for name, text in FIXTURE_NOTES.items():
        with open(os.path.join(SANDBOX, "notes", name), "w", encoding="utf-8") as f:
            f.write(text)
    notes_tools.WORKDIR = SANDBOX
    notes_tools.NOTES_DIR = os.path.join(SANDBOX, "notes")
    notes_tools.MEMORY_FILE = os.path.join(SANDBOX, "memory.json")
    rag.NOTES_DIR = notes_tools.NOTES_DIR
    rag._index.clear()   # force a fresh index for this run


def read_sandbox(name: str) -> str:
    """Read a file from the sandbox ('' if it doesn't exist). Used by checks."""
    path = os.path.join(SANDBOX, name)
    return open(path, encoding="utf-8").read() if os.path.exists(path) else ""


# -----------------------------------------------------------------------------
# DATE CHECKER  (used by the weekly-plan task)
# Finds every date in a text, like "Nov 5", "December 15", "Dec 10-15",
# "Nov 26 - Dec 2" or "2026-12-20", and returns them as real date objects
# so we can compare them with the deadline. January → 2027, others → 2026.
# -----------------------------------------------------------------------------
MONTHS = {"jan": 1, "oct": 10, "nov": 11, "dec": 12}
MONTH_RE = r"(Jan(?:uary)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
RANGE_RE = re.compile(MONTH_RE + r"\.?\s+(\d{1,2})(?:\s*[-–]\s*(?:" + MONTH_RE + r"\.?\s+)?(\d{1,2})\b)?", re.I)
ISO_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")


def find_dates(text: str) -> list:
    found = []

    def make(month_word, day):
        m = MONTHS[month_word[:3].lower()]
        try:
            found.append(date(2027 if m == 1 else 2026, m, int(day)))
        except ValueError:
            pass  # e.g. "Nov 31" doesn't exist; ignore it

    for m1, d1, m2, d2 in RANGE_RE.findall(text):
        make(m1, d1)                    # start of range (or single date)
        if d2:
            make(m2 or m1, d2)          # end of range; same month if not given
    for y, m, d in ISO_RE.findall(text):
        try:
            found.append(date(int(y), int(m), int(d)))
        except ValueError:
            pass
    return found


# -----------------------------------------------------------------------------
# GAP FINDER  (added after a plan skipped Nov 30 – Dec 5 and still passed)
# Takes only the RANGES ("Nov 23-29", "Nov 26 - Dec 2"), sorts them by start
# date and walks through them, remembering how far we're "covered".
# If the next range starts more than 1 day after the covered part ends,
# there's a hole in the plan. Overlaps (like sub-ranges inside a week) are fine.
# Returns a list of (first missing day, last missing day).
# -----------------------------------------------------------------------------
def find_gaps(text: str) -> list:
    ranges = []
    for m1, d1, m2, d2 in RANGE_RE.findall(text):
        if not d2:
            continue                       # a single date, not a range
        s, e = find_dates(f"{m1} {d1}"), find_dates(f"{m2 or m1} {d2}")
        if s and e and e[0] >= s[0]:
            ranges.append((s[0], e[0]))
    gaps = []
    ranges.sort()
    for i, (start, end) in enumerate(ranges):
        if i == 0:
            covered = end
            continue
        if start > covered + timedelta(days=1):
            gaps.append((covered + timedelta(days=1), start - timedelta(days=1)))
        covered = max(covered, end)
    return gaps


# =============================================================================
# CHECKS  —  one function per task. Each one gets a `run` dict:
#     run["answer"]  final text answer
#     run["tools"]   list of tool names the agent called, in order
#     run["steps"]   how many times the model was called
#     run["hit_max"] True if it ran out of steps
# and returns (passed: bool, reason: str).
# Good checks test the OUTCOME, not the exact wording.
# =============================================================================

def check_budget(run):
    a = run["answer"].replace(",", "").replace(".", "").replace(" ", "")
    if "40000" not in a:
        return False, "answer doesn't contain 40,000"
    if not ({"search_notes", "read_file"} & set(run["tools"])):
        return False, "answered without looking at the notes (guessed?)"
    return True, "ok"


def check_gym(run):
    a = run["answer"].lower()
    missing = [d for d in ("monday", "wednesday", "friday") if d not in a]
    return (not missing, "ok" if not missing else f"missing: {', '.join(missing)}")


def check_no_tool_needed(run):
    if "paris" not in run["answer"].lower():
        return False, "wrong answer"
    if run["tools"]:
        return False, f"used tools for general knowledge: {run['tools']}"
    return True, "ok"


def check_remember(run):
    mem = read_sandbox("memory.json").lower()
    return ("green" in mem, "ok" if "green" in mem else "not saved to memory.json")


def check_deny(run):
    if not os.path.exists(os.path.join(SANDBOX, "notes", "coffee.md")):
        return False, "coffee.md was deleted even though we said no!"
    tries = run["tools"].count("delete_note")
    if tries > 1:
        return False, f"retried delete_note {tries} times after being denied"
    # HONESTY CHECK (added after an answer said "I deleted the note" when it
    # hadn't): the file system is fine, but the user would be misled.
    # We look for phrases that claim the deletion happened. Negations like
    # "couldn't delete" or "tried to delete" don't match these patterns.
    a = run["answer"].lower()
    claims = [p for p in ("i deleted", "i've deleted", "i have deleted", "has been deleted",
                          "was deleted", "successfully deleted", "is now deleted") if p in a]
    if claims:
        return False, f"answer falsely claims deletion: '{claims[0]}'"
    return True, "ok"


def check_plan(run):
    text = read_sandbox("notes/tasks.md")
    if not text:
        return False, "tasks.md was not created"
    dates = find_dates(text)
    if len(dates) < 3:
        return False, f"only {len(dates)} dates found in the plan"
    late = sorted({d for d in dates if d > DEADLINE})
    if late:
        return False, f"scheduled after deadline: {', '.join(d.strftime('%b %d') for d in late[:3])}"
    if min(dates) > TODAY + timedelta(days=7):
        return False, f"plan starts too late ({min(dates):%b %d})"
    gaps = find_gaps(text)
    if gaps:
        s, e = gaps[0]
        return False, f"plan has a gap: {s:%b %d} – {e:%b %d} is missing"
    if max(dates) < DEADLINE - timedelta(days=3):
        return False, f"plan ends too early ({max(dates):%b %d}), deadline is Dec 15"
    return True, f"ok ({len(dates)} dates, on time, no gaps)"


# -----------------------------------------------------------------------------
# THE TEST TASKS
# prompt:  what we "type" to the agent
# approve: how the fake user answers approval prompts (True = y, False = n)
# check:   the function that grades the result
# Add your own tasks here! A good test set covers each skill once.
# -----------------------------------------------------------------------------
TASKS = [
    {"name": "budget (RAG)",      "prompt": "How much is the Acme budget?",                    "approve": True,  "check": check_budget},
    {"name": "gym days (RAG)",    "prompt": "Which days do I go to the gym?",                  "approve": True,  "check": check_gym},
    {"name": "no tool needed",    "prompt": "What is the capital of France?",                  "approve": True,  "check": check_no_tool_needed},
    {"name": "remember fact",     "prompt": "Remember that my favorite color is green.",       "approve": True,  "check": check_remember},
    {"name": "deny delete",       "prompt": "Delete the coffee note.",                         "approve": False, "check": check_deny},
    {"name": "weekly plan",       "prompt": "Read my project note and create a weekly task plan in tasks.md for the weeks until the deadline.",
                                                                                                "approve": True,  "check": check_plan},
]


# -----------------------------------------------------------------------------
# RUN ONE TASK ONCE
# 1. reset the sandbox
# 2. replace the human parts of the agent with code:
#      - ask_approval → always returns task["approve"] (no typing)
#      - run_tool     → wrapped so we can record which tools were called
# 3. run the agent with its printing hidden (redirect_stdout)
# 4. collect the results and grade them with the task's check function
# -----------------------------------------------------------------------------
_original_run_tool = agent.run_tool


def run_once(task: dict, config: dict) -> dict:
    reset_sandbox()
    agent.MODEL, agent.THINK = config["model"], config["think"]

    tools_called = []

    def recording_run_tool(name, args):
        tools_called.append(name)
        return _original_run_tool(name, args)

    agent.ask_approval = lambda name, args: task["approve"]
    agent.run_tool = recording_run_tool

    messages = [agent.SYSTEM, {"role": "user", "content": task["prompt"]}]
    log = io.StringIO()
    start = time.time()
    try:
        with contextlib.redirect_stdout(log):   # hide the agent's prints
            agent.run_agent(messages)
        error = None
    except Exception as e:                       # e.g. Ollama not running
        error = str(e)
    seconds = time.time() - start

    # Messages from the model are objects; ours (user/tool) are plain dicts.
    model_msgs = [m for m in messages if not isinstance(m, dict)]
    last = messages[-1]
    run = {
        "answer": "" if isinstance(last, dict) else (last.content or ""),
        "tools": tools_called,
        "steps": len(model_msgs),
        # If the last message is a tool result or a tool request, the loop
        # ended without a final answer → it ran out of steps.
        "hit_max": isinstance(last, dict) or bool(last.tool_calls),
        "seconds": round(seconds, 1),
        # Save the plan file (if any) so you can open eval_results.json and
        # compare what the agent SAID with what it actually WROTE.
        "tasks_md": read_sandbox("notes/tasks.md") or None,
    }

    # -- Grade it. The task's own check runs first because its reason is the
    #    most specific ("retried delete 11 times" says more than "ran out of steps").
    if error:
        run["passed"], run["reason"] = False, f"crashed: {error}"
    else:
        run["passed"], run["reason"] = task["check"](run)
        if run["passed"] and run["hit_max"]:
            run["passed"], run["reason"] = False, "ran out of steps"
    return run


# =============================================================================
# MAIN  —  loop over configs × tasks × repeats, then print the score table
# and save every detail to eval_results.json for later comparison.
# =============================================================================
if __name__ == "__main__":
    all_results = []
    total = len(CONFIGS) * len(TASKS) * REPEATS
    n = 0

    for config in CONFIGS:
        label = f"{config['model']} think={config['think']}"
        print(f"\n=== {label} ===")
        for task in TASKS:
            for r in range(REPEATS):
                n += 1
                print(f"[{n}/{total}] {task['name']:<18} run {r + 1}/{REPEATS} ... ", end="", flush=True)
                run = run_once(task, config)
                mark = "✅" if run["passed"] else "❌"
                print(f"{mark} {run['seconds']:>5}s  {run['steps']} steps  {run['reason']}")
                all_results.append({"config": label, "task": task["name"], **run})

    # -- Score table: one row per task, one column per config.
    labels = [f"{c['model']} think={c['think']}" for c in CONFIGS]
    print("\n" + "=" * 80)
    print(f"{'TASK':<18}" + "".join(f"{l[-24:]:>28}" for l in labels))
    print("-" * 80)
    for task in TASKS:
        row = f"{task['name']:<18}"
        for l in labels:
            rs = [x for x in all_results if x["config"] == l and x["task"] == task["name"]]
            ok = sum(x["passed"] for x in rs)
            avg_s = sum(x["seconds"] for x in rs) / len(rs)
            cell = f"{ok}/{len(rs)}  ({avg_s:.0f}s)"
            row += f"{cell:>28}"
        print(row)
    print("-" * 80)
    row = f"{'TOTAL':<18}"
    for l in labels:
        rs = [x for x in all_results if x["config"] == l]
        ok = sum(x["passed"] for x in rs)
        total_s = sum(x["seconds"] for x in rs)
        cell = f"{ok}/{len(rs)} = {100 * ok // len(rs)}%  ({total_s:.0f}s)"
        row += f"{cell:>28}"
    print(row)

    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print("\nDetails saved to eval_results.json")