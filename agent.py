# =============================================================================
# agent.py  —  "version 1": the agent calls the tools DIRECTLY (no MCP)
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
#
# After the restructure this file contains NO tool code. The tools live in
# notes_tools.py; this file imports them and calls them as normal functions.
# Compare with agent_mcp.py, which reaches the SAME tools through MCP.
# eval.py tests this file, because direct calls are easy to test.
# =============================================================================

from datetime import date      # to tell the model today's date
import ollama                  # talks to the Ollama server on localhost:11434
import notes_tools             # all the tools + their risk levels + usage rules


# -----------------------------------------------------------------------------
# SETTINGS
# MODEL: which local model to use.
# THINK: True = the model reasons before answering (slower, usually smarter).
# -----------------------------------------------------------------------------
MODEL = "qwen3.5:9b"
THINK = True


# -----------------------------------------------------------------------------
# TOOLS, built from the catalog in notes_tools.py
# TOOLS maps a name ("read_file") to the real function, so when the model says
# "call read_file" we can look it up. NEEDS_APPROVAL = every tool that is not
# "read". Both come from notes_tools.TOOLS, so there's nothing to keep in sync.
# -----------------------------------------------------------------------------
TOOLS = {fn.__name__: fn for fn in notes_tools.TOOLS}
NEEDS_APPROVAL = {fn.__name__ for fn, risk in notes_tools.TOOLS.items() if risk != "read"}


# -----------------------------------------------------------------------------
# SYSTEM PROMPT  =  the agent's own rules  +  the tools' usage rules.
# Agent rules: who it is, today's date, assume-vs-ask, how to plan.
# Tool rules: come from notes_tools.TOOL_INSTRUCTIONS (the same text the MCP
# server sends to its clients), so both versions of the agent get the same rules.
# -----------------------------------------------------------------------------
SYSTEM = {"role": "system", "content": (
    f"Today is {date.today().strftime('%A, %d %B %Y')}. "
    "You are a helpful personal assistant that manages the user's notes.\n"
    "- If details are missing, make reasonable assumptions, say what you assumed, "
    "and complete the task. Only ask a question if the task is truly impossible without the answer.\n"
    "- Do not ask for permission before using tools: the user is asked automatically when needed.\n"
    "- For tasks with several steps, first write a short numbered plan, then carry it out step by step.\n\n"
    + notes_tools.TOOL_INSTRUCTIONS
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