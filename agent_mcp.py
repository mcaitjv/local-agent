# =============================================================================
# agent_mcp.py  —  your agent as an MCP CLIENT
#
# Same agent loop as agent.py, with one big difference: this file contains
# NO tools. When it starts, it:
#   1. launches notes_server.py as a subprocess,
#   2. asks it "what tools do you have?" over MCP,
#   3. hands that list to the model,
#   4. and every time the model wants a tool, sends the call to the server.
#
# Want more tools? Don't edit this file — add another MCP server to SERVERS.
#
# MCP is asynchronous (async/await): talking to a server means waiting for
# its answer, and `await` lets Python wait without freezing. If async is new
# to you, read `await x` as "do x and wait for the result".
#
# Run:   python agent_mcp.py
# =============================================================================

import os
import sys
import json
import asyncio
from datetime import date

import ollama
from mcp import Client, StdioServerParameters


# -----------------------------------------------------------------------------
# SETTINGS
# SERVER: how to start the MCP server: "run notes_server.py with Python".
# This is exactly the information Claude Code stores when you run
# `claude mcp add notes -- <python> <path to notes_server.py>`:
# a command, not code. sys.executable = the Python running this file (your
# .venv), and the full path means it works from any folder.
# -----------------------------------------------------------------------------
MODEL = "qwen3.5:9b"
THINK = True
HERE = os.path.dirname(os.path.abspath(__file__))
SERVER = StdioServerParameters(
    command=sys.executable,
    args=[os.path.join(HERE, "notes_server.py")],
)


# -----------------------------------------------------------------------------
# MCP TOOL → OLLAMA TOOL
# The server describes each tool with a name, a description and a JSON schema
# of its parameters. Ollama wants the same information in "function" format.
# This is just repackaging — no information is added or lost.
# -----------------------------------------------------------------------------
def to_ollama_tool(tool) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.input_schema,
        },
    }


# -----------------------------------------------------------------------------
# DOES THIS TOOL NEED APPROVAL?
# Decided from the server's annotations: anything not marked read-only asks
# you first. Unknown or missing annotations → ask (safe default).
# In agent.py this was a hardcoded NEEDS_APPROVAL set; now the server tells us.
# -----------------------------------------------------------------------------
def needs_approval(tool) -> bool:
    a = tool.annotations
    return not (a and a.read_only_hint)


# -- Ask you y/n. `asyncio.to_thread(input, ...)` runs the blocking input()
#    without freezing the async connection to the server.
async def ask_approval(name: str, args: dict, destructive: bool) -> bool:
    print(f"\n{'🛑' if destructive else '⚠️ '} The agent wants to run: {name}"
          f"{'  (DESTRUCTIVE: can delete or overwrite data)' if destructive else ''}")
    for key, value in args.items():
        text = str(value)
        print(f"   {key}: {text if len(text) < 300 else text[:300] + '...'}")
    answer = await asyncio.to_thread(input, "   Allow? (y/n): ")
    return answer.strip().lower() == "y"


# -----------------------------------------------------------------------------
# CALL ONE TOOL ON THE SERVER (safely, like run_tool in agent.py)
# The server returns a list of "content blocks". For our tools that's one
# text block; we join all text blocks into one string for the model.
# is_error = the server reports the tool itself failed (e.g. bad arguments).
# -----------------------------------------------------------------------------
async def run_tool(client, tools_by_name: dict, name: str, args: dict) -> str:
    tool = tools_by_name.get(name)
    if tool is None:
        return f"Error: unknown tool '{name}'. Available tools: {', '.join(tools_by_name)}"
    if needs_approval(tool):
        destructive = bool(tool.annotations and tool.annotations.destructive_hint)
        if not await ask_approval(name, args, destructive):
            return "User denied this action. Do not retry it; ask the user what they want instead."
    try:
        result = await client.call_tool(name, args)
    except Exception as e:
        return f"Error: could not reach the tool server: {e}"
    text = "\n".join(block.text for block in result.content if getattr(block, "text", None))
    if result.is_error:
        return f"Error from tool: {text}"
    return text or "Error: tool returned nothing."


# =============================================================================
# THE AGENT LOOP  —  identical idea to agent.py; only the tool calls changed.
# =============================================================================
async def run_agent(client, model_client, ollama_tools, tools_by_name, messages, max_steps=12):
    for step in range(1, max_steps + 1):
        print(f"\n--- step {step} ---")
        response = await model_client.chat(
            model=MODEL, messages=messages, tools=ollama_tools,
            think=THINK, options={"num_ctx": 16384},
        )
        msg = response.message
        messages.append(msg)

        if getattr(msg, "thinking", None):
            t = msg.thinking.strip().replace("\n", " ")
            print(f"🧠 {t[:400]}{'...' if len(t) > 400 else ''}")
        if msg.content and msg.tool_calls:
            print("💭", msg.content)
        if not msg.tool_calls:
            print("🤖 ANSWER:", msg.content)
            return

        for call in msg.tool_calls:
            name, args = call.function.name, dict(call.function.arguments)
            short = {k: (str(v)[:60] + '...' if len(str(v)) > 60 else v) for k, v in args.items()}
            print(f"🔧 model wants: {name}({short})   → via MCP")
            result = await run_tool(client, tools_by_name, name, args)
            print(f"📄 result: {result[:200]}")
            messages.append({"role": "tool", "content": result, "tool_name": name})

    print("⚠️ Stopped: reached max steps.")


# =============================================================================
# MAIN
# `async with Client(SERVER)` starts notes_server.py, does the MCP handshake,
# and shuts the server down cleanly when we exit (even after Ctrl+C).
# =============================================================================
async def main():
    async with Client(SERVER) as client:

        # -- 1. DISCOVERY: ask the server for its tools. This is the heart of
        #       MCP: the client learns at runtime what it can do.
        listing = await client.list_tools()
        tools_by_name = {t.name: t for t in listing.tools}
        ollama_tools = [to_ollama_tool(t) for t in listing.tools]

        print(f"🔌 Connected to MCP server '{client.server_info.name}' with {len(listing.tools)} tools:")
        for t in listing.tools:
            print(f"   {'🔒' if needs_approval(t) else '✅'} {t.name}")

        # -- 2. SYSTEM PROMPT = the client's own part (date, role, behavior)
        #       + the server's instructions (how to use ITS tools).
        system = (
            f"Today is {date.today().strftime('%A, %d %B %Y')}. "
            "You are a helpful personal assistant.\n"
            "- If details are missing, make reasonable assumptions, say what you assumed, "
            "and complete the task. Only ask a question if the task is truly impossible without the answer.\n"
            "- Do not ask for permission before using tools: the user is asked automatically when needed.\n"
            "- For tasks with several steps, first write a short numbered plan, then carry it out.\n\n"
            f"Instructions from the '{client.server_info.name}' tool server:\n{client.instructions or '(none)'}"
        )
        messages = [{"role": "system", "content": system}]
        model_client = ollama.AsyncClient()

        # -- 3. CHAT LOOP, same as agent.py
        while True:
            q = await asyncio.to_thread(input, "\nYou: ")
            if q.lower() in ("exit", "quit"):
                break
            messages.append({"role": "user", "content": q})
            await run_agent(client, model_client, ollama_tools, tools_by_name, messages)
            print(f"   (conversation size: {len(messages)} messages)")


if __name__ == "__main__":
    asyncio.run(main())