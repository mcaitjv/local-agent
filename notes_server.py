# =============================================================================
# notes_server.py  —  publishes notes_tools.py over MCP
#
# What this file does:
#   Takes every tool in notes_tools.TOOLS and offers it over MCP (Model Context
#   Protocol). Any MCP client can start this program, ask "what tools do you
#   have?", and call them: agent_mcp.py, Claude Code, Claude Desktop, an IDE...
#
# What this file does NOT do:
#   No model, no agent loop, no chat, and no tool LOGIC either. The logic is in
#   notes_tools.py. This file is only the "MCP adapter" around it.
#
# Dependencies (notice: no agent.py anywhere):
#   notes_server.py → notes_tools.py → rag.py
#
# How it talks: "stdio" transport. The client starts this file as a
# subprocess and they exchange JSON messages through stdin/stdout.
# That has one big consequence → see the PRINTING section below.
#
# You normally don't run this file yourself; the client starts it.
# =============================================================================

import sys
import functools
import contextlib

from mcp.server.mcpserver import MCPServer   # MCP SDK 2.x (was FastMCP in 1.x)
from mcp.types import ToolAnnotations

import notes_tools   # the actual tool code + risk levels + usage rules


# -----------------------------------------------------------------------------
# THE SERVER OBJECT
# name:         how clients see this server ("notes")
# instructions: text sent to every client when it connects. It's the same
#               TOOL_INSTRUCTIONS agent.py uses, so every client (yours,
#               Claude Code...) learns the same rules for these tools.
# -----------------------------------------------------------------------------
server = MCPServer(name="notes", instructions=notes_tools.TOOL_INSTRUCTIONS)


# -----------------------------------------------------------------------------
# PRINTING  (the #1 beginner bug with stdio MCP servers!)
# stdout is the wire to the client. Every byte on stdout must be a valid MCP
# JSON message. Our tools print things like "📚 indexed 4 chunks" (rag.py) —
# if that went to stdout, the client would receive garbage and disconnect.
# So every tool runs inside redirect_stdout(sys.stderr): its prints go to
# stderr instead, which is just a log the client shows or ignores.
# functools.wraps keeps the original name, docstring and parameters, because
# the MCP SDK reads those to build the tool's description and JSON schema.
# -----------------------------------------------------------------------------
def quiet(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with contextlib.redirect_stdout(sys.stderr):
            return fn(*args, **kwargs)
    return wrapper


# -----------------------------------------------------------------------------
# RISK LEVEL → MCP ANNOTATIONS
# notes_tools.TOOLS says "read", "write" or "destructive" for each tool.
# MCP expresses that as annotations, which the server sends to clients:
#   read_only_hint=True    → only looks at things; safe to run automatically
#   destructive_hint=True  → may delete or overwrite data
# The CLIENT decides what to do with them (agent_mcp.py asks you to approve
# everything that isn't read-only). Annotations are hints, not security:
# a client should only trust servers it trusts.
# -----------------------------------------------------------------------------
ANNOTATIONS = {
    "read":        ToolAnnotations(read_only_hint=True),
    "write":       ToolAnnotations(read_only_hint=False, destructive_hint=False),
    "destructive": ToolAnnotations(read_only_hint=False, destructive_hint=True),
}

# Register every tool. The SDK turns each function's name, type hints and
# docstring into an MCP tool definition (name + description + JSON schema).
for fn, risk in notes_tools.TOOLS.items():
    server.add_tool(quiet(fn), annotations=ANNOTATIONS[risk])


# =============================================================================
# START  —  run as a stdio server and wait for a client to talk to us.
# =============================================================================
if __name__ == "__main__":
    print(f"notes MCP server starting ({len(notes_tools.TOOLS)} tools, "
          f"data in {notes_tools.BASE_DIR})...", file=sys.stderr)   # stderr, never stdout!
    server.run(transport="stdio")