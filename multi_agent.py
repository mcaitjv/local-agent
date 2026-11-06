# =============================================================================
# multi_agent.py  —  four specialist agents working together
#
# Instead of ONE agent that does everything, we use a team:
#   🔍 researcher  finds the facts            (read-only tools)
#   📝 planner     writes the draft           (week_ranges only)
#   🧐 reviewer    checks the draft           (week_ranges only)
#   💾 writer      saves the approved draft   (write_note only)
#
# Each agent has its own role, its own tools, its own memory and its own
# settings. They never see each other's conversations: they only pass TEXT
# along (facts → draft → review → saved file). That's called a "handoff".
#
# Who decides the order? Plain Python code (the ORCHESTRATOR at the bottom),
# not a model. A fixed workflow is much more reliable with a small model than
# letting the model decide who to call next.
#
# All four agents share ONE connection to notes_server.py (MCP).
#
# Run:   python multi_agent.py
# Try:   Make a weekly plan in tasks.md until the project deadline.
# =============================================================================

import asyncio
from datetime import date

import ollama
from mcp import Client

# Reuse the MCP plumbing from agent_mcp.py instead of copying it:
# how to start the server, how to convert tools for Ollama, and how to call
# a tool safely (with the approval prompt for anything not read-only).
from agent_mcp import SERVER, MODEL, to_ollama_tool, run_tool


# -----------------------------------------------------------------------------
# THE AGENT CLASS
# One class, four instances. What makes each agent different is only:
#   role:   its system prompt (who it is, what it must and must NOT do)
#   tools:  the names of the tools it may use (a subset of the server's 9)
#   think:  whether it reasons before answering (slower, smarter)
# messages: its OWN conversation. Agents never share this list.
# -----------------------------------------------------------------------------
class Agent:
    def __init__(self, name: str, emoji: str, role: str, tools: list, think: bool):
        self.name, self.emoji, self.think = name, emoji, think
        self.tool_names = tools
        self.role = role
        self.reset()

    # -- Start a fresh conversation (used at the start of each new request).
    def reset(self):
        self.messages = [{"role": "system", "content": (
            f"Today is {date.today().strftime('%A, %d %B %Y')}.\n{self.role}")}]

    # -- Called once after connecting: keep only the tools this agent is
    #    allowed to use. LEAST PRIVILEGE: if the reviewer tries to call
    #    delete_note, run_tool answers "unknown tool", because for this
    #    agent it simply doesn't exist.
    def connect(self, client, all_tools: dict):
        self.client = client
        self.tools = {n: all_tools[n] for n in self.tool_names}
        self.ollama_tools = [to_ollama_tool(t) for t in self.tools.values()]

    # -- The same agent loop as always, but it RETURNS the final answer as text
    #    (so the orchestrator can hand it to the next agent) and prints with
    #    the agent's emoji so you can see who is doing what.
    async def run(self, task: str, max_steps: int = 8) -> str:
        print(f"\n{self.emoji} {self.name.upper()}  (think={self.think}, tools={self.tool_names or 'none'})")
        self.messages.append({"role": "user", "content": task})
        for _ in range(max_steps):
            response = await ollama.AsyncClient().chat(
                model=MODEL, messages=self.messages, tools=self.ollama_tools or None,
                think=self.think, options={"num_ctx": 16384},
            )
            msg = response.message
            self.messages.append(msg)
            if not msg.tool_calls:
                answer = (msg.content or "").strip()
                preview = answer if len(answer) < 500 else answer[:500] + " ..."
                print("   " + preview.replace("\n", "\n   "))
                return answer
            for call in msg.tool_calls:
                name, args = call.function.name, dict(call.function.arguments)
                print(f"   🔧 {name}({ {k: (str(v)[:50] + '...' if len(str(v)) > 50 else v) for k, v in args.items()} })")
                result = await run_tool(self.client, self.tools, name, args)
                # Show a one-line result. Errors and blocks are flagged so you can
                # SEE least privilege at work (e.g. the reviewer trying to delete).
                first = result.splitlines()[0] if result else ""
                flag = "⛔" if first.startswith(("Error", "User denied")) else "📄"
                print(f"      {flag} {first[:110]}")
                self.messages.append({"role": "tool", "content": result, "tool_name": name})
        return "(stopped: too many steps)"


# =============================================================================
# THE TEAM
# Notice how SHORT and STRICT each role is. A specialist with one job and a
# clear "do NOT" list behaves much more predictably than one big prompt that
# tries to cover everything.
# think: only the planner and reviewer need to reason. The researcher and
# writer do simple jobs, so we keep them fast (think=False).
# =============================================================================
researcher = Agent(
    "researcher", "🔍", think=False,
    tools=["search_notes", "read_file", "list_files", "recall"],
    role=("You are a RESEARCHER. Find every fact in the user's notes and memory that is "
          "needed for the request. Use your tools; never guess.\n"
          "Reply with a short bullet list of facts, each with its source file.\n"
          "Do NOT make plans, do NOT write files, do NOT give advice."),
)

planner = Agent(
    "planner", "📝", think=True,
    tools=["week_ranges"],
    role=("You are a PLANNER. You get a request and a list of facts. Write the complete "
          "content of the requested note in markdown.\n"
          "- For any schedule, call week_ranges and use exactly the weeks it returns.\n"
          "- If details are missing, make reasonable assumptions and list them at the top.\n"
          "- Reply with ONLY the note content. Do NOT save it; another agent does that."),
)

reviewer = Agent(
    "reviewer", "🧐", think=True,
    tools=["week_ranges"],
    role=("You are a strict REVIEWER. You did not write the draft; your job is to find its mistakes.\n"
          "Check: (1) it answers the request, (2) it matches the facts, (3) every week from "
          "week_ranges is present with the exact dates, (4) nothing is scheduled after a deadline.\n"
          "Call week_ranges yourself to compare the dates.\n"
          "Your reply MUST start with exactly APPROVED or REJECTED on the first line. "
          "If REJECTED, list each problem on its own line."),
)

writer = Agent(
    "writer", "💾", think=False,
    tools=["write_note"],
    role=("You are a WRITER. Save the given content to the note file named in the request, "
          "using write_note, exactly as given (do not change the text). "
          "Then reply with one sentence saying what you saved, or what went wrong."),
)

TEAM = [researcher, planner, reviewer, writer]
MAX_REVISIONS = 2


# =============================================================================
# THE ORCHESTRATOR  —  plain code that runs the workflow.
# Each step passes TEXT to the next agent. The agents never see each other's
# tool calls or thinking, only the final text handed over. That keeps every
# context small and focused.
# =============================================================================
async def handle(request: str):
    for agent in TEAM:
        agent.reset()   # fresh memory for every new request

    # -- 1. RESEARCH
    facts = await researcher.run(f"Request: {request}\nCollect the facts needed for this request.")

    # -- 2. DRAFT
    draft = await planner.run(f"Request: {request}\n\nFacts:\n{facts}\n\nWrite the note content.")

    # -- 3. REVIEW LOOP: reviewer checks, planner fixes, at most MAX_REVISIONS times.
    #    The planner keeps its memory between rounds, so it knows its own draft
    #    and only has to fix the reported problems.
    approved = False
    for round_ in range(MAX_REVISIONS + 1):
        verdict = await reviewer.run(
            f"Request: {request}\n\nFacts:\n{facts}\n\nDraft to review:\n{draft}")
        approved = verdict.strip().upper().startswith("APPROVED")
        if approved:
            print("\n✅ Reviewer approved the draft.")
            break
        if round_ == MAX_REVISIONS:
            print(f"\n⚠️  Still rejected after {MAX_REVISIONS} revisions.")
            break
        print(f"\n🔁 Revision {round_ + 1}: sending the reviewer's feedback to the planner.")
        reviewer.reset()   # next review starts fresh, judging only the new draft
        draft = await planner.run(
            f"The reviewer rejected your draft:\n{verdict}\n\n"
            "Fix every problem and reply with the complete corrected note content only.")

    # -- 4. SAVE. If the draft never got approved, a human decides.
    if not approved:
        answer = await asyncio.to_thread(input, "   Save the unapproved draft anyway? (y/n): ")
        if answer.strip().lower() != "y":
            print("Nothing saved.")
            return
    await writer.run(f"Request: {request}\n\nContent to save:\n{draft}")


# =============================================================================
# MAIN  —  one MCP connection, shared by the whole team.
# =============================================================================
async def main():
    async with Client(SERVER) as client:
        all_tools = {t.name: t for t in (await client.list_tools()).tools}
        for agent in TEAM:
            agent.connect(client, all_tools)
        print(f"🔌 Connected to '{client.server_info.name}' ({len(all_tools)} tools). Team:")
        for agent in TEAM:
            print(f"   {agent.emoji} {agent.name:<10} → {', '.join(agent.tool_names)}")

        while True:
            request = await asyncio.to_thread(input, "\nYou: ")
            if request.lower() in ("exit", "quit"):
                break
            await handle(request)


if __name__ == "__main__":
    asyncio.run(main())