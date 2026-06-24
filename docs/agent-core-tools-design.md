# Agent core tools always-on (cure the gating) + Claude Code engine (phase 2)

## Context / problem

Agent mode (`mode=agent`) feels dumb because tool selection HIDES the core agentic
toolkit from the model. `src/tool_index.py:ALWAYS_AVAILABLE` is deliberately tiny
(`manage_memory, ask_user, update_plan, spawn_agents`). The shell/file toolkit —
the `files` domain in `src/agent_loop.py:_DOMAIN_TOOL_MAP`
(`{bash, python, read_file, write_file, edit_file, grep, glob, ls, get_workspace}`)
— only reaches the model when RAG retrieval, a keyword hint, or `_detect_domains`
"files" fires. So a vague agent prompt ("test", "oi", "me ajuda com isso")
collapses to the ~4 always-available tools, and the agent cannot run a command or
read/edit a file. It is not a weak agent; it is a gated one (same Opus 4.8 via the
Claude subscription path in `claude_subscription.py`).

Goal: in agent mode, the core terminal toolkit is ALWAYS available, so the agent
behaves like a real coding/terminal agent, while domain tools (email/calendar/web/…)
keep being RAG-selected on top to bound prompt size.

## Phase 1 — Always-on core tools (this change)

### Design
- `CORE_AGENT_TOOLS` = the `files` domain set (`_DOMAIN_TOOL_MAP["files"]`).
- A small PURE helper, e.g. in `src/tool_security.py` (it already owns
  `PLAN_MODE_READONLY_TOOLS`):
  `augment_with_core_tools(relevant: set[str], *, plan_mode: bool, guide_only: bool) -> set[str]`
  - `guide_only` → return `relevant` unchanged (advisory mode never force-adds
    executable tools).
  - `plan_mode` → union `CORE_AGENT_TOOLS & PLAN_MODE_READONLY_TOOLS`
    (`{read_file, grep, glob, ls, get_workspace}`; bash/python/write/edit stay out,
    and the plan-mode denylist blocks mutators at execution anyway).
  - otherwise (normal agent) → union the full `CORE_AGENT_TOOLS`.
- Call it in `stream_agent_loop` (`src/agent_loop.py`) right after `_relevant_tools`
  is finalized (after the RAG/keyword/domain assembly, before the round loop). The
  RAG/domain/keyword logic is UNCHANGED — it keeps ADDING domain tools on top.

### Why this is safe / surgical
- Only agent execution turns are affected: `stream_agent_loop` runs for
  `mode=agent`; chat mode never calls it.
- plan_mode keeps its read-only guarantee (union only the read-only subset; and
  `execute_tool_block`'s plan denylist still blocks mutators).
- guide_only is unaffected.
- Token cost: +1 domain's schemas (the `files` sections) per agent turn. Accepted —
  this is the decided "shell sempre presente no modo agente" fix. We deliberately do
  NOT make every domain always-on (that was the original token-budget rationale).
- The existing partial branch (`agent_loop.py` ~1942: "low-signal + workspace active
  → read-only files") becomes a subset of this and is left in place (harmless).

### Prompt check
`_assemble_prompt` derives tool sections from the final tool names, so once the
files tools are in the set their sections appear automatically. Confirm no prompt
line tells the model to disclaim shell/file capability it now has (the gating was by
omission, not explicit denial — verify and fix if present).

### Tasks
1. Add `CORE_AGENT_TOOLS` + `augment_with_core_tools()` (pure) with unit tests.
2. Wire the helper into `stream_agent_loop` after tool selection.
3. Fix any prompt disclaimer that contradicts always-on tools (if found).
4. Run focused tests (tool selection / agent / services area) + relevant suite.

### Verification (machine-checkable)
- Unit: `augment_with_core_tools({"ask_user"}, plan_mode=False, guide_only=False)`
  ⊇ `{bash, read_file, write_file, edit_file, python, grep, glob, ls, get_workspace}`;
  with `plan_mode=True` ⊇ `{read_file, grep, glob, ls}` and ∌ `{bash, python,
  write_file, edit_file}`; with `guide_only=True` returns the input unchanged.
- Integration: a vague agent query's selected tool set includes the core toolkit
  (mirror the existing tool-selection test pattern; stub embeddings if needed).
- Suite: the agent / services focus area is green.

## Phase 2 — Claude Code engine (designed, DEFERRED per "depois")

A new opt-in agent engine that runs the real `claude` CLI per chat session:
- **Auth:** materialize the subscription OAuth token (already stored encrypted in
  `ProviderAuthSession`, refreshed by `claude_subscription.resolve_runtime_credentials`)
  into the CLI's credential file before each run; reuse the refresh path.
- **Tools:** expose Odysseus's domain tools (email/calendar/recall/notes/…) to the
  CLI via an MCP server (the repo already has `mcp_servers/`); the CLI keeps its
  native Bash/Read/Edit/Glob/Grep/WebSearch.
- **Stream bridge:** spawn `claude -p --output-format stream-json --resume <id>`;
  translate stream-json events (assistant deltas, tool_use, tool_result, thinking,
  result, usage) into Odysseus's existing SSE shapes so the current vanilla-JS
  frontend renders them with no UI rewrite.
- **Session:** map Odysseus `session_id` ↔ a CLI session id for `--resume`.
- **Sandbox/permissions:** headless (no interactive approval) → a restricted
  `--permission-mode` + Claude Code sandbox settings; cwd = a per-session workspace
  under `DATA_DIR` (or a configured repo).
- **Infra:** add Node + `@anthropic-ai/claude-code` to the Docker image
  (`python:3.14-slim` ships no Node).
- **Mode:** a third engine alongside chat/agent (e.g. `engine=claude-code` toggle).

This phase is a multi-day build with image + security work and is NOT implemented in
this change. Phase 1 ships the immediate "stop being dumb" win on its own.
