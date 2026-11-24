"""The agent system prompt: one document for every engine.

``system_prompts/agent.md`` is the whole prompt. It has three
engine-specific slots, filled here:

- ``$engine_title``: the engine's name, for the header;
- ``$tool_naming``: how hylo's tools (and toolkit tools) appear in
  this engine's tool list;
- ``$native_tools``: the engine's own file and shell tools, if any.

The toolkit and model guidance (and the agent's exact toolkit tools)
are appended when the runtime has toolbase or a model catalog.
Every engine builds its prompt through :func:`build_system_prompt`, so
a change to the document reaches all of them.
"""

from __future__ import annotations

import functools
import re
import string
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hylo.record import AgentRecord
    from hylo.runtime import Runtime

PROMPT_PATH = Path(__file__).resolve().parent / "system_prompts" / "agent.md"

_ENGINE_TITLES = {
    "claude_agent": "Claude Code engine",
    "codex": "Codex engine",
    "orchestral": "orchestral engine",
}

_TOOL_NAMING = {
    "claude_agent": (
        "hylo's tools appear in your tool list as `mcp__hylo__<Name>` "
        "(`mcp__hylo__Spawn`, `mcp__hylo__WaitFor`, ...); this prompt "
        "uses the bare names. Toolkit tools appear as "
        "`mcp__toolbase__<toolkit>__<tool>`."
    ),
    "codex": (
        "hylo's tools come from the `hylo` MCP server, as "
        "`mcp__hylo__<Name>` (`mcp__hylo__Spawn`, ...); this prompt "
        "uses the bare names. Toolkit tools appear as "
        "`mcp__hylo__<toolkit>__<tool>`."
    ),
    "orchestral": (
        "Your tools are named as in this prompt (`Spawn`, `Send`, "
        "`WaitFor`, ...). Toolkit tools appear as `<toolkit>__<tool>`."
    ),
}

_CLAUDE_NATIVE = (
    "Claude Code's tools: `Read`, `Write`, `Edit`, `Bash`, `Grep`, `Glob`, "
    "and any others listed in your spec's `tools`. They work inside your "
    "sandbox directory. Use them to read, change and run code."
)
_CODEX_NATIVE = (
    "Codex's shell and file editing. They work inside your sandbox "
    "directory (your working directory): you can read and write there, and "
    "Codex's sandbox blocks writes elsewhere."
)
_ORCHESTRAL_FILES = (
    "`Read`, `Write`, `Edit`, `Bash`, `Grep` and `Glob`. They work inside "
    "your sandbox directory."
)
_ORCHESTRAL_NONE = (
    "None: you have no shell or file tools. Do work that needs them through "
    "a toolkit<!-- spawn -->, or a child granted the right tools<!-- /spawn -->"
    "<!-- worker -->, or ask your parent for it<!-- /worker -->."
)


@functools.cache
def _template() -> string.Template:
    try:
        text = PROMPT_PATH.read_text(encoding="utf-8")
    except OSError:  # a packaging accident; keep agents usable
        text = (
            "You are an agent in the hylo multi-agent framework.\n\n"
            "Identity: address $addr_id, label $label, depth $depth "
            "(max $max_depth), working directory $sandbox.\n\nRole: $role_prompt\n"
        )
    return string.Template(text)


def native_tools(record: "AgentRecord", engine: str) -> str:
    if engine == "claude_agent":
        return _CLAUDE_NATIVE
    if engine == "codex":
        return _CODEX_NATIVE
    groups = set(record.spec.tools or [])
    return _ORCHESTRAL_FILES if groups & {"filesystem", "all"} else _ORCHESTRAL_NONE


def _sandbox(record: "AgentRecord", runtime: "Runtime") -> str:
    try:
        path = runtime.sandbox_of(record.addr)
    except Exception:
        path = None
    return str(path) if path else "(none)"


_SECTION = re.compile(r"<!-- (spawn|nospawn|root|worker) -->\n?(.*?)<!-- /\1 -->\n?", re.S)


def can_spawn(record: "AgentRecord", runtime: "Runtime") -> bool:
    """Whether this agent may create agents: its children would sit within ``max_depth``."""
    return record.depth < getattr(runtime, "max_depth", 3)


def _for_capability(text: str, spawn: bool, root: bool = True) -> str:
    """Keep the sections for what the agent can do: `spawn` (it can create agents) or `nospawn`, and `root`
    (it has no parent, so it serves itself) or `worker` (it asks its parent)."""
    keep = {"spawn": spawn, "nospawn": not spawn, "root": root, "worker": not root}
    while _SECTION.search(text):
        text = _SECTION.sub(lambda m: m.group(2) if keep[m.group(1)] else "", text)
    return text


def build_system_prompt(record: "AgentRecord", runtime: "Runtime", engine: str) -> str:
    spawn = can_spawn(record, runtime)
    prompt = _template().safe_substitute(
        engine_title=_ENGINE_TITLES.get(engine, f"{engine} engine"),
        tool_naming=_TOOL_NAMING.get(engine, _TOOL_NAMING["orchestral"]),
        native_tools=native_tools(record, engine),
        addr_id=record.addr.id,
        label=record.addr.label or "(none)",
        role_prompt=record.spec.role_prompt,
        depth=str(record.depth),
        max_depth=str(getattr(runtime, "max_depth", 3)),
        sandbox=_sandbox(record, runtime),
    )
    root = record.parent is None
    prompt = _for_capability(prompt, spawn, root)
    hub = getattr(runtime, "toolbase", None)
    if hub is not None:
        from hylo.toolbase import TOOLKIT_GUIDANCE, toolkit_brief

        prompt += "\n" + _for_capability(TOOLKIT_GUIDANCE, spawn, root)
        prompt += toolkit_brief(hub, record.spec, engine)
    if spawn and getattr(runtime, "models", None) is not None:
        from hylo.models import MODEL_GUIDANCE

        prompt += "\n" + MODEL_GUIDANCE
    return prompt
