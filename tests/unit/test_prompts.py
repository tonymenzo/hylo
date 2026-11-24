"""One system prompt document for every engine, with engine slots."""

from __future__ import annotations

import pytest

from hylo.engines.registry import build_tools
from hylo.prompts import build_system_prompt
from hylo.record import AgentSpec
from hylo.runtime import Runtime


@pytest.mark.parametrize("engine,title,naming", [
    ("claude_agent", "(Claude Code engine)", "mcp__hylo__<Name>"),
    ("codex", "(Codex engine)", "`hylo` MCP server"),
    ("orchestral", "(orchestral engine)", "named as in this prompt"),
])
def test_every_engine_builds_from_the_same_document(engine, title, naming):
    rt = Runtime()
    rec = rt.record_for(rt.root(AgentSpec(role_prompt="study Z decays", label="root")))
    p = build_system_prompt(rec, rt, engine)
    assert title in p and naming in p and "study Z decays" in p
    # Shared sections, and none of the drift the old per-engine prompts had.
    for section in ("## How messaging works", "## Tools available to you", "## When to stop", "## Child lifecycle events"):
        assert section in p
    assert "Spawn multi-agent framework" not in p and "default to the **haiku**" not in p
    assert "$" + "native_tools" not in p and "$" + "tool_naming" not in p
    rt.shutdown()


def test_orchestral_says_whether_it_has_file_tools():
    rt = Runtime()
    root = rt.root(AgentSpec(role_prompt="r"))
    bare = rt.record_for(rt._spawn(parent=root, spec=AgentSpec(role_prompt="c", tools=["primitive"])))
    fs = rt.record_for(rt._spawn(parent=root, spec=AgentSpec(role_prompt="c", tools=["primitive", "filesystem"])))
    assert "no shell or file tools" in build_system_prompt(bare, rt, "orchestral")
    assert "`Bash`" in build_system_prompt(fs, rt, "orchestral")
    rt.shutdown()


def test_tool_names_match_the_prompt_on_orchestral():
    names = [t.get_name() for t in build_tools("t", ["all"])]
    for n in ("Spawn", "WaitFor", "Recv", "AgentMap", "ToolCatalog", "GrantToolkits", "ModelCatalog", "Bash"):
        assert n in names


def test_an_agent_that_cannot_spawn_is_not_taught_to():
    """Depth at max_depth: no Spawn/Call/combinator sections, no child supervision, the note on working
    without children instead; messaging stays. An agent that can spawn gets the reverse."""
    for max_depth, spawns in ((0, False), (2, True)):
        rt = Runtime(max_depth=max_depth)
        rec = rt.record_for(rt.root(AgentSpec(role_prompt="r", label="root")))
        p = build_system_prompt(rec, rt, "orchestral")
        for text in ("**Reach for `Call` first.**", "AgentMap(spec, items", "## Child lifecycle events",
                     "## When to spawn vs. when to do it yourself", "## Common patterns"):
            assert (text in p) == spawns, (max_depth, text)
        assert ("## Working without children" in p) == (not spawns)
        for text in ("`Send(to, body)`", "`Recv(", "`WaitFor(", "`ListInbox(", "## When to stop"):
            assert text in p
        assert "<!-- spawn -->" not in p and "<!-- nospawn -->" not in p
        rt.shutdown()


def test_a_worker_at_the_depth_limit_asks_its_parent_and_a_root_serves_itself():
    from hylo.prompts import _for_capability
    from hylo.toolbase import TOOLKIT_GUIDANCE

    rt = Runtime(max_depth=1)
    root = rt.root(AgentSpec(role_prompt="r", label="root"))
    leaf = rt.record_for(rt._spawn(parent=root, spec=AgentSpec(role_prompt="w", tools=["primitive"])))
    p = build_system_prompt(leaf, rt, "orchestral")
    assert "## Working without children" in p and "**Reach for `Call` first.**" not in p
    assert "a child granted the right tools" not in p and "ask your parent for it" in p
    worker = _for_capability(TOOLKIT_GUIDANCE, spawn=False, root=False)
    solo = _for_capability(TOOLKIT_GUIDANCE, spawn=False, root=True)
    team = _for_capability(TOOLKIT_GUIDANCE, spawn=True, root=True)
    assert 'GrantToolkits(child="self", toolkits)' not in worker and "ask it with `RequestToolkits`" in worker
    assert 'GrantToolkits(child="self", toolkits)' in solo and "Spawn(..., toolkits" not in solo
    assert "Spawn(..., toolkits" in team and "mainly for the agents you spawn" not in team
    for text in (worker, solo, team):
        assert "<!--" not in text
    rt.shutdown()
