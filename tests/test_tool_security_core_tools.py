"""Agent mode must always offer the core terminal toolkit (cure the tool gating).

Tool selection (tool_index RAG + `_detect_domains`) hides the files/shell domain
from the model unless the user message matches a keyword, so a vague agent prompt
("test", "oi") collapsed to the ~4 ALWAYS_AVAILABLE tools and the agent could not
run a command or read/edit a file. `augment_with_core_tools` force-includes the
core toolkit on every agent turn, with a read-only subset in plan mode and a no-op
in guide_only mode.
"""
from src.tool_security import (
    augment_with_core_tools,
    CORE_AGENT_TOOLS,
    PLAN_MODE_READONLY_TOOLS,
)


def test_normal_agent_always_gets_full_core_toolkit():
    # A vague prompt selects almost nothing; the core toolkit must still be there.
    out = augment_with_core_tools({"ask_user"}, plan_mode=False, guide_only=False)
    for t in ("bash", "python", "read_file", "write_file", "edit_file",
              "grep", "glob", "ls", "get_workspace"):
        assert t in out, t
    assert "ask_user" in out  # the caller's selection is preserved


def test_plan_mode_only_read_only_core():
    out = augment_with_core_tools({"ask_user"}, plan_mode=True, guide_only=False)
    for t in ("read_file", "grep", "glob", "ls", "get_workspace"):
        assert t in out, t
    for mutator in ("bash", "python", "write_file", "edit_file"):
        assert mutator not in out, mutator


def test_guide_only_is_untouched():
    selected = {"ask_user", "manage_memory"}
    out = augment_with_core_tools(set(selected), plan_mode=False, guide_only=True)
    assert out == selected


def test_does_not_mutate_caller_set():
    src = {"ask_user"}
    augment_with_core_tools(src, plan_mode=False, guide_only=False)
    assert src == {"ask_user"}  # returns a new set; caller's set is unchanged


def test_core_toolkit_is_the_files_domain():
    assert {"bash", "python", "read_file", "write_file", "edit_file"} <= set(CORE_AGENT_TOOLS)
    # plan-mode injects exactly the read-only files tools
    assert (set(CORE_AGENT_TOOLS) & PLAN_MODE_READONLY_TOOLS) == {
        "read_file", "grep", "glob", "ls", "get_workspace",
    }
