"""
LLM adapter for the NLP frontend.

Addresses PR #9 review item 2: the previous version left call_claude()
raising NotImplementedError despite listing `anthropic` as a dependency.

This module isolates all network/API code behind one function shape:

    adapter(system_prompt: str, user_prompt: str) -> str

nlp_to_mps.py never imports anthropic or touches the network directly --
callers always inject an adapter explicitly. Current state:

  - `claude_adapter` is a real, working call against the Anthropic API.
    It requires ANTHROPIC_API_KEY to be set in the environment.
  - `make_canned_adapter` builds a zero-network, deterministic adapter
    for tests and offline development. This is what the test suite uses
    exclusively (PR #9 review item 7) -- CI never calls the real API or
    consumes credits.
"""

import os


class AdapterError(RuntimeError):
    pass


def claude_adapter(system_prompt: str, user_prompt: str) -> str:
    """Real Claude API call. Requires `anthropic` installed and
    ANTHROPIC_API_KEY set in the environment."""
    try:
        import anthropic
    except ImportError as e:
        raise AdapterError("anthropic package not installed; run `pip install anthropic`") from e

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise AdapterError("ANTHROPIC_API_KEY is not set in the environment")

    client = anthropic.Anthropic()
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return msg.content[0].text


def make_canned_adapter(canned_response: str):
    """Returns an adapter that always returns `canned_response`,
    regardless of input, and never touches the network. Use this in
    tests to mock the LLM call deterministically."""

    def _adapter(system_prompt: str, user_prompt: str) -> str:
        return canned_response

    return _adapter