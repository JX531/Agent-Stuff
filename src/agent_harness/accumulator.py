"""Accumulator responsible assembling Deltas into completed Events"""

import json
from typing import Any
from uuid import UUID

from events import AssistantMessage, TextDelta, ToolCall, ToolCallDelta, ToolCallStart


class Accumulator:
    """Assembles streaming deltas into one complete ``AssistantMessage``.

    Feed deltas one at a time with ``feed``, then call ``build`` once the stream
    has ended. Tool calls are tracked by stream ``index`` because several can
    stream interleaved. Their argument fragments are kept as raw text and parsed
    as JSON only in ``build``, since a partial fragment is not valid JSON.

    Attributes:
        texts: Text fragments received so far, in arrival order.
        tool_calls: In-progress tool calls keyed by stream index. Each value holds
            ``call_id``, ``name`` and ``args`` (a list of raw JSON fragments).
    """

    def __init__(self):
        """Builds an empty Accumulator"""
        self.texts: list[str] = []
        self.tool_calls: dict[int, Any] = {}

    def feed(self, delta: TextDelta | ToolCallStart | ToolCallDelta):
        """Absorb one streaming delta into the accumulated state.

        Args:
            delta: The next chunk from ``VLLMClient.stream_chat``.

        Raises:
            ValueError: If a ``ToolCallDelta`` arrives for an index that never had
                a ``ToolCallStart``.
        """

        if isinstance(delta, TextDelta):
            self.texts.append(delta.text_fragment)

        elif isinstance(delta, ToolCallStart):
            self.tool_calls[delta.index] = {
                "call_id": delta.call_id,
                "name": delta.name,
                "args": [],
            }

        elif isinstance(delta, ToolCallDelta):
            if delta.index not in self.tool_calls:
                raise ValueError(
                    f"Received args fragment for unknown tool call index {delta.index}"
                )

            self.tool_calls[delta.index]["args"].append(delta.arguments_fragment)

    def build(self, parent_id: UUID) -> AssistantMessage:
        """
        Turn the accumulated state into a complete event.

        Call this once, after the stream has ended.

        Args:
            parent_id: ID of the event the model was responding to.

        Returns:
            The finished ``AssistantMessage``, with tool calls in index order.

        Raises:
            json.JSONDecodeError: If a call's complete arguments are not valid JSON.
        """
        tool_calls = tuple(
            ToolCall(
                call_id=call["call_id"],
                name=call["name"],
                args=json.loads("".join(call["args"]) or {}),
            )
            for _, call in self.tool_calls.items()
        )

        return AssistantMessage(
            parent_id=parent_id, content="".join(self.texts), tool_calls=tool_calls
        )
