"""Client for talking to a vLLM server over its OpenAI-compatible HTTP API."""

import json
from collections.abc import Iterator
from typing import Any

from agent_harness.events import (
    AssistantMessage,
    Event,
    SessionStart,
    TextDelta,
    ToolCallDelta,
    ToolCallStart,
    ToolResultBatch,
    UserMessage,
)


def parse_chunk(chunk: dict[str, Any]) -> Iterator[TextDelta | ToolCallStart | ToolCallDelta]:
    """Turn one decoded SSE JSON chunk into zero or more streaming deltas.

    A tool call's first fragment carries ``id`` and ``function.name`` (a start);
    later fragments carry only ``index`` and a piece of the argument JSON.

    Args:
        chunk: A decoded ``chat.completion.chunk`` object.

    Yields:
        ``TextDelta``, ``ToolCallStart`` or ``ToolCallDelta`` instances.
    """

    choices = chunk.get("choices") or []
    if not choices:
        return

    delta = choices[0].get("delta") or {}

    text = delta.get("content")
    if text:
        yield TextDelta(text_fragment=text)

    tool_calls = delta.get("tool_calls") or []
    for call in tool_calls:
        call_id = call.get("id")
        index = call.get("index")
        call_function = call.get("function")
        if call_id:  # first chunk
            yield ToolCallStart(index=index, call_id=call_id, name=call_function.get("name"))

        arguments = call_function.get("arguments")
        if arguments:
            yield ToolCallDelta(index=index, arguments_fragment=arguments)


def events_to_messages(events: list[Event]) -> list[dict[str, Any]]:
    """Convert a linear event history into OpenAI-style chat messages.

    Tool calls are read from each ``AssistantMessage`` and tool results from each
    ``ToolResultBatch``; the two are paired by the provider's ``call_id``.
    ``SessionStart`` has no wire equivalent and is skipped.

    Args:
        events: Events along a single path through the history, oldest first.

    Returns:
        A list of message dicts ready to send as the ``messages`` field.

    Raises:
        TypeError: If an event of an unrecognized type is encountered.
    """

    messages: list[dict[str, Any]] = []
    for event in events:
        if isinstance(event, SessionStart):
            continue

        elif isinstance(event, UserMessage):
            messages.append({"role": "user", "content": event.content})

        elif isinstance(event, AssistantMessage):
            message: dict[str, Any] = {"role": "assistant", "content": event.content}
            if event.tool_calls:
                message["tool_calls"] = [
                    {
                        "id": tool_call.call_id,
                        "type": "function",
                        "function": {
                            "name": tool_call.name,
                            "arguments": json.dumps(tool_call.args),
                        },
                    }
                    for tool_call in event.tool_calls
                ]

                if not event.content:
                    message["content"] = None

            messages.append(message)

        elif isinstance(event, ToolResultBatch):
            for result in event.results:
                messages.append(
                    {"role": "tool", "tool_call_id": result.call_id, "content": result.content}
                )

        else:
            raise TypeError(f"{type(event).__name__} is not a recognized message event type.")

    return messages
