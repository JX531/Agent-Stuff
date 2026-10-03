"""Client for talking to a vLLM server over its OpenAI-compatible HTTP API.
"""

from typing import Any, Iterator
from agent_harness.events import TextDelta, ToolCallStart, ToolCallDelta

def parse_chunk(chunk:dict[str, Any]) -> Iterator[TextDelta | ToolCallStart | ToolCallDelta]:
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
        if call_id: # first chunk
            yield ToolCallStart(index=index, call_id=call_id, name=call_function.get("name"))

        arguments = call_function.get("arguments")
        if arguments:
            yield ToolCallDelta(index=index, arguments_fragment=arguments)