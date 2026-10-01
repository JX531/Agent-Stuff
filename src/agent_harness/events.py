"""Event and streaming-chunk types for the agent harness.

The conversation history is a linear list of immutable ``Event`` objects. Each event
records one thing that happened: a user message, an assistant response, a tool call,
or a tool result. ``parent_id`` links an event to the event that caused it, which is
how a ``ToolResult`` is paired with its ``ToolCall``.

Streaming chunks (``TextDelta``, ``ToolCallStart``, ``ToolCallDelta``) are not events.
They exist only between the model provider and the accumulator that assembles them into
complete events, and are never stored in the history.
"""

import time
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

# Events


class Event(BaseModel):
    """Base class for everything stored in the conversation history.

    Events are frozen: once created they cannot be reassigned. Subclasses add a
    ``type`` literal so events can be told apart when serialized.

    Attributes:
        id: Unique identifier of this event.
        timestamp: Creation time as seconds since the Unix epoch.
        parent_id: ID of the event that caused this one, or None if there is no
            specific cause (for example a user message).
        metadata: Free-form extra information, such as token counts or tracebacks.
    """

    model_config = ConfigDict(frozen=True)
    id: UUID = Field(default_factory=uuid4)
    timestamp: float = Field(default_factory=time.time)
    parent_id: UUID | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class UserMessage(Event):
    """A message typed by the user.

    Attributes:
        content: The message text.
    """

    type: Literal["user_message"] = "user_message"
    content: str


class ToolCall(Event):
    """A request from the model to run a tool.

    A single model response can produce several tool calls. Each is stored as its own
    event whose ``parent_id`` is the ``AssistantMessage`` that issued it.

    Attributes:
        call_id: The provider's identifier for this call, which must be sent back
            alongside the result when building the next request.
        name: Name of the tool to run.
        args: Parsed arguments to pass to the tool.
    """

    type: Literal["tool_call"] = "tool_call"
    call_id: str
    name: str
    args: dict[str, Any]


class ToolResult(Event):
    """The outcome of running a tool, including failures.

    Its ``parent_id`` is the ``ToolCall`` it answers. A failed call still produces a
    result, with ``is_error`` set and the error described in ``content``.

    Attributes:
        is_error: True if the tool failed.
        content: Text shown to the model, or the error as ``ExceptionType: message``.
    """

    type: Literal["tool_result"] = "tool_result"
    is_error: bool = False
    content: str


class AssistantMessage(Event):
    """A response from the model.

    Tool calls made in the same response are stored as separate ``ToolCall`` events
    that point back to this message through ``parent_id``.

    Attributes:
        content: The response text. Empty if the response only called tools.
    """

    type: Literal["assistant_message"] = "assistant_message"
    content: str = ""


# Streaming deltas


class TextDelta(BaseModel):
    """A fragment of response text received while streaming.

    Attributes:
        text_fragment: The next piece of text, to be appended to the buffer.
    """

    model_config = ConfigDict(frozen=True)
    text_fragment: str


class ToolCallStart(BaseModel):
    """Signals that the model has begun a new tool call in the stream.

    Attributes:
        index: Position of this tool call within the response, used to match
            later argument fragments when several calls stream together.
        call_id: The provider's identifier for this call.
        name: Name of the tool being called.
    """

    model_config = ConfigDict(frozen=True)
    index: int
    call_id: str
    name: str


class ToolCallDelta(BaseModel):
    """A fragment of a tool call's arguments received while streaming.

    Fragments must be concatenated and parsed only once the call is complete.

    Attributes:
        index: Which tool call this fragment belongs to.
        arguments_fragment: Raw JSON text, not yet parsed.
    """

    model_config = ConfigDict(frozen=True)
    index: int
    arguments_fragment: str
