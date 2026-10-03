"""Event and streaming-chunk types for the agent harness.

The conversation history is a linear list of immutable ``Event`` objects. Each event
records one thing that happened: the session starting, a user message, an assistant
response, or a batch of tool results. ``parent_id`` links an event to the event that
caused it, forming a tree rooted at a ``SessionStart``.

Tool calls and tool results are not events. They are value objects nested inside the
events that carry them: ``ToolCall`` objects live in ``AssistantMessage.tool_calls``
and ``ToolResult`` objects live in ``ToolResultBatch.results``. A result is paired
with its call through the provider's ``call_id``.

Streaming chunks (``TextDelta``, ``ToolCallStart``, ``ToolCallDelta``) are not events
either. They exist only between the model provider and the accumulator that assembles
them into complete events, and are never stored in the history.
"""

import time
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

# Tool calls and results

class ToolCall(BaseModel):
    """A request from the model to run a tool.

    A single model response can produce several tool calls. They are stored together
    in the ``tool_calls`` tuple of the ``AssistantMessage`` that issued them.

    Attributes:
        call_id: The provider's identifier for this call, which must be sent back
            alongside the result when building the next request.
        name: Name of the tool to run.
        args: Parsed arguments to pass to the tool.
    """

    model_config = ConfigDict(frozen=True)
    call_id: str
    name: str
    args: dict[str, Any]


class ToolResult(BaseModel):
    """The outcome of running a tool, including failures.

    It answers the ``ToolCall`` with the same ``call_id``. A failed call still
    produces a result, with ``is_error`` set and the error described in ``content``.

    Attributes:
        call_id: The ``call_id`` of the ``ToolCall`` this result answers.
        is_error: True if the tool failed.
        content: Text shown to the model, or the error as ``ExceptionType: message``.
        metadata: Free-form extra information about this one result, such as a
            traceback or how long the tool took.
    """

    model_config = ConfigDict(frozen=True)
    call_id: str
    is_error: bool = False
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)

# Events

class Event(BaseModel):
    """Base class for everything stored in the conversation history.

    Events are frozen: once created they cannot be reassigned. Subclasses add a
    ``type`` literal so events can be told apart when serialized.

    Attributes:
        id: Unique identifier of this event.
        timestamp: Creation time as seconds since the Unix epoch.
        parent_id: ID of the event that caused this one. Only ``SessionStart``, the
            root, may leave this as None; every other subclass makes it required.
        metadata: Free-form extra information about the event as a whole, such as
            token counts.
    """

    model_config = ConfigDict(frozen=True)
    id: UUID = Field(default_factory=uuid4)
    timestamp: float = Field(default_factory=time.time)
    parent_id: UUID | None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionStart(Event):
    """The root of the history, marking the start of a session.

    It is the only event without a parent. The first ``UserMessage`` points at it.

    Attributes:
        parent_id: Always None, since nothing caused the session to start.
    """

    parent_id: UUID | None = None
    type: Literal["session_start"] = "session_start"


class UserMessage(Event):
    """A message typed by the user.

    Attributes:
        parent_id: ID of the event this message follows.
        content: The message text.
    """

    parent_id: UUID
    type: Literal["user_message"] = "user_message"
    content: str


class AssistantMessage(Event):
    """A response from the model.

    Attributes:
        parent_id: ID of the event this response answers.
        content: The response text. Empty if the response only called tools.
        tool_calls: Tools the model asked to run in this response, in order. Empty
            if the response made no tool calls.
    """

    parent_id: UUID
    type: Literal["assistant_message"] = "assistant_message"
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()


class ToolResultBatch(Event):
    """The results of every tool call from one assistant message.

    The batch is created only once all of the calls have finished, so it joins the
    results of parallel calls into a single event that the next assistant message
    can point at.

    Attributes:
        parent_id: ID of the ``AssistantMessage`` whose tool calls these answer.
        results: One result for each call in that message.
    """

    parent_id: UUID
    type: Literal["tool_result_batch"] = "tool_result_batch"
    results: tuple[ToolResult, ...]

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
