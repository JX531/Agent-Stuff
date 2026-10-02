"""Tests for the event and streaming-chunk models in ``events``."""

from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from agent_harness.events import (
    AssistantMessage,
    Event,
    TextDelta,
    ToolCall,
    ToolCallDelta,
    ToolCallStart,
    ToolResult,
    UserMessage,
)


def test_event_defaults_are_generated() -> None:
    """A new event gets a UUID, a timestamp, no parent and empty metadata."""
    event = UserMessage(content="hi")

    assert isinstance(event.id, UUID)
    assert event.timestamp > 0
    assert event.parent_id is None
    assert event.metadata == {}


def test_event_ids_and_metadata_are_not_shared() -> None:
    """Each event has its own id and its own metadata dict."""
    first = UserMessage(content="a")
    second = UserMessage(content="b")

    assert first.id != second.id
    assert first.metadata is not second.metadata


def test_events_are_frozen() -> None:
    """Assigning to a field of an event raises a validation error."""
    event = UserMessage(content="hi")

    with pytest.raises(ValidationError):
        event.content = "changed"


def test_assistant_message_content_defaults_to_empty() -> None:
    """An assistant message that only calls tools may omit its content."""
    assert AssistantMessage().content == ""


def test_tool_result_is_error_defaults_to_false() -> None:
    """A tool result is a success unless marked otherwise."""
    assert ToolResult(content="ok").is_error is False


def test_required_fields_are_enforced() -> None:
    """Omitting a required field raises a validation error."""
    with pytest.raises(ValidationError):
        ToolCall(name="search", args={})  # missing call_id


@pytest.mark.parametrize(
    ("event", "expected_type"),
    [
        (UserMessage(content="hi"), "user_message"),
        (AssistantMessage(), "assistant_message"),
        (ToolCall(call_id="c1", name="t", args={}), "tool_call"),
        (ToolResult(content="ok"), "tool_result"),
    ],
)
def test_type_literal_is_set_and_serialized(event: Event, expected_type: str) -> None:
    """Each subclass carries its type literal into the serialized output."""
    assert event.type == expected_type
    assert event.model_dump()["type"] == expected_type


def test_tool_result_pairs_with_tool_call_via_parent_id() -> None:
    """A tool result links back to its call through ``parent_id``."""
    call = ToolCall(call_id="c1", name="search", args={"q": "x"})
    result = ToolResult(parent_id=call.id, content="found")

    assert result.parent_id == call.id


def test_json_round_trip_preserves_event() -> None:
    """Serializing to JSON and back yields an equal event."""
    original = ToolCall(
        call_id="c1",
        name="search",
        args={"q": "x"},
        parent_id=uuid4(),
        metadata={"tokens": 12},
    )

    restored = ToolCall.model_validate_json(original.model_dump_json())

    assert restored == original


def test_streaming_deltas_hold_fields_and_are_frozen() -> None:
    """Streaming chunks store their fields and cannot be reassigned."""
    text = TextDelta(text_fragment="Hel")
    start = ToolCallStart(index=0, call_id="c1", name="search")
    delta = ToolCallDelta(index=0, arguments_fragment='{"q"')

    assert text.text_fragment == "Hel"
    assert (start.index, start.call_id, start.name) == (0, "c1", "search")
    assert delta.arguments_fragment == '{"q"'
    with pytest.raises(ValidationError):
        text.text_fragment = "other"
