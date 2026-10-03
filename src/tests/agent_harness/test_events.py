"""Tests for the event and streaming-chunk models in ``events``."""

from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from agent_harness.events import (
    AssistantMessage,
    Event,
    SessionStart,
    TextDelta,
    ToolCall,
    ToolCallDelta,
    ToolCallStart,
    ToolResult,
    ToolResultBatch,
    UserMessage,
)

PARENT = uuid4()


def test_event_defaults_are_generated() -> None:
    """A new event gets a UUID, a timestamp and empty metadata."""
    event = SessionStart()

    assert isinstance(event.id, UUID)
    assert event.timestamp > 0
    assert event.metadata == {}


def test_event_ids_and_metadata_are_not_shared() -> None:
    """Each event has its own id and its own metadata dict."""
    first = SessionStart()
    second = SessionStart()

    assert first.id != second.id
    assert first.metadata is not second.metadata


def test_events_are_frozen() -> None:
    """Assigning to a field of an event raises a validation error."""
    event = UserMessage(parent_id=PARENT, content="hi")

    with pytest.raises(ValidationError):
        event.content = "changed"


def test_session_start_is_the_only_root() -> None:
    """SessionStart has no parent, and every other event requires one."""
    assert SessionStart().parent_id is None

    with pytest.raises(ValidationError):
        UserMessage(content="hi")  # missing parent_id
    with pytest.raises(ValidationError):
        AssistantMessage()  # missing parent_id
    with pytest.raises(ValidationError):
        ToolResultBatch(results=())  # missing parent_id


def test_assistant_message_defaults_to_no_content_and_no_tool_calls() -> None:
    """An assistant message may omit both its text and its tool calls."""
    message = AssistantMessage(parent_id=PARENT)

    assert message.content == ""
    assert message.tool_calls == ()


def test_tool_result_is_error_defaults_to_false() -> None:
    """A tool result is a success unless marked otherwise."""
    assert ToolResult(call_id="c1", content="ok").is_error is False


def test_required_fields_are_enforced() -> None:
    """Omitting a required field raises a validation error."""
    with pytest.raises(ValidationError):
        ToolCall(name="search", args={})  # missing call_id


def test_tool_calls_and_results_are_value_objects_not_events() -> None:
    """ToolCall and ToolResult are nested values, so they are not events."""
    call = ToolCall(call_id="c1", name="search", args={})
    result = ToolResult(call_id="c1", content="ok")

    assert not isinstance(call, Event)
    assert not isinstance(result, Event)
    assert not hasattr(call, "parent_id")
    assert not hasattr(result, "parent_id")


@pytest.mark.parametrize(
    ("event", "expected_type"),
    [
        (SessionStart(), "session_start"),
        (UserMessage(parent_id=PARENT, content="hi"), "user_message"),
        (AssistantMessage(parent_id=PARENT), "assistant_message"),
        (ToolResultBatch(parent_id=PARENT, results=()), "tool_result_batch"),
    ],
)
def test_type_literal_is_set_and_serialized(event: Event, expected_type: str) -> None:
    """Each event subclass carries its type literal into the serialized output."""
    assert event.type == expected_type
    assert event.model_dump()["type"] == expected_type


def test_parent_ids_link_a_turn_into_a_chain() -> None:
    """Each event points at the one that caused it, back to the session root."""
    session = SessionStart()
    user = UserMessage(parent_id=session.id, content="hi")
    assistant = AssistantMessage(parent_id=user.id)
    batch = ToolResultBatch(parent_id=assistant.id, results=())

    assert [e.parent_id for e in (session, user, assistant, batch)] == [
        None,
        session.id,
        user.id,
        assistant.id,
    ]


def test_results_pair_with_calls_via_call_id() -> None:
    """A batch holds one result per call in the assistant message, matched by id."""
    assistant = AssistantMessage(
        parent_id=PARENT,
        tool_calls=(
            ToolCall(call_id="c1", name="search", args={"q": "x"}),
            ToolCall(call_id="c2", name="read", args={"path": "a"}),
        ),
    )
    batch = ToolResultBatch(
        parent_id=assistant.id,
        results=(
            ToolResult(call_id="c1", content="found"),
            ToolResult(call_id="c2", content="oops", is_error=True),
        ),
    )

    assert [r.call_id for r in batch.results] == [
        c.call_id for c in assistant.tool_calls
    ]


def test_json_round_trip_preserves_nested_tool_calls() -> None:
    """Serializing to JSON and back restores the event and its nested tuple."""
    original = AssistantMessage(
        parent_id=uuid4(),
        content="Let me look that up.",
        tool_calls=(ToolCall(call_id="c1", name="search", args={"q": "x"}),),
        metadata={"tokens": 12},
    )

    restored = AssistantMessage.model_validate_json(original.model_dump_json())

    assert restored == original
    assert isinstance(restored.tool_calls, tuple)


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