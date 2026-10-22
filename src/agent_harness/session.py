"""Implementation of session management"""

from agent_harness.accumulator import Accumulator
from agent_harness.client import VLLMClient
from agent_harness.events import AssistantMessage
from agent_harness.history import SessionHistory
from agent_harness.tool_registry import ToolRegistry


class Session:
    """A single session, made up of its event history and the agent loop that extends it"""

    def __init__(
        self,
        client: VLLMClient,
        tool_registry: ToolRegistry,
        history: SessionHistory = None,
        max_steps: int = 20,
    ):
        """Create a session.

        Args:
            client: Model client used for every request in this session.
            tool_registry: Tools the model may call.
            history: SessionHistory that may be imported.
            max_steps: Maximum model calls allowed in a single turn.
        """
        self.client = client
        self.tool_registry = tool_registry
        self.history = history if history else SessionHistory()
        self.max_steps = max_steps

    def call_model(self) -> AssistantMessage:  # TODO Yield deltas for streaming
        """Calls the model

        Sends the path of events from current conversation head to model,
        feeds the response deltas into an accumulator,
        adds the resulting AssistantMessage into history, and returns it.

        Returns:
            message: The AssistantMessage response from model.
        """
        accum = Accumulator()
        for delta in self.client.stream_chat(
            events=self.history.ascend_to_root(), tools=self.tool_registry.schemas()
        ):
            accum.feed(delta=delta)

        message = accum.build(parent_id=self.history.head.id)
        self.history.add_event(message)

        return message
