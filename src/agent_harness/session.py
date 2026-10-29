"""Implementation of session management"""

from agent_harness.accumulator import Accumulator
from agent_harness.client import VLLMClient
from agent_harness.events import AssistantMessage, ToolResultBatch, UserMessage
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
        """Call the model on the current branch and record its reply.

        Sends the path from the root to the current head to the model, feeds the
        streamed deltas into a fresh ``Accumulator``, appends the resulting
        ``AssistantMessage`` to the history (making it the new head), and returns it.
        Nothing is added to the history if the call fails.

        Returns:
            The ``AssistantMessage`` produced by the model.

        Raises:
            RuntimeError: If the server responds with an HTTP error status.
            json.JSONDecodeError: If a tool call's arguments are not valid JSON.
        """

        accum = Accumulator()
        for delta in self.client.stream_chat(
            events=self.history.ascend_to_root(), tools=self.tool_registry.schemas()
        ):
            accum.feed(delta=delta)

        message = accum.build(parent_id=self.history.head.id)
        self.history.add_event(message)

        return message

    def run_turn(self, user_message: str) -> AssistantMessage:
        """Run one user turn: send a message, then loop until the model stops calling tools.

        Appends a ``UserMessage`` to the current head, then alternates between
        calling the model and running the tools it requests. Each pass adds an
        ``AssistantMessage`` and, if that message made tool calls, a
        ``ToolResultBatch``. The turn ends when the model replies without tool calls.

        Args:
            user_text: The text the user typed.

        Returns:
            The final ``AssistantMessage``, which has no tool calls.

        Raises:
            RuntimeError: If max steps exceeded or the server responds with an HTTP error status.
            JSONDecodeError: If a tool call's arguments are not valid JSON.
        """
        self.history.add_event(UserMessage(parent_id=self.history.head.id, content=user_message))

        for _ in range(self.max_steps):
            response = self.call_model()

            if not response.tool_calls:
                return response

            results = tuple(self.registry.run(call) for call in response.tool_calls)
            self.history.add(ToolResultBatch(parent_id=response.id, results=results))

        raise RuntimeError(f"Exceed max steps: {self.max_steps}")
