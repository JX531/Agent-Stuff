"""Tool base class and tool registry for the agent harness."""

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict

from agent_harness.events import ToolCall, ToolResult


class Tool(BaseModel):
    """A tool the model can invoke.

    Attributes:
        name: Tool name. Must match what the model sends in ``ToolCall.name``. Must be unique.
        description: Description of tool shown to the model.
        parameters: JSON Schema describing the arguments.
        func: The Python callable that runs the tool. Receives the parsed args as
            keyword arguments and returns text for the model.
    """

    model_config = ConfigDict(frozen=True)
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., str]

    def to_schema(self) -> dict[str, Any]:
        """Render this tool in OpenAI ``tools`` format.

        Returns:
            A dict suitable for the ``tools`` field of a chat completion request.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def tool(name: str, description: str, parameters: dict[str, Any]):
    """Decorator that turns a function into a ``Tool``.

    Args:
        name: Tool name, as the model will call it.
        description: Description of the tool shown to the model.
        parameters: JSON Schema describing the arguments.

    Returns:
        A decorator that replaces the decorated function with a ``Tool``.
    """

    def decorator(func: Callable):
        return Tool(name=name, description=description, parameters=parameters, func=func)

    return decorator


class ToolRegistry:
    """Registry that holds tools by name and executes tool calls.

    Attributes:
        tools: Registered tools, keyed by tool name.
    """

    def __init__(self, tools: list[Tool]):
        """Create a registry.

        Args:
            tools: Tools to register. Names must be unique.

        Raises:
            ValueError: If two tools share a name.
        """
        self.tools: dict[str, Tool] = {}
        for tool in tools:
            if tool.name in self.tools:
                raise ValueError(f"Duplicate tool name: {tool.name}")

            else:
                self.tools[tool.name] = tool

    def schemas(self) -> list[dict[str, Any]]:
        """List the schemas of every registered tool.

        Returns:
            One OpenAI-format schema dict per registered tool.
        """
        return [tool.to_schema() for tool in self.tools.values()]

    def run(self, call: ToolCall) -> ToolResult:
        """Execute a tool call, flagging the result as an error on failure.

        Unknown tool names and exceptions raised by the tool are returned as error
        results rather than raised.

        Args:
            call: A ``ToolCall`` from the model.

        Returns:
            A ``ToolResult`` with the same ``call_id`` as ``call``.
        """

        tool = self.tools.get(call.name)
        if tool is None:
            return ToolResult(
                call_id=call.call_id, is_error=True, content=f"Unknown tool invoked: {call.name}"
            )

        try:
            return ToolResult(call_id=call.call_id, content=tool.func(**call.args))

        except Exception as e:
            return ToolResult(
                call_id=call.call_id, is_error=True, content=f"{type(e).__name__}: {e}"
            )
