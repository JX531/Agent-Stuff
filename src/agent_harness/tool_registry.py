"""Implementation of tool base class and registry for agent harness"""

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
    """
    Decorator used to produce tools from functions
    """

    def decorator(func: Callable):
        return Tool(name=name, description=description, parameters=parameters, func=func)

    return decorator


class ToolRegistry:
    """Registry that holds a dict of tools and supports tool call execution"""

    def __init__(self, tools: list[Tool]):
        """
        Create a ToolRegistry

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
        """
        Return a list of schemas for each tool in registry

        Returns:
            One OpenAI-format schema dict per registered tool.
        """
        return [tool.to_schema() for tool in self.tools.values()]

    def run(self, call: ToolCall) -> ToolResult:
        """
        Execute a ToolCall, returning the ToolResult, flagging as an error on failure.

        Args:
            call: A ToolCall from the model

        Returns:
            A ToolResult with the same call_id as the input ToolCall.
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
