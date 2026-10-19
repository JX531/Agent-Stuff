"""Implementation of tool base class and registry for agent harness"""

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict


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
