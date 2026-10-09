"""Accumulator responsible assembling Deltas into completed Events"""

from typing import Any


class Accumulator:
    """Assembles streaming deltas into one complete ``AssistantMessage``.

    Feed deltas one at a time with ``feed``, then call ``build`` once the stream
    has ended. Tool calls are tracked by stream ``index`` because several can
    stream interleaved. Their argument fragments are kept as raw text and parsed
    as JSON only in ``build``, since a partial fragment is not valid JSON.

    Attributes:
        texts: Text fragments received so far, in arrival order.
        tool_calls: In-progress tool calls keyed by stream index. Each value holds
            ``call_id``, ``name`` and ``args`` (a list of raw JSON fragments).
    """

    def __init__(self):
        """Init"""
        self.texts: list[str] = []
        self.tool_calls: dict[int, Any] = {}
