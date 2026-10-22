"""Implementation of session event history management"""

from collections import deque
from uuid import UUID

from agent_harness.events import Event, SessionStart


class SessionHistory:
    """Events history of a single session"""

    def __init__(self):
        """
        Create a SessionHistory
        """
        self.session_root: SessionStart = SessionStart()
        self.session_history: dict[UUID, Event] = {self.session_root.id: self.session_root}
        self.children: dict[UUID, list[UUID]] = {self.session_root.id: []}
        self.head: Event = self.session_root

    def add_event(self, event: Event):
        """
        Add an event to the session and make it the new head.

        Registers the event, links it as a child of its parent, gives it an
        empty child list, and moves the head to it.

        Args:
            event: Event being added.

        Raises:
            ValueError: If parent event is not found, or another existing event with same id.
        """
        if event.parent_id not in self.session_history:
            raise ValueError(f"Missing parent event with ID: {event.parent_id}")

        if event.id in self.session_history:
            raise ValueError(f"Duplicate event already exists with ID: {event.id}")

        self.session_history[event.id] = event
        self.children[event.parent_id].append(event.id)
        self.children[event.id] = []
        self.head = event

    def ascend_to_root(self, event: Event | None = None) -> list[Event]:
        """
        Collects the path from root to a given event.

        Args:
            event: Event to start from. Defaults to the current head.

        Returns:
            The events from the root to event, oldest first, root included.

        Raises:
            ValueError: If the event is not in the session history.
        """
        cur = self.head if event is None else event
        if cur.id not in self.session_history:
            raise ValueError(f"No event found in session history with ID: {cur.id}")

        path: deque[Event] = deque()
        while True:
            path.appendleft(cur)

            if cur.parent_id is None:  # Only in SessionStart event type
                break

            cur = self.session_history[cur.parent_id]

        return list(path)

    def descend_to_leaf(self, event: Event, index: int = 0) -> None:
        """
        Descends to a leaf from a given event and sets the session head to it.

        Args:
            event: Event to descend to leaf from.
            index: Index of child event to descend to leaf by. Defaults to first child event.

        Raises:
            ValueError: If event is not found in session history or index out of range
        """

        if event.id not in self.session_history:
            raise ValueError(f"No event found in session history with ID: {event.id}")

        children = self.children[event.id]
        if children and not 0 <= index < len(children):
            raise ValueError(
                f"Index {index} out of range for Event {event.id} with {len(children)} children"
            )

        cur = self.session_history[children[index]] if children else event

        while self.children[cur.id]:
            cur = self.session_history[self.children[cur.id][0]]

        self.head = cur
