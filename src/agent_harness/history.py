"""Per-session event history stored as a tree, with support for branch switching."""

from collections import deque
from uuid import UUID

from agent_harness.events import Event, SessionStart


class SessionHistory:
    """The tree of events for a single session.

    Events are stored flat by ID and linked through ``parent_id``, so a fork is
    simply two events that share a parent. A separate children index allows
    walking down the tree, and a per-node record of the active child remembers
    which branch was last followed through each fork.

    Attributes:
        session_root: The ``SessionStart`` event at the root of the tree.
        session_history: Every event in the session, keyed by event ID.
        children: Child event IDs for every event, in creation order. Leaves map
            to an empty list.
        active_child: For each event that has children, the ID of the child on
            the currently selected branch.
        head: The tip of the current branch, which new events are normally
            attached to.
    """

    def __init__(self):
        """Create a history containing only the root ``SessionStart`` event.

        The root is registered in both ``session_history`` and ``children`` so
        that every stored event always has an entry in each, and the head starts
        at the root.
        """
        self.session_root: SessionStart = SessionStart()
        self.session_history: dict[UUID, Event] = {self.session_root.id: self.session_root}
        self.children: dict[UUID, list[UUID]] = {self.session_root.id: []}
        self.active_child: dict[UUID, UUID] = {}
        self.head: Event = self.session_root

    def add_event(self, event: Event):
        """Add an event to the session and make it the new head.

        Registers the event, appends it to its parent's children, marks it as the
        parent's active child, gives it an empty child list, and moves the head to
        it. All validation happens before any state is changed, so a failed call
        leaves the history untouched.

        Args:
            event: The event to add. Its ``parent_id`` must already be in the
                history, but need not be the current head, which is how new
                branches are created.

        Raises:
            ValueError: If the parent event is not in the history, or an event
                with the same ID already exists.
        """
        if event.parent_id not in self.session_history:
            raise ValueError(f"Missing parent event with ID: {event.parent_id}")

        if event.id in self.session_history:
            raise ValueError(f"Duplicate event already exists with ID: {event.id}")

        self.session_history[event.id] = event
        self.children[event.parent_id].append(event.id)
        self.active_child[event.parent_id] = event.id
        self.children[event.id] = []
        self.head = event

    def ascend_to_root(self, event: Event | None = None) -> list[Event]:
        """Collect the path from the root down to an event.

        Follows ``parent_id`` links upward from the event until it reaches the
        ``SessionStart``, which is the only event without a parent. The result is
        the linear history for that branch, in the order expected by
        ``events_to_messages``.

        Args:
            event: The event to end the path at. Defaults to the current head.

        Returns:
            The events from the root to ``event``, oldest first, with both the
            root and ``event`` included.

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

    def descend_to_leaf(self, event: Event, index: int) -> None:
        """Select a child of an event, descend to a leaf, and move the head there.

        The child at ``index`` is recorded as the event's active child. Below that
        child, the descent follows each node's remembered active child, so a branch
        that was visited before resumes where it left off. If ``event`` is already
        a leaf, ``index`` is ignored and the head moves to ``event``.

        Args:
            event: The event whose children are being chosen between.
            index: Position of the child to select within the event's children,
                in creation order.

        Raises:
            ValueError: If the event is not in the session history, or the event
                has children and ``index`` is out of range.
        """

        if event.id not in self.session_history:
            raise ValueError(f"No event found in session history with ID: {event.id}")

        children = self.children[event.id]
        if children and not 0 <= index < len(children):
            raise ValueError(
                f"Index {index} out of range for Event {event.id} with {len(children)} children"
            )

        cur = event
        if children:
            target_id = children[index]
            self.active_child[cur.id] = target_id
            cur = self.session_history[target_id]

        while cur.id in self.active_child:
            cur = self.session_history[self.active_child[cur.id]]

        self.head = cur
