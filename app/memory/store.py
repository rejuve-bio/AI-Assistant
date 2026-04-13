from typing import Dict

class MemoryStore:
    """Holds key-value facts for one agent session."""

    def __init__(self):
        self._store: Dict[str, str] = {}

    def write(self, key: str, value: str) -> str:
        """Store a fact. Returns a confirmation string."""
        self._store[key.strip()] = value.strip()
        return f"Stored '{key}': '{value}'"

    def read(self, key: str) -> str:
        """Retrieve a fact by key. Returns the value or a not-found message."""
        return self._store.get(key.strip(), f"No memory found for key: '{key}'")

    def read_all(self) -> str:
        """Return all stored facts as a formatted string."""
        if not self._store:
            return "No facts stored yet."
        return "\n".join(f"- {k}: {v}" for k, v in self._store.items())

    def clear(self):
        """Clear the memory store."""
        self._store.clear()
