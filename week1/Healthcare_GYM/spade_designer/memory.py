import json
from pathlib import Path
from typing import Any


class EnvironmentMemory:
    """
    Stores previously generated Healthcare_GYM environments.

    This is a lightweight adaptation of SPADE's environment memory M.
    Since this assignment focuses on environment design rather than
    policy training, we store complexity and validation information
    instead of hint-based regret.
    """

    def __init__(self, path: str = "spade_designer/generated/memory.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: list[dict[str, Any]] = []

        if self.path.exists():
            self.load()

    def add(
        self,
        iteration: int,
        task: dict,
        complexity_score: float,
        valid: bool,
    ) -> None:
        """Add one generated environment to memory."""

        record = {
            "iteration": iteration,
            "task_id": task.get("id"),
            "difficulty": task.get("description", {}).get("difficulty"),
            "complexity_score": complexity_score,
            "valid": valid,
            "task": task,
        }

        self.records.append(record)
        self.save()

    def get_recent(self, n: int = 3) -> list[dict]:
        """Return the most recently generated environments."""
        return self.records[-n:]

    def save(self) -> None:
        """Persist memory to disk."""
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=2, ensure_ascii=False)

    def load(self) -> None:
        """Load existing memory from disk."""
        with self.path.open("r", encoding="utf-8") as f:
            self.records = json.load(f)

    def clear(self) -> None:
        """Clear all stored environments."""
        self.records = []

        if self.path.exists():
            self.path.unlink()