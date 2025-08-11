from __future__ import annotations
from typing import Dict, List, Sequence, Tuple
from .base import MemoryStore

class InProcMemoryStore(MemoryStore):
    """In-memory chat history keyed by (session_id, model_version)."""
    def __init__(self):
        self._m: Dict[Tuple[str,str], List[str]] = {}

    async def load(self, session_id: str, model_version: str) -> Sequence[str]:
        return list(self._m.get((session_id, model_version), []))

    async def append(self, session_id: str, model_version: str, role: str, content: str) -> None:
        key = (session_id, model_version)
        self._m.setdefault(key, []).append(f"{role}:{content}")

    async def clear(self, session_id: str, model_version: str) -> None:
        self._m.pop((session_id, model_version), None)
