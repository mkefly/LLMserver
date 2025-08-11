"""Chat memory store ABC."""
from __future__ import annotations
from typing import Sequence
from abc import ABC, abstractmethod

class MemoryStore(ABC):
    @abstractmethod
    async def load(self, session_id: str, model_version: str) -> Sequence[str]: ...
    @abstractmethod
    async def append(self, session_id: str, model_version: str, role: str, content: str) -> None: ...
    @abstractmethod
    async def clear(self, session_id: str, model_version: str) -> None: ...
