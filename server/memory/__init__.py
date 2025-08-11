from __future__ import annotations
from ..registry import register_memory
from .inproc import InProcMemoryStore

# Register minimal memory backend; others (redis, uc_delta) can be added later
register_memory("inproc", lambda **_: InProcMemoryStore())
