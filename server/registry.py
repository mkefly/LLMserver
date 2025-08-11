"""Simple plugin registries for adapters, memory stores, and resolvers."""
from __future__ import annotations
from typing import Callable, Dict, Any
from .capabilities import CapLLM
from .memory.base import MemoryStore
from .models.resolvers import ModelResolver

_ADAPTERS: Dict[str, Callable[..., CapLLM]] = {}
_MEMORY: Dict[str, Callable[..., MemoryStore]] = {}
_RESOLVERS: Dict[str, Callable[..., ModelResolver]] = {}

def register_adapter(name: str, factory: Callable[..., CapLLM]) -> None:
    _ADAPTERS[name] = factory

def register_memory(name: str, factory: Callable[..., MemoryStore]) -> None:
    _MEMORY[name] = factory

def register_resolver(name: str, factory: Callable[..., ModelResolver]) -> None:
    _RESOLVERS[name] = factory

def adapter(name: str) -> Callable[..., CapLLM]:
    if name not in _ADAPTERS:
        raise ValueError(f"Unknown adapter '{name}'. Installed: {list(_ADAPTERS)}")
    return _ADAPTERS[name]

def memory(name: str) -> Callable[..., MemoryStore]:
    if name not in _MEMORY:
        raise ValueError(f"Unknown memory '{name}'. Installed: {list(_MEMORY)}")
    return _MEMORY[name]

def resolver(name: str) -> Callable[..., ModelResolver]:
    if name not in _RESOLVERS:
        raise ValueError(f"Unknown resolver '{name}'. Installed: {list(_RESOLVERS)}")
    return _RESOLVERS[name]
