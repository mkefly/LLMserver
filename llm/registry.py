from typing import Callable, Dict, Any

class Registry:
    def __init__(self) -> None:
        self._items: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str):
        def deco(fn: Callable[..., Any]):
            self._items[name.lower()] = fn
            return fn
        return deco

    def create(self, name: str, **kwargs):
        key = (name or "").lower()
        if key not in self._items:
            raise ValueError(f"Unknown: {name}. Options: {sorted(self._items)}")
        return self._items[key](**kwargs)

ADAPTERS = Registry()
