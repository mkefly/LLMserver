from __future__ import annotations
from ..registry import register_resolver
from .resolvers import UCStageResolver, StaticURIResolver

register_resolver("uc_stage",   lambda **_: UCStageResolver())
register_resolver("static_uri", lambda **_: StaticURIResolver())
