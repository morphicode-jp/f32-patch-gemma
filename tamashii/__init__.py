"""Tamashii — intelligence-dedicated parallel-additive matryoshka.

Independent of twelve/ (optimization matryoshka).
Each shell runs concurrently, writes additive deltas to a shared state S.
"""
from tamashii.core import Tamashii
from tamashii.shell_base import Shell

__all__ = ["Tamashii", "Shell"]
