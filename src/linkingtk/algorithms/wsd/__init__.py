"""Word Sense Disambiguation (WSD) algorithms.

WSD identifies the relevant dictionary sense for a mention in context.
Similar to EL, but the target dataset is a dictionary rather than a KB.
Related systems: Lesk, EWISER, GlossBERT, ESC, SANDWiCH.
"""

from linkingtk.algorithms.wsd.ewiser import EwiserEncoder, EwiserLinker
from linkingtk.algorithms.wsd.glossbert import GlossBertEncoder, GlossBertLinker
from linkingtk.algorithms.wsd.lesk import LeskLinker
from linkingtk.algorithms.wsd.sandwich import (
    JsonNeighbours,
    SandwichEncoder,
    SandwichLinker,
    WordNetNeighbours,
)

__all__ = [
    "LeskLinker",
    "GlossBertEncoder",
    "GlossBertLinker",
    "EwiserEncoder",
    "EwiserLinker",
    "SandwichEncoder",
    "SandwichLinker",
    "WordNetNeighbours",
    "JsonNeighbours",
]
