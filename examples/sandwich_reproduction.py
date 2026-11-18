"""Reproduces SANDWiCH's released results using the authors' own checkpoints,
through this repo's `SandwichLinker`/`UfsacDataset` pipeline.

No training happens here (unless ``--encoders`` points at your own; see
``sandwich_full_training.py``) -- this loads Guzman-Olivares et al.'s
released cross-encoders (https://github.com/DanielGuzmanOlivares/sandwich,
NAACL 2025) via `SandwichLinker.from_checkpoint` and evaluates them via
`SandwichLinker.link()`, the same code path a freshly trained model uses.

Setup:

1. The released weights (the Google Drive link in the reference README's
   "Download the model weights" step, ``models.zip``), extracted so that
   `_CHECKPOINT_DIR` below contains ``encoder-nv/`` and ``encoder-v/``.
2. UFSAC 2.1 (https://github.com/getalp/UFSAC#content-of-the-repository)
   extracted to `_UFSAC_DIR` below (same convention as
   ``esc_reproduction.py``).

The sense graph is derived from WordNet (`WordNetNeighbours`), not the
reference's non-commercial BabelNet 5.3 extract -- see
``docs/examples/sandwich.md`` for how the two compare, and for the
reference-oracle numbers (the reference code run unchanged on its own
data). Pass ``--babelnet-dir`` (the reference repo's unzipped
``data/babelnet``) to score with its BabelNet definitions/neighbours
instead -- this needs ``--bn2wn`` too, a JSON map from WordNet synset ids
back to ``bn:`` ids, since `UfsacDataset` candidates are WordNet synsets.

Candidates are restricted to each mention's tagged part of speech, as in
the reference's own benchmark files (every instance's candidate list is
single-POS) -- the same convention as ``esc_reproduction.py``.

Published target (the reference README, released code): SE07=80.9,
SE2=87.8, SE3=85.7, SE13=92.6, SE15=91.2, ALL=88.6 (88.4 on the standard
Raganato ALL, which is what this script compares against). The paper's own Table
1 (unreleased code): SE07=81.2, SE2=88.5, SE3=84.9, SE13=92.5, SE15=91.7,
ALL=89.0.

Run with: `uv run python examples/sandwich_reproduction.py`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from linkingtk.algorithms.wsd.sandwich import (
    JsonNeighbours,
    NeighbourProvider,
    SandwichEncoder,
    SandwichLinker,
    WordNetNeighbours,
)
from linkingtk.blocking.base import BlockingStrategy
from linkingtk.blocking.exact import ExactMatch
from linkingtk.core.entity import Entity
from linkingtk.core.source import EntitySource
from linkingtk.datasets.ufsac import UfsacDataset
from linkingtk.eval import Evaluator

_CHECKPOINT_DIR = Path.home() / ".cache" / "linkingtk" / "sandwich_checkpoint"
_UFSAC_DIR = Path.home() / "data" / "ufsac-public-2.1"

# (UFSAC filename stem, label, released-checkpoint F1). Per-split values are
# the reference README's; ALL is the standard 7,253-instance Raganato ALL
# recomputed from the reference's own predictions -- the README's "88.6"
# mixes in SemEval-2010 and leaves out SE07 (see docs/examples/sandwich.md).
_EVAL_SETS = [
    ("raganato_semeval2007", "SE07", 80.9),
    ("raganato_senseval2", "SE2", 87.8),
    ("raganato_senseval3", "SE3", 85.7),
    ("raganato_semeval2013", "SE13", 92.6),
    ("raganato_semeval2015", "SE15", 91.2),
    ("raganato_ALL", "ALL", 88.4),
]

_PENN_TO_WORDNET_POS = {"N": "n", "V": "v", "J": "a", "R": "r"}


def _wordnet_pos_of_synset_id(synset_id: str) -> str | None:
    pos = synset_id.rsplit("-", 1)[-1]
    if pos == "s":  # adjective satellite counts as a plain adjective
        return "a"
    return pos if pos in "nvar" else None


def _mention_wordnet_pos(mention: Entity) -> str | None:
    tag = mention.properties.get("pos")
    return _PENN_TO_WORDNET_POS.get(tag[0]) if tag else None


class _PosRestrictedMatch(BlockingStrategy):
    """`ExactMatch`, keeping only candidates whose WordNet POS matches the mention's tag."""

    def __init__(self, top_k: int = 50) -> None:
        self._inner = ExactMatch(top_k=top_k)

    def candidate_pairs(
        self, dataset1: list[Entity], dataset2: list[Entity] | EntitySource
    ) -> list[tuple[Entity, Entity]]:
        return [
            (entity1, entity2)
            for entity1, entity2 in self._inner.candidate_pairs(dataset1, dataset2)
            if (pos := _mention_wordnet_pos(entity1)) is None
            or _wordnet_pos_of_synset_id(entity2.id) == pos
        ]


class _MappedJsonNeighbours:
    """`JsonNeighbours` addressed by WordNet ids, via a WordNet->BabelNet id map.

    Neighbours come back as ``bn:`` ids (they never need mapping back --
    only their definitions are looked up), so the provider accepts both.
    """

    def __init__(self, babelnet_dir: Path, wn_to_bn: dict[str, str]) -> None:
        self._inner = JsonNeighbours(
            babelnet_dir / "definitions.json", babelnet_dir / "neighbours.json"
        )
        self._wn_to_bn = wn_to_bn

    def _bn(self, synset_id: str) -> str:
        return self._wn_to_bn.get(synset_id, synset_id)

    def neighbours(self, synset_id: str) -> list[str]:
        return self._inner.neighbours(self._bn(synset_id))

    def definition(self, synset_id: str) -> str:
        return self._inner.definition(self._bn(synset_id))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint-dir", type=Path, default=_CHECKPOINT_DIR)
    parser.add_argument(
        "--encoders", type=Path, nargs="+", help="Own encoder dirs instead of the released pair"
    )
    parser.add_argument("--ufsac-dir", type=Path, default=_UFSAC_DIR)
    parser.add_argument("--babelnet-dir", type=Path)
    parser.add_argument("--bn2wn", type=Path, help="JSON {bn_id: wn_synset_id}")
    parser.add_argument(
        "--relations",
        nargs="*",
        help="WordNet relations defining neighbours (default: taxonomic); none = no clusters",
    )
    parser.add_argument("--splits", nargs="+", help="Subset of labels, e.g. SE07 ALL")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    neighbours: NeighbourProvider
    separable = True
    if args.babelnet_dir:
        if not args.bn2wn:
            parser.error("--babelnet-dir needs --bn2wn")
        bn_to_wn: dict[str, str] = json.loads(args.bn2wn.read_text())
        neighbours = _MappedJsonNeighbours(args.babelnet_dir, {w: b for b, w in bn_to_wn.items()})
        separable = False  # already sense-separable as released
    else:
        neighbours = (
            WordNetNeighbours("omw-en:1.4", relations=args.relations)
            if args.relations is not None
            else WordNetNeighbours("omw-en:1.4")
        )

    if args.encoders:
        encoders = [
            SandwichEncoder.from_checkpoint(path, forward_batch_size=args.batch_size)
            for path in args.encoders
        ]
        linker = SandwichLinker(encoders=encoders, neighbours=neighbours, separable=separable)
    else:
        linker = SandwichLinker.from_checkpoint(
            args.checkpoint_dir,
            neighbours=neighbours,
            separable=separable,
            forward_batch_size=args.batch_size,
        )
    linker.to("cuda" if torch.cuda.is_available() else "cpu")
    blocking = _PosRestrictedMatch(top_k=50)

    print(f"{'dataset':<8} {'precision@1':>12} {'published':>10}")
    for stem, label, published in _EVAL_SETS:
        if args.splits and label not in args.splits:
            continue
        mentions, senses, ground_truth = UfsacDataset(
            source=str(args.ufsac_dir / f"{stem}.xml")
        ).load()
        results = linker.link(mentions, senses, blocking=blocking)
        predictions = [(result.source_id, result.target_id) for result in results]
        report = Evaluator.evaluate(predictions=predictions, ground_truth=ground_truth)
        score = report.metrics["precision@1"] * 100
        print(f"{label:<8} {score:>11.1f}% {published:>9.1f}%", flush=True)


if __name__ == "__main__":
    main()
