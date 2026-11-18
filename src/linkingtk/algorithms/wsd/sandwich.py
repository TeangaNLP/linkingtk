"""SANDWiCH Word Sense Disambiguation.

SANDWiCH (Guzman-Olivares, Quijano-Sanchez & Liberatore, "SANDWiCH:
Semantical Analysis of Neighbours for Disambiguating Words in Context ad
Hoc", NAACL 2025, https://arxiv.org/abs/2503.05958,
https://github.com/DanielGuzmanOlivares/sandwich) scores each candidate
sense not on its own definition alone, but on its **neighbour cluster**:
the sense itself plus its immediate neighbours in a sense graph made
*sense-separable* (no neighbour shared between two candidate senses of
the same word). Ported from the reference implementation's own inference
code (`sandwich/__init__.py`, `Sandwich.disambiguate`), not just the paper:

- Every ``(context, definition)`` pair -- for every synset in every
  candidate's cluster -- is scored by **two** DeBERTa-v3-small
  cross-encoders (the released ``encoder-nv`` / ``encoder-v``,
  ``DebertaV2ForSequenceClassification`` with ``num_labels=2``); their
  2-class softmax outputs are averaged.
- The target word is marked in the context with the literal text
  ``[D] word [D]`` (plain text -- ``[D]`` is *not* an added token in the
  released tokenizer).
- The definition side is the bare definition (no lemma prefix, unlike
  [GlossBertLinker][linkingtk.algorithms.wsd.glossbert.GlossBertLinker]).
- Each candidate's score is ``sum(p1 * softmax(|p1 - p0|))`` over its
  cluster, where ``(p0, p1)`` is a cluster member's averaged probability
  pair and the softmax runs across the cluster -- a confidence-weighted
  sum, so larger clusters of confidently-matching neighbours win.

The reference ships its graph as a small non-commercially-licensed
BabelNet 5.3 extract (``definitions.json``/``neighbours.json``, keyed by
``bn:`` ids, covering only the benchmark candidates). This module does not
ship it: [WordNetNeighbours][linkingtk.algorithms.wsd.sandwich.WordNetNeighbours]
derives clusters from WordNet relations instead (works for any lemma),
and [JsonNeighbours][linkingtk.algorithms.wsd.sandwich.JsonNeighbours]
loads the reference's own JSON format for anyone who has it.

Two deliberate deviations from the reference's ``__add_target_tokens``,
both bugs there rather than design choices: it marks the *first*
occurrence of the target's surface form in the sentence (not the
instance's actual position), and it silently omits the markers entirely
when that occurrence is the sentence's last token. `_marked_context` marks
the mention's real character span, always.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import torch
from torch import nn

from linkingtk.algorithms.base import DEFAULT_BLOCKING, BaseLinker
from linkingtk.blocking.base import BlockingStrategy
from linkingtk.core.entity import Entity, description_text, label_texts
from linkingtk.core.result import AlignmentResult
from linkingtk.core.source import EntitySource
from linkingtk.matchers import DEFAULT_MATCHER, Matcher
from linkingtk.utils.graph import Graph

TARGET_MARKER = "[D]"

#: WordNet relations whose targets join a synset's cluster by default
#: (synset-level relations; see `WordNetNeighbours`). Taxonomic links
#: dominate the reference's own BabelNet clusters -- of its edges that map
#: back to a WordNet one-hop relation, the large majority are hyponym/
#: hypernym (incl. instance) links.
DEFAULT_WORDNET_RELATIONS: tuple[str, ...] = (
    "hypernym",
    "hyponym",
    "instance_hypernym",
    "instance_hyponym",
)


def _marked_context(entity: Entity) -> str:
    """The mention's context sentence with its span wrapped as ``[D] span [D]``."""
    context = entity.context
    if isinstance(context, tuple):
        text, start, end = context
        before = text[:start]
        after = text[end:]
        if before and not before.endswith(" "):
            before += " "
        if after and not after.startswith(" "):
            after = " " + after
        return f"{before}{TARGET_MARKER} {text[start:end]} {TARGET_MARKER}{after}"
    label = " ".join(label_texts(entity))
    context_str = context if isinstance(context, str) else ""
    return f"{TARGET_MARKER} {label} {TARGET_MARKER} {context_str}".strip()


def _capitalize(text: str) -> str:
    return text[:1].upper() + text[1:]


class NeighbourProvider(Protocol):
    """Sense graph + definitions that a
    [SandwichLinker][linkingtk.algorithms.wsd.sandwich.SandwichLinker] scores clusters over."""

    def neighbours(self, synset_id: str) -> list[str]:
        """The synset's raw neighbour ids (excluding itself), before separability filtering."""
        ...

    def definition(self, synset_id: str) -> str:
        """The synset's definition text, as fed to the cross-encoders."""
        ...


class WordNetNeighbours:
    """Clusters derived from WordNet relations via the ``wn`` package.

    Args:
        lexicon: ``wn`` lexicon specifier the candidate synset ids belong
            to (e.g. ``"omw-en:1.4"`` for
            [UfsacDataset][linkingtk.datasets.ufsac.UfsacDataset]'s WN 3.0
            ids, ``"oewn:2025+"`` for
            [SemCorDataset][linkingtk.datasets.semcor.SemCorDataset]'s).
        relations: Synset-level relation names whose targets become
            neighbours. Defaults to `DEFAULT_WORDNET_RELATIONS`.
        max_neighbours: Optional cap on neighbours per synset (in ``wn``'s
            relation order), bounding the cost of hub synsets with
            hundreds of hyponyms.
        capitalize_definitions: Upper-case each definition's first letter,
            matching the reference's BabelNet definitions (WordNet glosses,
            capitalized) that the released cased DeBERTa checkpoints saw.
    """

    def __init__(
        self,
        lexicon: str = "omw-en:1.4",
        relations: Sequence[str] = DEFAULT_WORDNET_RELATIONS,
        max_neighbours: int | None = None,
        capitalize_definitions: bool = True,
    ) -> None:
        import wn

        self._wordnet = wn.Wordnet(lexicon)
        self.relations = tuple(relations)
        self.max_neighbours = max_neighbours
        self.capitalize_definitions = capitalize_definitions
        self._neighbours = lru_cache(maxsize=None)(self._compute_neighbours)
        self._definition = lru_cache(maxsize=None)(self._compute_definition)

    def neighbours(self, synset_id: str) -> list[str]:
        return list(self._neighbours(synset_id))

    def definition(self, synset_id: str) -> str:
        return self._definition(synset_id)

    def _compute_neighbours(self, synset_id: str) -> tuple[str, ...]:
        synset = self._wordnet.synset(synset_id)
        seen: dict[str, None] = {}
        for related in synset.get_related(*self.relations):
            if related.id != synset_id:
                seen.setdefault(related.id, None)
        ids = tuple(seen)
        return ids[: self.max_neighbours] if self.max_neighbours is not None else ids

    def _compute_definition(self, synset_id: str) -> str:
        definition = self._wordnet.synset(synset_id).definition() or ""
        return _capitalize(definition) if self.capitalize_definitions else definition


class JsonNeighbours:
    """The reference implementation's own ``definitions.json``/``neighbours.json`` format.

    Both are JSON objects keyed by synset id: definitions map to a string,
    neighbours to a list of ids (the reference's lists include the synset
    itself, which is dropped here -- the linker always adds it back first).
    The reference's own files are a BabelNet 5.3 extract under BabelNet's
    non-commercial license and are not bundled with this package.
    """

    def __init__(self, definitions_path: str | Path, neighbours_path: str | Path) -> None:
        with open(definitions_path) as handle:
            self._definitions: dict[str, str] = json.load(handle)
        with open(neighbours_path) as handle:
            self._neighbours: dict[str, list[str]] = json.load(handle)

    def neighbours(self, synset_id: str) -> list[str]:
        return [n for n in self._neighbours.get(synset_id, []) if n != synset_id]

    def definition(self, synset_id: str) -> str:
        return self._definitions.get(synset_id, "")


def separable_clusters(
    candidate_ids: Sequence[str], provider: NeighbourProvider, separable: bool = True
) -> dict[str, list[str]]:
    """Each candidate's cluster: itself first, then its neighbours.

    With ``separable=True`` (the paper's "sense-separable" graph), a
    neighbour is dropped if it is itself one of the candidates, or if it
    would appear in more than one candidate's cluster -- so no evidence is
    shared between two competing senses of the same word.
    """
    raw = {cid: provider.neighbours(cid) for cid in candidate_ids}
    if not separable:
        return {cid: [cid, *(n for n in raw[cid] if n != cid)] for cid in candidate_ids}
    owners: dict[str, int] = defaultdict(int)
    for cid in candidate_ids:
        for n in set(raw[cid]):
            owners[n] += 1
    candidates = set(candidate_ids)
    return {
        cid: [cid, *(n for n in dict.fromkeys(raw[cid]) if n not in candidates and owners[n] == 1)]
        for cid in candidate_ids
    }


def cluster_score(probs: torch.Tensor) -> float:
    """The reference's aggregation, ``sum(p1 * softmax(|p1 - p0|))``.

    Args:
        probs: A cluster's ``(n, 2)`` encoder-averaged softmax probabilities.
    """
    p0, p1 = probs[:, 0], probs[:, 1]
    weights = torch.softmax((p1 - p0).abs(), dim=0)
    return float((p1 * weights).sum())


class SandwichEncoder(nn.Module):
    """One SANDWiCH cross-encoder: a 2-label HF sequence classifier (DeBERTa-v3 in the paper).

    Satisfies [CrossEncoderModel][linkingtk.train.cross_encoder.CrossEncoderModel],
    so it trains via [CrossEncoderTrainer][linkingtk.train.cross_encoder.CrossEncoderTrainer].

    Args:
        model_name_or_path: A Hugging Face model id or local directory
            (e.g. a released ``encoder-nv``/``encoder-v`` checkpoint, which
            is a plain HF directory). Defaults to the paper's
            ``microsoft/deberta-v3-small``.
        max_length: Maximum token length for the ``(context, definition)`` pair.
        forward_batch_size: Chunk size per forward pass (see
            [GlossBertEncoder][linkingtk.algorithms.wsd.glossbert.GlossBertEncoder]).
    """

    def __init__(
        self,
        model_name_or_path: str = "microsoft/deberta-v3-small",
        max_length: int = 512,
        forward_batch_size: int = 64,
    ) -> None:
        super().__init__()
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        # Always fp32: transformers 5 loads a checkpoint in its *stored* dtype by
        # default, and ``microsoft/deberta-v3-small`` is stored in fp16 -- AdamW
        # steps on fp16 weights turn the loss NaN after the first update.
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, num_labels=2
        ).float()
        self.max_length = max_length
        self.forward_batch_size = forward_batch_size

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: str | Path, max_length: int = 512, forward_batch_size: int = 64
    ) -> SandwichEncoder:
        """Load one released encoder directory (``encoder-nv`` or ``encoder-v``)."""
        return cls(
            str(checkpoint_path), max_length=max_length, forward_batch_size=forward_batch_size
        )

    def logits(self, text_pairs: Sequence[tuple[str, str]]) -> torch.Tensor:
        """``(len(text_pairs), 2)`` logits for raw ``(context, definition)`` text pairs."""
        chunks = [
            self._forward(text_pairs[start : start + self.forward_batch_size])
            for start in range(0, len(text_pairs), self.forward_batch_size)
        ]
        return torch.cat(chunks) if chunks else torch.empty(0, 2)

    def _forward(self, text_pairs: Sequence[tuple[str, str]]) -> torch.Tensor:
        device = next(self.model.parameters()).device
        encoded = self.tokenizer(
            [context for context, _ in text_pairs],
            [definition for _, definition in text_pairs],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(device)
        logits: torch.Tensor = self.model(**encoded).logits
        return logits

    def score(self, pairs: list[tuple[Entity, Entity]]) -> torch.Tensor:
        """``(len(pairs),)`` class-1-minus-class-0 logit margins for ``(mention, sense)`` pairs."""
        logits = self.logits(
            [(_marked_context(mention), description_text(sense)) for mention, sense in pairs]
        )
        margin: torch.Tensor = logits[:, 1] - logits[:, 0]
        return margin


class SandwichLinker(BaseLinker):
    """WSD by cluster-level cross-encoder scoring (SANDWiCH).

    Load the released checkpoints with
    [from_checkpoint][linkingtk.algorithms.wsd.sandwich.SandwichLinker.from_checkpoint],
    or train fresh `SandwichEncoder`s via `CrossEncoderTrainer` and pass them in:

    ```python
    linker = SandwichLinker.from_checkpoint("~/.cache/linkingtk/sandwich_checkpoint")
    results = linker.link(mentions, WnEntitySource("omw-en:1.4"))
    ```

    Args:
        encoders: Cross-encoders whose softmax outputs are averaged.
            Defaults to two fresh `SandwichEncoder`s (the paper's pair).
        neighbours: Sense graph/definitions. Defaults to
            ``WordNetNeighbours()`` (WN 3.0 ids, matching `UfsacDataset`).
        separable: Apply sense-separability filtering (see
            `separable_clusters`). Set ``False`` with `JsonNeighbours` to
            score the reference's clusters exactly as given.
        matching: Strategy resolving scored candidates into links.
    """

    def __init__(
        self,
        encoders: Sequence[SandwichEncoder] | None = None,
        neighbours: NeighbourProvider | None = None,
        separable: bool = True,
        matching: Matcher = DEFAULT_MATCHER,
    ) -> None:
        self.encoders = (
            list(encoders) if encoders is not None else [SandwichEncoder(), SandwichEncoder()]
        )
        self.neighbours = neighbours if neighbours is not None else WordNetNeighbours()
        self.separable = separable
        self.matching = matching

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        neighbours: NeighbourProvider | None = None,
        separable: bool = True,
        matching: Matcher = DEFAULT_MATCHER,
        forward_batch_size: int = 64,
    ) -> SandwichLinker:
        """Load the released ``models/`` directory (``encoder-nv`` + ``encoder-v`` subdirs)."""
        root = Path(checkpoint_dir).expanduser()
        encoders = [
            SandwichEncoder.from_checkpoint(root / name, forward_batch_size=forward_batch_size)
            for name in ("encoder-nv", "encoder-v")
        ]
        return cls(encoders=encoders, neighbours=neighbours, separable=separable, matching=matching)

    def to(self, device: str | torch.device) -> SandwichLinker:
        for encoder in self.encoders:
            encoder.to(device)
        return self

    def score_candidates(
        self,
        dataset1: list[Entity],
        dataset2: list[Entity] | EntitySource,
        blocking: BlockingStrategy = DEFAULT_BLOCKING,
    ) -> dict[str, list[tuple[str, float]]]:
        """Blocked candidates per mention, scored at cluster level but not yet matched.

        Satisfies [CandidateScorer][linkingtk.algorithms.llm_reranker.CandidateScorer].
        """
        mentions: dict[str, Entity] = {}
        candidates: dict[str, list[str]] = defaultdict(list)
        for mention, sense in blocking.candidate_pairs(dataset1, dataset2):
            mentions[mention.id] = mention
            if sense.id not in candidates[mention.id]:
                candidates[mention.id].append(sense.id)
        if not candidates:
            return {}

        clusters = {
            mid: separable_clusters(cids, self.neighbours, self.separable)
            for mid, cids in candidates.items()
        }
        # Each distinct (mention, synset) pair is scored once, however many clusters it is in.
        pair_index: dict[tuple[str, str], int] = {}
        for mid, by_candidate in clusters.items():
            for members in by_candidate.values():
                for synset_id in members:
                    pair_index.setdefault((mid, synset_id), len(pair_index))
        probs = self._probs(
            [
                (_marked_context(mentions[mid]), self.neighbours.definition(synset_id))
                for mid, synset_id in pair_index
            ]
        )

        scored: dict[str, list[tuple[str, float]]] = {}
        for mid, by_candidate in clusters.items():
            scored[mid] = [
                (cid, cluster_score(probs[[pair_index[(mid, s)] for s in members]]))
                for cid, members in by_candidate.items()
            ]
        return scored

    def _probs(self, text_pairs: list[tuple[str, str]]) -> torch.Tensor:
        """Encoder-averaged ``(n, 2)`` softmax probabilities, forwarded in length-sorted order."""
        order = sorted(
            range(len(text_pairs)), key=lambda i: len(text_pairs[i][0]) + len(text_pairs[i][1])
        )
        sorted_pairs = [text_pairs[i] for i in order]
        total = torch.zeros(len(text_pairs), 2)
        with torch.no_grad():
            for encoder in self.encoders:
                was_training = encoder.training
                encoder.eval()
                total += torch.softmax(encoder.logits(sorted_pairs).float(), dim=1).cpu()
                encoder.train(was_training)
        averaged = total / len(self.encoders)
        result = torch.empty_like(averaged)
        result[order] = averaged
        return result

    def link(
        self,
        dataset1: list[Entity],
        dataset2: list[Entity] | EntitySource,
        graph: Graph = None,
        blocking: BlockingStrategy = DEFAULT_BLOCKING,
    ) -> list[AlignmentResult]:
        return self.matching.match(self.score_candidates(dataset1, dataset2, blocking))
