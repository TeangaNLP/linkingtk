"""Unit tests for [SandwichLinker][linkingtk.algorithms.wsd.sandwich.SandwichLinker].

Fast/offline-shaped: encoder tests use
``hf-internal-testing/tiny-random-BertForSequenceClassification`` (same
tiny-model convention as ``test_glossbert.py`` -- `SandwichEncoder` is
backbone-agnostic), linker-scoring tests use a deterministic stub encoder,
and ``wn`` is replaced by an in-memory fake. The released DeBERTa
checkpoints are only exercised in ``examples/sandwich_reproduction.py``.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from linkingtk.algorithms.wsd.sandwich import (  # noqa: E402
    JsonNeighbours,
    SandwichEncoder,
    SandwichLinker,
    WordNetNeighbours,
    _marked_context,
    cluster_score,
    separable_clusters,
)
from linkingtk.blocking.base import BlockingStrategy  # noqa: E402
from linkingtk.blocking.exact import ExactMatch  # noqa: E402
from linkingtk.core.entity import Entity  # noqa: E402
from linkingtk.train.arguments import TrainingArguments  # noqa: E402
from linkingtk.train.cross_encoder import CrossEncoderTrainer  # noqa: E402

_TINY_MODEL = "hf-internal-testing/tiny-random-BertForSequenceClassification"


class _DictNeighbours:
    def __init__(self, neighbours: dict[str, list[str]], definitions: dict[str, str]) -> None:
        self._neighbours = neighbours
        self._definitions = definitions

    def neighbours(self, synset_id: str) -> list[str]:
        return self._neighbours.get(synset_id, [])

    def definition(self, synset_id: str) -> str:
        return self._definitions[synset_id]


class _StubEncoder(torch.nn.Module):
    """Logits from a fixed per-definition relevance table; records every pair it sees."""

    def __init__(self, relevance: dict[str, float]) -> None:
        super().__init__()
        self.relevance = relevance
        self.seen: list[tuple[str, str]] = []

    def logits(self, text_pairs: list[tuple[str, str]]) -> torch.Tensor:
        self.seen.extend(text_pairs)
        p1 = torch.tensor([self.relevance[definition] for _context, definition in text_pairs])
        return torch.stack([torch.log1p(-p1), torch.log(p1)], dim=1)


class _FixedPairs(BlockingStrategy):
    def __init__(self, pairs: list[tuple[Entity, Entity]]) -> None:
        self.pairs = pairs

    def candidate_pairs(self, dataset1, dataset2):  # type: ignore[no-untyped-def]
        return self.pairs


class TestMarkedContext:
    def test_wraps_span_mid_sentence(self) -> None:
        mention = Entity(id="m", labels=["art"], context=("The art of war", 4, 7))
        assert _marked_context(mention) == "The [D] art [D] of war"

    def test_marks_last_token_too(self) -> None:
        # The reference omits the markers here (a bug); we always mark.
        mention = Entity(id="m", labels=["art"], context=("I love art", 7, 10))
        assert _marked_context(mention) == "I love [D] art [D]"

    def test_marks_the_instance_not_the_first_occurrence(self) -> None:
        mention = Entity(id="m", labels=["art"], context=("art for art sake", 8, 11))
        assert _marked_context(mention) == "art for [D] art [D] sake"

    def test_without_span_prefixes_label(self) -> None:
        mention = Entity(id="m", labels=["art"], context="a sentence")
        assert _marked_context(mention) == "[D] art [D] a sentence"


class TestSeparableClusters:
    def test_self_first_and_shared_or_candidate_neighbours_dropped(self) -> None:
        provider = _DictNeighbours({"a": ["x", "shared", "b"], "b": ["y", "shared"]}, {})
        clusters = separable_clusters(["a", "b"], provider)
        assert clusters == {"a": ["a", "x"], "b": ["b", "y"]}

    def test_not_separable_keeps_everything(self) -> None:
        provider = _DictNeighbours({"a": ["x", "shared", "b"], "b": ["shared"]}, {})
        clusters = separable_clusters(["a", "b"], provider, separable=False)
        assert clusters == {"a": ["a", "x", "shared", "b"], "b": ["b", "shared"]}


class TestClusterScore:
    def test_matches_hand_computation(self) -> None:
        probs = torch.tensor([[0.2, 0.8], [0.6, 0.4]])
        margins = torch.tensor([0.6, 0.2])
        weights = torch.exp(margins) / torch.exp(margins).sum()
        expected = float(0.8 * weights[0] + 0.4 * weights[1])
        assert cluster_score(probs) == pytest.approx(expected)


class TestLinker:
    def _setup(self) -> tuple[list[Entity], _FixedPairs, _DictNeighbours]:
        mention = Entity(id="m", labels=["bass"], context=("He caught a bass", 12, 16))
        other = Entity(id="m2", labels=["bass"], context=("Turn the bass up", 9, 13))
        fish = Entity(id="fish", labels=["bass"])
        music = Entity(id="music", labels=["bass"])
        pairs = [(mention, fish), (mention, music), (other, fish), (other, music)]
        provider = _DictNeighbours(
            {"fish": ["perch", "trout"], "music": ["tone"]},
            {
                "fish": "a fish",
                "music": "low sound",
                "perch": "a perch",
                "trout": "a trout",
                "tone": "a tone",
            },
        )
        return [mention, other], _FixedPairs(pairs), provider

    def test_cluster_evidence_decides(self) -> None:
        # On its own definition "music" wins (0.6 vs 0.5), but fish's cluster is stronger.
        mentions, blocking, provider = self._setup()
        encoder = _StubEncoder(
            {"a fish": 0.5, "low sound": 0.6, "a perch": 0.9, "a trout": 0.9, "a tone": 0.1}
        )
        linker = SandwichLinker(encoders=[encoder], neighbours=provider)  # type: ignore[list-item]
        results = {r.source_id: r.target_id for r in linker.link(mentions, [], blocking=blocking)}
        assert results == {"m": "fish", "m2": "fish"}

    def test_each_pair_scored_once_per_encoder_and_averaged(self) -> None:
        mentions, blocking, provider = self._setup()
        first = _StubEncoder(
            {"a fish": 0.9, "low sound": 0.1, "a perch": 0.5, "a trout": 0.5, "a tone": 0.5}
        )
        second = _StubEncoder(
            {"a fish": 0.1, "low sound": 0.9, "a perch": 0.5, "a trout": 0.5, "a tone": 0.5}
        )
        linker = SandwichLinker(encoders=[first, second], neighbours=provider)  # type: ignore[list-item]
        scores = linker.score_candidates(mentions, [], blocking=blocking)
        # 2 mentions x 5 cluster synsets, no duplicates.
        assert len(first.seen) == len(set(first.seen)) == 10
        assert len(second.seen) == 10
        averaged = {cid: score for cid, score in scores["m"]}
        expected_fish = cluster_score(torch.tensor([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]))
        assert averaged["fish"] == pytest.approx(expected_fish, abs=1e-5)


class _FakeSynset:
    def __init__(self, synset_id: str, definition: str, related: dict[str, list[str]]) -> None:
        self.id = synset_id
        self._definition = definition
        self._related = related
        self.wordnet: _FakeWordnet | None = None

    def definition(self) -> str:
        return self._definition

    def get_related(self, *relations: str) -> list[_FakeSynset]:
        assert self.wordnet is not None
        return [
            self.wordnet.synset(target)
            for relation in relations
            for target in self._related.get(relation, [])
        ]


class _FakeWordnet:
    synsets: dict[str, _FakeSynset] = {}

    def __init__(self, lexicon: str) -> None:
        self.lexicon = lexicon

    def synset(self, synset_id: str) -> _FakeSynset:
        synset = self.synsets[synset_id]
        synset.wordnet = self
        return synset


@pytest.fixture
def fake_wn(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeWordnet.synsets = {
        "s1": _FakeSynset(
            "s1",
            "a musical instrument",
            {"hypernym": ["s2"], "hyponym": ["s3", "s4", "s1"], "mero_part": ["s5"]},
        ),
        "s2": _FakeSynset("s2", "a device", {}),
        "s3": _FakeSynset("s3", "an electric bass", {}),
        "s4": _FakeSynset("s4", "a double bass", {}),
        "s5": _FakeSynset("s5", "a string", {}),
    }
    module = types.ModuleType("wn")
    module.Wordnet = _FakeWordnet  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "wn", module)


@pytest.mark.usefixtures("fake_wn")
class TestWordNetNeighbours:
    def test_default_taxonomic_relations_excluding_self(self) -> None:
        provider = WordNetNeighbours("fake:1")
        assert provider.neighbours("s1") == ["s2", "s3", "s4"]

    def test_custom_relations_and_cap(self) -> None:
        provider = WordNetNeighbours("fake:1", relations=("mero_part", "hyponym"), max_neighbours=2)
        assert provider.neighbours("s1") == ["s5", "s3"]

    def test_definitions_capitalized_by_default(self) -> None:
        assert WordNetNeighbours("fake:1").definition("s1") == "A musical instrument"
        raw = WordNetNeighbours("fake:1", capitalize_definitions=False)
        assert raw.definition("s1") == "a musical instrument"


class TestJsonNeighbours:
    def test_reference_format(self, tmp_path: Path) -> None:
        (tmp_path / "d.json").write_text(json.dumps({"bn:1n": "Def one"}))
        (tmp_path / "n.json").write_text(json.dumps({"bn:1n": ["bn:1n", "bn:2n"]}))
        provider = JsonNeighbours(tmp_path / "d.json", tmp_path / "n.json")
        assert provider.neighbours("bn:1n") == ["bn:2n"]
        assert provider.neighbours("bn:9n") == []
        assert provider.definition("bn:1n") == "Def one"
        assert provider.definition("bn:9n") == ""


class TestEncoder:
    def _data(self) -> tuple[list[Entity], list[Entity], list[tuple[str, str]]]:
        mentions = [
            Entity(id="m1", labels=["bass"], context=("He caught a bass while fishing", 12, 16)),
            Entity(id="m2", labels=["bass"], context=("The bass was too quiet", 4, 8)),
        ]
        senses = [
            Entity(id="fish", labels=["bass"], description="a fish"),
            Entity(id="music", labels=["bass"], description="low-frequency music"),
        ]
        return mentions, senses, [("m1", "fish"), ("m2", "music")]

    def test_loads_fp32_even_from_fp16_checkpoint(self, tmp_path: Path) -> None:
        source = SandwichEncoder(_TINY_MODEL, max_length=32)
        source.model.half().save_pretrained(tmp_path / "fp16")
        source.tokenizer.save_pretrained(tmp_path / "fp16")
        reloaded = SandwichEncoder.from_checkpoint(tmp_path / "fp16")
        assert {p.dtype for p in reloaded.parameters()} == {torch.float32}

    def test_score_shapes_and_from_checkpoint_roundtrip(self, tmp_path: Path) -> None:
        mentions, senses, _ = self._data()
        encoder = SandwichEncoder(_TINY_MODEL, max_length=32, forward_batch_size=1)
        pairs = [(m, s) for m in mentions for s in senses]
        assert encoder.score(pairs).shape == (4,)
        assert encoder.logits([("a", "b")] * 3).shape == (3, 2)
        encoder.model.save_pretrained(tmp_path / "encoder-v")
        encoder.tokenizer.save_pretrained(tmp_path / "encoder-v")
        reloaded = SandwichEncoder.from_checkpoint(tmp_path / "encoder-v", max_length=32)
        encoder.eval()
        reloaded.eval()
        with torch.no_grad():
            assert torch.allclose(encoder.score(pairs), reloaded.score(pairs), atol=1e-5)

    def test_trains_with_explicit_negatives_and_links(self, tmp_path: Path) -> None:
        mentions, senses, ground_truth = self._data()
        by_id = {s.id: s for s in senses}
        positives = [(m, by_id[g]) for m, (_mid, g) in zip(mentions, ground_truth, strict=True)]
        negatives = [
            (m, s)
            for m, (_mid, g) in zip(mentions, ground_truth, strict=True)
            for s in senses
            if s.id != g
        ]
        encoder = SandwichEncoder(_TINY_MODEL, max_length=32)
        epochs: list[int] = []
        args = TrainingArguments(
            output_dir=str(tmp_path / "out"),
            learning_rate=1e-3,
            num_epochs=3,
            batch_size=4,
            lr_scheduler="cosine",
            max_grad_norm=1.0,
        )
        CrossEncoderTrainer(
            model=encoder,
            args=args,
            train_data=positives,
            negative_data=negatives,
            on_epoch_end=epochs.append,
        ).train()
        assert epochs == [0, 1, 2]

        provider = _DictNeighbours({}, {"fish": "a fish", "music": "low-frequency music"})
        linker = SandwichLinker(encoders=[encoder], neighbours=provider)
        results = linker.link(mentions, senses, blocking=ExactMatch())
        assert {r.source_id for r in results} == {"m1", "m2"}
