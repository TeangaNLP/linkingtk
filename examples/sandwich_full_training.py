"""Trains SANDWiCH's two cross-encoders from scratch on SemCor, with the
paper's own recipe, then evaluates them through `SandwichLinker`.

The reference repository releases no training code, so this follows the
paper (arXiv 2503.05958, Sec. 3.2/4.1, Table 7):

- Backbone ``microsoft/deberta-v3-small``; binary cross-entropy on the
  2-class logit margin; AdamW (weight decay 0.01), lr 2e-5, batch 64,
  10 epochs, cosine annealing, gradient clipping at 1.0, max 512 tokens.
- "We generate positive and negative examples by sampling from the
  neighborhoods of correct and incorrect senses ... all elements within a
  neighborhood share the same label": positives are each gold sense plus
  (up to ``--pos-neighbours``) of its sense-separable cluster members;
  negatives are each wrong candidate plus (up to ``--neg-neighbours``) of
  its own, capped at ``--max-negatives`` per instance. The paper gives no
  sampling ratios -- these are this script's own choices.
- Two encoders split by part of speech: ``encoder-v`` is trained on
  noun+verb instances, ``encoder-nv`` on every POS except verbs (the
  paper's ``E_v`` / ``E_v̄``; the released directory names match this
  reading). At inference both score every pair, as in the reference.
- SemCor (UFSAC's ``semcor.xml``, WordNet 3.0 -- the same sense ids as the
  Raganato evaluation data) for training; SemEval-2007 as dev. After every
  epoch the encoder is evaluated on SE07 *through the real cluster-level
  `SandwichLinker`* (alone), and the best epoch is kept.

Monosemous instances (a single candidate) are skipped: they contribute no
negative and only an uninformative positive.

Outputs ``<output-dir>/encoder-v`` and ``<output-dir>/encoder-nv`` (plain
HF directories, loadable with `SandwichLinker.from_checkpoint`), then
prints Raganato results for the pair.

Run with: `uv run python examples/sandwich_full_training.py`
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import torch
from sandwich_reproduction import _EVAL_SETS, _PosRestrictedMatch, _wordnet_pos_of_synset_id

from linkingtk.algorithms.wsd.sandwich import (
    SandwichEncoder,
    SandwichLinker,
    WordNetNeighbours,
    separable_clusters,
)
from linkingtk.core.entity import Entity
from linkingtk.datasets.ufsac import UfsacDataset
from linkingtk.eval import Evaluator
from linkingtk.sources.wn import WnEntitySource
from linkingtk.train.arguments import TrainingArguments
from linkingtk.train.cross_encoder import CrossEncoderTrainer

_UFSAC_DIR = Path.home() / "data" / "ufsac-public-2.1"
_OUTPUT_DIR = Path.home() / ".cache" / "linkingtk" / "sandwich_trained"
_POS_GROUPS = {"encoder-v": {"n", "v"}, "encoder-nv": {"n", "a", "r"}}

logger = logging.getLogger("sandwich_full_training")


def build_examples(
    mentions: list[Entity],
    senses: WnEntitySource,
    ground_truth: list[tuple[str, str]],
    neighbours: WordNetNeighbours,
    pos_neighbours: int,
    neg_neighbours: int,
    max_negatives: int,
    seed: int,
) -> dict[str, tuple[list[tuple[Entity, Entity]], list[tuple[Entity, Entity]]]]:
    """Per POS group: (positive pairs, negative pairs)."""
    rng = random.Random(seed)
    gold: dict[str, set[str]] = {}
    for mention_id, synset_id in ground_truth:
        gold.setdefault(mention_id, set()).add(synset_id)
    candidates: dict[str, list[str]] = {}
    by_id = {m.id: m for m in mentions}
    for mention, sense in _PosRestrictedMatch(top_k=50).candidate_pairs(mentions, senses):
        candidates.setdefault(mention.id, []).append(sense.id)

    sense_cache: dict[str, Entity] = {}

    def sense_entity(synset_id: str) -> Entity:
        if synset_id not in sense_cache:
            sense_cache[synset_id] = Entity(
                id=synset_id, labels=[], description=neighbours.definition(synset_id)
            )
        return sense_cache[synset_id]

    groups: dict[str, tuple[list[tuple[Entity, Entity]], list[tuple[Entity, Entity]]]] = {
        name: ([], []) for name in _POS_GROUPS
    }
    for mention_id, cids in candidates.items():
        golds = gold.get(mention_id, set()) & set(cids)
        if len(cids) < 2 or not golds:
            continue
        pos = _wordnet_pos_of_synset_id(cids[0])
        mention = by_id[mention_id]
        clusters = separable_clusters(cids, neighbours)
        positives: list[str] = []
        negatives: list[str] = []
        for cid, members in clusters.items():
            if cid in golds:
                positives += [cid, *rng.sample(members[1:], min(pos_neighbours, len(members) - 1))]
            else:
                negatives += [cid, *rng.sample(members[1:], min(neg_neighbours, len(members) - 1))]
        if len(negatives) > max_negatives:
            # Always keep the wrong candidates themselves; trim sampled neighbours first.
            wrong = [n for n in negatives if n in clusters]
            extra = [n for n in negatives if n not in clusters]
            negatives = wrong + rng.sample(extra, max(0, max_negatives - len(wrong)))
        for name, allowed in _POS_GROUPS.items():
            if pos in allowed:
                groups[name][0].extend((mention, sense_entity(s)) for s in positives)
                groups[name][1].extend((mention, sense_entity(s)) for s in negatives)
    return groups


def evaluate(
    linker: SandwichLinker, ufsac_dir: Path, labels: set[str] | None = None
) -> dict[str, float]:
    scores = {}
    for stem, label, _published in _EVAL_SETS:
        if labels is not None and label not in labels:
            continue
        mentions, senses, ground_truth = UfsacDataset(source=str(ufsac_dir / f"{stem}.xml")).load()
        results = linker.link(mentions, senses, blocking=_PosRestrictedMatch(top_k=50))
        report = Evaluator.evaluate(
            predictions=[(r.source_id, r.target_id) for r in results], ground_truth=ground_truth
        )
        scores[label] = report.metrics["precision@1"] * 100
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ufsac-dir", type=Path, default=_UFSAC_DIR)
    parser.add_argument("--output-dir", type=Path, default=_OUTPUT_DIR)
    parser.add_argument("--backbone", default="microsoft/deberta-v3-small")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pos-neighbours", type=int, default=3)
    parser.add_argument("--neg-neighbours", type=int, default=2)
    parser.add_argument("--max-negatives", type=int, default=12)
    parser.add_argument("--limit", type=int, help="Only the first N SemCor mentions (smoke test)")
    parser.add_argument("--encoders", nargs="+", default=list(_POS_GROUPS))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("linkingtk").setLevel(logging.INFO)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    neighbours = WordNetNeighbours("omw-en:1.4")
    mentions, senses, ground_truth = UfsacDataset(source=str(args.ufsac_dir / "semcor.xml")).load()
    if args.limit:
        mentions = mentions[: args.limit]
        keep = {m.id for m in mentions}
        ground_truth = [(m, s) for m, s in ground_truth if m in keep]
    assert isinstance(senses, WnEntitySource)
    groups = build_examples(
        mentions,
        senses,
        ground_truth,
        neighbours,
        args.pos_neighbours,
        args.neg_neighbours,
        args.max_negatives,
        args.seed,
    )

    for name in args.encoders:
        positives, negatives = groups[name]
        logger.info("%s: %d positive, %d negative pairs", name, len(positives), len(negatives))
        encoder = SandwichEncoder(args.backbone, forward_batch_size=args.batch_size)
        best_dir = args.output_dir / name
        best = {"score": -1.0, "epoch": -1}

        def on_epoch_end(
            epoch: int,
            encoder: SandwichEncoder = encoder,
            best_dir: Path = best_dir,
            best: dict[str, float] = best,
            name: str = name,
        ) -> None:
            dev = evaluate(
                SandwichLinker(encoders=[encoder], neighbours=neighbours), args.ufsac_dir, {"SE07"}
            )["SE07"]
            logger.info("%s epoch %d: SE07 %.1f (best %.1f)", name, epoch + 1, dev, best["score"])
            if dev > best["score"]:
                best.update(score=dev, epoch=epoch + 1)
                encoder.model.save_pretrained(best_dir)
                encoder.tokenizer.save_pretrained(best_dir)

        training_args = TrainingArguments(
            output_dir=str(args.output_dir / f"{name}-last"),
            learning_rate=args.lr,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
            weight_decay=0.01,
            lr_scheduler="cosine",
            max_grad_norm=1.0,
        )
        CrossEncoderTrainer(
            model=encoder,
            args=training_args,
            train_data=positives,
            negative_data=negatives,
            on_epoch_end=on_epoch_end,
        ).train()
        logger.info("%s: best SE07 %.1f at epoch %d", name, best["score"], best["epoch"])
        del encoder
        torch.cuda.empty_cache()

    linker = SandwichLinker(
        encoders=[
            SandwichEncoder.from_checkpoint(args.output_dir / name) for name in args.encoders
        ],
        neighbours=neighbours,
    ).to(device)
    print(f"{'dataset':<8} {'precision@1':>12} {'released':>10}")
    for label, score in evaluate(linker, args.ufsac_dir).items():
        published = next(p for _s, lab, p in _EVAL_SETS if lab == label)
        print(f"{label:<8} {score:>11.1f}% {published:>9.1f}%", flush=True)


if __name__ == "__main__":
    main()
