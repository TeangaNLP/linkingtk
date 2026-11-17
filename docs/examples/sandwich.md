# SANDWiCH

[`SandwichLinker`](../reference/algorithms.md#sandwich) ports
[Guzman-Olivares, Quijano-Sanchez & Liberatore](https://aclanthology.org/2025.naacl-long.358/)'s
SANDWiCH (NAACL 2025). SANDWiCH scores each candidate sense by its
**neighbour cluster** rather than by its definition alone. A cluster is
the sense plus its immediate neighbours in a *sense-separable* graph, one
where no neighbour is shared between two candidate senses of the same word.

- **[How it works](#how-it-works)**
- **[Reproducing the released results](#reproducing-the-released-results)**
- **[What the reference's numbers mean](#what-the-references-numbers-mean)**
- **[Training from scratch](#training-from-scratch)**

## How it works

This is ported from the reference implementation's own inference code
(`Sandwich.disambiguate` in
[DanielGuzmanOlivares/sandwich](https://github.com/DanielGuzmanOlivares/sandwich)),
not just from the paper:

1. Each candidate is expanded to its cluster. Any neighbour that is itself
   a candidate, or that belongs to more than one candidate's cluster, is
   dropped (see `separable_clusters`).
2. Every `(context, definition)` pair, for every synset in every cluster,
   is scored by **two** DeBERTa-v3-small cross-encoders. `encoder-v` was
   trained on nouns and verbs, `encoder-nv` on every POS except verbs. Both
   encoders score every pair, and their 2-class softmax outputs are averaged.
   The target word is marked in the context as plain text, `[D] word [D]`.
   The definition side is the bare gloss.
3. A candidate's score is `sum(p1 * softmax(|p1 - p0|))` over its cluster,
   a confidence-weighted sum. The highest-scoring candidate wins.

The released sense graph is a BabelNet 5.3 extract (`definitions.json` /
`neighbours.json`). It is under BabelNet's **non-commercial** license and
covers only the benchmark candidates. linkingtk does not ship it.
Instead, `WordNetNeighbours` builds clusters from WordNet's
hypernym/hyponym links, which works for any lemma. `JsonNeighbours` reads
the reference's own format for anyone who has the BabelNet extract.

```python
from linkingtk.algorithms.wsd import SandwichLinker, WordNetNeighbours
from linkingtk.sources.wn import WnEntitySource

linker = SandwichLinker.from_checkpoint(
    "~/.cache/linkingtk/sandwich_checkpoint",  # contains encoder-nv/ and encoder-v/
    neighbours=WordNetNeighbours("omw-en:1.4"),
).to("cuda")
results = linker.link(mentions, WnEntitySource("omw-en:1.4"))
```

## Reproducing the released results

`examples/sandwich_reproduction.py` loads the released cross-encoders
(`models.zip` from the Google Drive link in the reference README) and
evaluates them on UFSAC's Raganato et al. (2017) splits. Candidates are
restricted to the gold POS, as in the reference's own benchmark files
(every instance's candidate list there is single-POS).

```bash
uv run python examples/sandwich_reproduction.py
# with the reference's BabelNet graph instead of WordNet's:
uv run python examples/sandwich_reproduction.py \
    --babelnet-dir sandwich/data/babelnet --bn2wn bn2wn.json
```

| | SE07 | SE2 | SE3 | SE13 | SE15 | ALL |
|---|---|---|---|---|---|---|
| Paper, Table 1 (unreleased code) | 81.2 | 88.5 | 84.9 | 92.5 | 91.7 | 89.0 |
| Reference code run unchanged ("oracle") | 80.9 | 87.8 | 85.7 | 92.6 | 91.2 | **88.4** |
| linkingtk, reference BabelNet graph | 81.3 | 87.2 | 84.5 | 92.6 | 91.0 | 87.9 |
| **linkingtk, `WordNetNeighbours` (default)** | 78.9 | 86.9 | 83.5 | 91.2 | 90.5 | **87.0** |
| linkingtk, no neighbours (`--relations`) | 79.3 | 85.6 | 82.9 | 89.8 | 89.8 | 86.1 |
| *ESC, for comparison (`esc.md`)* | 76.3 | | | | | 80.6 |

**The scoring code is exact.** We fed the reference's own SE07 instances
straight through `SandwichLinker` with `JsonNeighbours` and
`separable=False`. Its predictions were identical to the oracle's on
**455/455** instances, with the same 80.9 F1.

**The remaining 0.5 points between the oracle and the BabelNet row come
from inputs, not code.** A per-instance diff on SE3 shows:

- **Candidate inventories differ.** The reference's candidate lists come
  from BabelNet's lemma index. They sometimes *omit* real WordNet 3.0
  senses (e.g. *willing*: 1 candidate vs WordNet's 2), and sometimes add
  BabelNet-only senses that can never be gold. Across the 7,253 ALL
  instances, 251 reference candidate sets miss at least one WordNet sense,
  and 21 of those collapse to a single trivially-correct candidate.
- **Marker placement differs.** The reference's `__add_target_tokens`
  marks the *first* occurrence of the target's surface form, not the
  instance's actual position. It also omits the markers entirely when that
  occurrence is the sentence's last token. `SandwichLinker` always marks
  the real span. That changes a handful of predictions per split, and on
  SE3 it changed slightly more of them for the worse.

**The further 0.9 points down to the WordNet row is the graph itself.**
Of the reference's BabelNet neighbour edges that map back to WordNet,
about 67% are WordNet one-hop relations, mostly a subset of
hypernym/hyponym links. About 29% are BabelNet-only edges with no WordNet
counterpart (e.g. *melody* → *cantus firmus*, *aria*). Swapping in other
WordNet relations (meronymy, domains, entailment, a neighbour cap) made
no difference beyond noise on the SE07 dev set (±2 instances). Turning
off sense-separability cost 2.4 points on SE07. Even so, WordNet
clusters are a real gain over no clusters at all: ALL 87.0 vs 86.1.

## What the reference's numbers mean

Two things to keep in mind when comparing SANDWiCH to other systems:

- **The README's "ALL = 88.6" includes SemEval-2010.** The reference's
  `test.json` has 8,062 instances: SE2, SE3, SE13 and SE15 plus 1,264
  SemEval-2010 instances (87.1 on their own). It excludes SE07. On the
  standard 7,253-instance Raganato ALL (SE2+SE3+SE07+SE13+SE15), computed
  from the reference's own predictions, the score is **88.4**.
- **We checked the graph for gold-label leakage and found none.** Gold
  candidates have larger BabelNet clusters on average (11.1 vs 5.8 synsets),
  and the score sums over the cluster. But picking the largest *WordNet*
  degree predicts gold slightly better (45.1% on ambiguous instances) than
  picking the largest BabelNet cluster (42.1%). So the size gap reflects
  the ordinary sense-frequency effect, not a graph built with knowledge of
  the test labels.

## Training from scratch

The reference releases no training code. `examples/sandwich_full_training.py`
follows the paper (Sec. 3.2/4.1, Table 7):

- Backbone, loss and schedule: DeBERTa-v3-small, binary cross-entropy,
  AdamW (weight decay 0.01), lr 2e-5, batch 64, 10 epochs, cosine
  annealing, gradient clipping at 1.0.
- Data: trained on SemCor (UFSAC `semcor.xml`), with SE07 as dev. The best
  epoch is selected by SE07 F1 *through the real cluster-level linker*, not
  by single-pair Hits@1.
- Pairs: "all elements within a neighborhood share the same label". Each
  gold sense plus up to 3 of its cluster members are positives. Each wrong
  candidate plus up to 2 of its cluster members are negatives, capped at
  12 per instance. These sampling sizes are our own choice, because the
  paper does not state them. Monosemous instances are skipped.
- Encoders: `encoder-v` is trained on noun+verb instances, `encoder-nv` on
  noun/adjective/adverb instances.

Supporting this needed three generic additions to
[`CrossEncoderTrainer`](../reference/train.md):

- explicit `negative_data`, used instead of blocking-mined hard negatives;
- an `on_epoch_end` callback;
- `TrainingArguments.lr_scheduler="cosine"` and `max_grad_norm`.

### Results

The full run took about 16 hours on one RTX 4090 in fp32: 10 epochs of
`encoder-v` (2.17M pairs), then 10 of `encoder-nv`. The best SE07 epochs
were `encoder-v` 8 (71.9) and `encoder-nv` 4 (58.9). `encoder-nv` scores
low on SE07 alone because SE07 is mostly verbs, which it is never trained on.

All rows below go through the same `SandwichLinker` with
`WordNetNeighbours`, so only the encoders differ:

| Encoders | SE07 | SE2 | SE3 | SE13 | SE15 | ALL |
|---|---|---|---|---|---|---|
| Released pair | 78.9 | 86.9 | 83.5 | 91.2 | 90.5 | **87.0** |
| **Trained here on SemCor** | 69.9 | 78.0 | 76.1 | 77.1 | 79.2 | **77.0** |
| Released `encoder-v` alone | 80.4 | | | | | 82.7 |
| Trained `encoder-v` alone | 71.9 | | | | | 74.3 |
| *GlossBERT, SemCor-trained BERT-base (`glossbert.md`)* | | | | | | *76.7* |

**Our recipe reaches about the level that SemCor-only cross-encoders
usually reach, but not the released checkpoints' level.** We traced the
10-point gap to the training data, not the recipe:

- **The released encoders know senses that SemCor never shows.** For ALL
  instances whose gold sense does not occur in SemCor (863 of 7,253), the
  released `encoder-v` is *more* accurate (86.1) than on SemCor-seen senses
  (82.3). Every SemCor-only system drops on unseen senses, and ours does
  too (75.5 seen vs 65.6 unseen).
- **They know WordNet's own example sentences.** We disambiguated 3,000
  WordNet example sentences, taking the example's own synset as gold,
  half with senses seen in SemCor and half unseen. The released
  `encoder-v` scored 76.5 on the seen half and **80.3 on the unseen
  half**. Ours scored 60.9 and **31.1**. Having no SemCor frequency prior
  at all, as the released model does, cannot come from SemCor-only
  training.
- **It is not memorisation of the test sets.** A context-free probe gives
  the released model the same advantage on unseen senses on the dev set
  SE07 (+10.5 over ours) as on the four test sets (+5.7 to +11.6). The
  probe scores `[D] word [D]` with the sentence removed. A model trained on
  the test documents would show the advantage on the test sets only.

So the released encoders were very likely trained on SemCor **plus**
sense-annotated data covering WordNet broadly, for example WordNet's glosses
and examples, which BabelNet bundles. The paper only says it "train[s] on
the SemCor corpus". The obvious next step, not run here, is to add WordNet
examples and glosses as training contexts (`WordNetNeighbours` already
supplies the definitions) and retrain.

This run also exposed a generic bug, now fixed. `CrossEncoderTrainer`
never put the model into train mode, and HF `from_pretrained` returns
models in eval mode. As a result, epoch 1 of any freshly loaded
transformer trained with dropout off. Dropout came back on only after the
first evaluation. In this run that affected only epoch 1, and the best
epochs are 8 and 4.
