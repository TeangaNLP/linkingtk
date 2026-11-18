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
| Trained here on SemCor + WordNet examples (`--wordnet-examples`) | 71.0 | 78.3 | 76.5 | 79.2 | 81.1 | 78.0 |
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

So the released encoders were very likely trained on SemCor **plus**
sense-annotated data covering WordNet broadly. The paper only says it
"train[s] on the SemCor corpus".

### Is it test-set leakage?

We found no sign of it. Every probe below compares the released
`encoder-v` with ours through the same linker. These are statistical
probes, not proof.

- **Probe 1: with the sentence removed, the advantage is the same on dev
  and test.** Here the model sees only `[D] word [D]`. The released model's
  lead over ours on SemCor-unseen senses is +10.5 on the dev set SE07 and
  +5.7 to +11.6 on the four test sets. A model that had memorised the test
  documents would gain on the test sets only.
- **Probe 2: the pattern holds on a corpus outside the benchmark.**
  SemEval-2007 Task 7 is not a Raganato split. There, too, the released
  model is more accurate on SemCor-unseen gold senses (97.9) than on seen
  ones (90.9).
- **Probe 3: deleting words doesn't break the test sets more.** We deleted
  30% of the non-target words at random and measured the accuracy drop
  for the released model / ours. A model that memorised test sentences
  should break more on the test sets. Instead:
  - SE07 (dev): 9.2 / 7.0
  - SE2: 4.3 / 2.1
  - SE3: 8.2 / 3.0
  - SE13: 6.1 / 0.8
  - SE15: 5.5 / 1.7
  - Task 7: 4.0 / 1.9

  The released model drops more everywhere, including on the dev set. That
  suggests it uses the context more and the SemCor sense frequencies less,
  not that it memorised test sentences.
- **Probe 4: the graph carries no hidden knowledge of the answers.** See
  [above](#what-the-references-numbers-mean).

Two loose ends remain:

- **MASC.** On a 6,000-instance sample of MASC, also outside the
  benchmark, the released model's advantage mostly disappears: 67.1 vs our
  68.6 on SemCor-seen senses, +7.1 on unseen. We have not explained this.
  MASC's different genres and noisier annotation are plausible causes.
- **The paper's unreleased first stage.** The paper's 89.0 also uses a
  candidate-filtering model trained on "the SemCor corpus [...] combined
  with datasets from Raganato et al. (2017a)", which could include the test
  sets. It is not released and none of the numbers here use it, so this
  page cannot check it.

### Which extra data? Probing WNGT

UFSAC's `wngt.xml` (WordNet Gloss Tagged) is the usual extra WSD training
corpus, and it is not part of Raganato's evaluation collection. It holds
every WordNet gloss, with the content words *inside* it sense-tagged, plus
the synset's examples, with the headword tagged in its own example. So the
WordNet-examples probe above was mostly scoring WNGT instances.

We took 1,500 random noun/verb instances per cell. Monosemous instances
were not excluded; they are trivially correct for both models. Below is the
released `encoder-v`'s lead over ours, in points, next to its lead on
data it could not have been trained on:

| | SemCor-seen gold | SemCor-unseen gold |
|---|---|---|
| Raganato ALL (benchmark) | +6.8 | +20.5 |
| SemEval-2007 Task 7 (held out) | +2.9 | +17.7 |
| WNGT, words in example sentences | +3.4 | +24.3 |
| WNGT, tagged words *inside glosses* | −0.6 | +9.6 |

- **Example sentences:** the released model's lead is slightly larger than
  on held-out text, which fits training on WordNet's examples.
- **Gloss-internal words:** its lead is *smaller* than on held-out text,
  and on SemCor-seen senses it ties our SemCor-only model. A model trained
  on WNGT's gloss annotations should do best there.

So WNGT as a whole was probably *not* in the training data, which left
WordNet's example sentences as the leading candidate.

### Retraining with WordNet examples

`--wordnet-examples` adds 47,876 instances to SemCor's: every WordNet
example sentence, with its synset's own headword as the target and the
example alone as context (the gloss is left out, since it would put the
answer in the context). Otherwise it uses the same recipe and seeds as
the SemCor-only run. Two other things changed: the trainer fix below was
in place from epoch 1, and the extra data added about 10% more pairs.

**It barely helps: ALL rises from 77.0 to 78.0** (table above). The best
epochs are `encoder-v` 7 (SE07 71.9) and `encoder-nv` 3 (60.9). On ALL's
SemCor-unseen gold senses, the new `encoder-v` reaches 67.6, against 65.6
before and 86.1 for the released one.

The breakdown below shows why. These are ambiguous instances only (at
least 2 candidates), released `encoder-v` / ours (SemCor + examples):

| | SemCor-seen gold | Unseen, has WordNet example | Unseen, no WordNet example |
|---|---|---|---|
| Raganato ALL | 79.1 / 71.1 | 66.0 / 38.3 | **89.4 / 53.3** |
| SemEval-2007 Task 7 (held out) | 89.6 / 86.1 | 94.7 / 64.0 | **97.6 / 79.8** |
| MASC sample (held out) | 65.8 / 68.4 | 47.1 / 49.0 | 57.6 / 54.3 |

- **The released model's strongest cell has no WordNet example to learn
  from.** That is ambiguous senses that appear in neither SemCor nor any
  WordNet example. There it scores 89.4 on Raganato, *higher* than on
  senses SemCor does cover. So WordNet examples cannot be the missing data.
- **OMSTI is not it either.** OMSTI is the other training corpus in
  Raganato et al.'s framework; the paper's first stage is said to use
  "datasets from Raganato et al.". It covers only 70 of these 509
  ambiguous unseen Raganato instances, and the released model is *less*
  accurate on those (64.3) than on the rest (78.4).
- **The effect is present on SemEval-2007 Task 7 but absent on MASC.**
  Task 7 is outside the benchmark, which argues against plain test-set
  memorisation. On MASC the released model does no better than ours in
  any cell, even on SemCor-seen senses. Noisy MASC annotation could cap
  both models; we have not verified that.

**Where the released encoders' knowledge of these senses comes from is
still open.** The remaining candidates are:

- BabelNet-only material such as Wiktionary or Wikipedia glosses and
  examples. The authors use BabelNet 5 throughout, and under its licence
  it can't be shipped or tested here.
- Training data that overlaps the evaluation sets.

The probes above argue against the second, but cannot exclude it.

This run also exposed a generic bug, now fixed. `CrossEncoderTrainer`
never put the model into train mode, and HF `from_pretrained` returns
models in eval mode. As a result, epoch 1 of any freshly loaded
transformer trained with dropout off. Dropout came back on only after the
first evaluation. In this run that affected only epoch 1, and the best
epochs are 8 and 4.
