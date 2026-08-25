# Agentic Autoresearch for Cell-Edge Power Control

Code, evaluator and campaign log for:

> A. A. Khan, A. Bin Sediq, S. Azadegi Naeini and R. S. Adve, "Agentic Autoresearch for Cell-Edge Power Control: Radically Redefining the Researcher's Role."

An AI coding agent was given an immutable evaluator and a research charter, and
authority over the architecture, input representation, output parameterization,
loss function and task-sampling law of a learned power-control model. Over
roughly eighty unattended experiments spanning six architecture families and
twenty-six hours, it closed **94% of the gap** between its own first working
architecture and a converged fractional-programming reference on
sum-least-percentile-rate power control — a problem that is non-convex,
non-smooth and strongly NP-hard away from its max-min vertex.

The repository has two halves, and you probably want one of them:

### → [`verify/`](verify/) — check the claim

One command reproduces the paper's held-out number.

```bash
pip install -r requirements.txt
cd verify && python verify.py --qft
```



```
HELDOUT_SCORE : 1.477473      [paper: 1.4775]
QFT_SCORE     : 1.4850        [paper: 1.4850]
ratio         : 99.5%
```

### → [`autoresearch/`](autoresearch/) — run your own campaign

The protocol, the charter, and the seed script. Start with
[`autoresearch/PROTOCOL.md`](autoresearch/PROTOCOL.md).

---

## The evaluator is the whole argument

`verify/prepare.py` is **immutable**. The agent could import it and never edit
it, and its hash was verified every iteration. That is what makes the campaign's
numbers mean anything: a self-improving system that can edit its own test will
eventually make the test easier instead of the model better.

Confirm the judge you were shipped is the one the campaign was scored against:

```bash
sha256sum verify/prepare.py
```

```
SHA-256: 7d43c742c4d8fd76f304240562018b7c913d423c5d43255a06ea0b2c2c0af95a
```

The held-out set is **not shipped as data**. `prepare.py` regenerates it from
pinned seeds (`TEST` seeds 5000–5010), disjoint from the training stream
(20,000,000+) and the pool (1000–1520). There is no curated file to distrust.

## The result in one table

| | Score | Notes |
|---|---|---|
| Full-power floor | 1.0000 | the trivial baseline the metric is normalised against |
| First working architecture (exp 1) | 1.3687 | equivariant cell-coordinated MPNN |
| **Champion (exp 81)** | **1.4775** | one fixed feed-forward pass |
| QFT reference | 1.4850 | converged, sample-matched on identical pinned drops |

An independent reproduction on separate hardware recomputed the QFT bar from
scratch at **1.485645** and confirmed the exactness result against the CVXPY
solver; see [`verify/VERIFICATION.md`](verify/VERIFICATION.md).

### Reproducibility of the headline number

The paper reports `HELDOUT_SCORE = 1.4775`, the score banked by experiment 81
and recorded in commit `5b5c132` of the campaign repository. Retraining from
that same commit on the same machine reproduces `1.477022`. The difference of
`0.00045` lies **inside** the ±0.0005 noise band reported in the paper, which
arises from non-deterministic reduction order under multithreaded PyTorch; the
band was calibrated by exactly this kind of repeated identical run. The shipped
`verify/last_model.pt` is from that retrain and scores `1.477022`.

The per-cell grid is far more stable than the sixth decimal of the mean. In
particular the minimum-percentile row reprints exactly —
`1.096 / 1.233 / 1.526 / 1.825 / 2.024 / 2.258` — as Proposition 2 requires,
and matches the QFT reference on those cells to three decimals.

### A note on `autoresearch/log.csv`

The log is the complete, unedited campaign record and runs to experiment 88.
The paper reports the campaign through **experiment 81**, the champion. The
experiments after it explored a second inference pass and sampling-law variants
whose gains were within or near the noise band while roughly doubling inference
time; they were outside the scope the paper reports, and the log is published
whole rather than truncated to the reported range.

At the minimum percentile the model's output is the exact max-min optimum
**for every value of its trained weights** — an algebraic identity, not a
learned property. See Theorem 1 in the paper; the mechanism is the cut clamp
followed by the balancing recursion.

## Environment

Bit-exact reproduction depends on the torch version and thread count. See
[`requirements.txt`](requirements.txt) and record what you verified under. Small
deviations are expected across environments; the campaign's own noise band was
±0.0005.

## Citing

```bibtex
@misc{khan2026autoresearch,
  author        = {Khan, Ahmad Ali and Bin Sediq, Akram and
                   Azadegi Naeini, Sara and Adve, Raviraj S.},
  title         = {Agentic Autoresearch for Cell-Edge Power Control:
                   Radically Redefining the Researcher's Role},
  year          = {2026},
  eprint        = {ARXIVID},
  archivePrefix = {arXiv},
  primaryClass  = {eess.SP},
  doi           = {10.48550/arXiv.ARXIVID},
  url           = {https://arxiv.org/abs/ARXIVID},
  note          = {Code and experiment log:
                   \url{https://github.com/ahmadkhan2020-cyber/autoresearch}}
```

## License

<!-- FILL IN. Check Ericsson and University of Toronto clearance before
     publishing prepare.py and program.md in particular. -->
