# `verify/` — reproduce the paper's held-out claim

One command, one screen of output.

```bash
pip install -r ../requirements.txt
python verify.py --qft
```

Expected:

```
HELDOUT_SCORE : 1.477473      [paper reports 1.4775]
QFT_SCORE     : 1.4856        [paper reports 1.4850]
ratio         : 99.5%
```

An independent single-core reproduction of all of this is recorded in
[`VERIFICATION.md`](VERIFICATION.md), including the QFT bar recomputed from
scratch (1.485645) and Theorem 1 checked against the CVXPY solver.

## Contents

| File | Role |
|---|---|
| `prepare.py` | **The immutable evaluator.** Channel model, the 17-cell grid, the scoring rule and the inference contract. Its SHA-256 is in the top-level README; check it. |
| `train.py` | The champion (experiment 81): model, loss, training loop, and the campaign changelog in its docstring. |
| `qft_reference.py` | The fractional-programming reference that sets the 1.4850 bar. |
| `verify.py` | Runs the grid and prints the table, `HELDOUT_SCORE` and `INFERENCE_S`. |
| `last_model.pt` | Champion weights (`1.477022`), so you can score without retraining. |
| `qft_grid.py` | Recomputes the QFT reference bar over the whole grid (~40 min single-core). |
| `WEIGHTS.md` | How to supply `last_model.pt`, and how to confirm it is the right one. |
| `VERIFICATION.md` | An independent end-to-end reproduction and what it established. |

## Why you do not need a test-set file

The held-out drops are **not shipped as data**. `prepare.py` regenerates them
from pinned seeds (`TEST` seeds 5000–5010), disjoint from the training stream's
seeds (20,000,000+) and from the pool seeds (1000–1520). Nobody has to trust a
curated file: the evaluator and the seed define the test set.

Training is cheap — roughly one to two minutes on a multicore CPU — so
`verify.py` retrains from scratch by default and needs no checkpoint at all.
Pass `--ckpt last_model.pt` to skip training and score shipped weights instead,
once those weights have been added (see [`WEIGHTS.md`](WEIGHTS.md)).

## The metric, in one paragraph

Raw SLqP values differ by orders of magnitude across the grid, so each of the 17
cells is scored as the ratio of the model's mean SLqP to the **full-power** mean
SLqP on the *same* pinned realizations. `HELDOUT_SCORE` is the mean of those
ratios. A score of 1.000 is therefore the trivial full-power floor. The grid is
17 cells rather than 18 because at K = 1 (7 users) the `min` and `p10` targets
coincide.
