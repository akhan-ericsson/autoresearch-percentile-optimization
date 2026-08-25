# The champion weights — `last_model.pt`

**`verify/last_model.pt` is included**, and scores `1.477022` — see the
reproducibility note in the top-level README. This file documents where it came
from and how to confirm any checkpoint's identity, since weights are the product
of a specific training run on specific hardware and cannot be reconstructed from
source.

## Where it already exists

`train.py` writes it automatically at the end of every run:

```python
score = evaluate(model)
torch.save({"state_dict": model.state_dict(),
            "score": score,
            "arch": type(model).__name__}, "last_model.pt")
```

So the working directory of the **experiment-81** run already contains the right
file. Recover it the same way as `train.py` itself:

```bash
git log --oneline | grep -i "exp 81"
git show <commit>:last_model.pt > verify/last_model.pt   # if it was committed
```

If the checkpoint was not committed, it is whatever `last_model.pt` was left on
disk when experiment 81 finished — or re-create it by checking out the exp-81
commit and running `python train.py` once on the campaign machine.

## If `last_model.pt` was overwritten

Every run writes to the same bare filename, so the file on disk is from the
*last* run the harness executed, not from experiment 81. If the campaign kept a
per-family checkpoint folder, identify what is in it:

```bash
python identify_checkpoints.py families/
```

Note that experiments 81-89 all belonged to the same family
(`interference_attention`), so that family's *last* checkpoint is from the end of
the campaign, not from experiment 81. If nothing scores 1.477473, regenerate:
check out the experiment-81 commit and run `python train.py` once. It retrains
in one to two minutes and rewrites `last_model.pt`.

## Confirm before shipping

```bash
cd verify && python identify_checkpoints.py last_model.pt
```

**It must print `1.477473`** (the paper's 1.4775). If it prints anything else,
it is from a different experiment and should not be shipped as the champion:

| Value | What it is |
|---|---|
| `1.477473` | experiment 81 — **the champion; ship this** |
| `1.478752` | experiment 85 — later, out of the paper's scope |
| `1.476605` | a single-core reconstruction — see `VERIFICATION.md`; **not** the champion |
| anything else | identify it before shipping |

## Then

```bash
python verify.py --ckpt last_model.pt
```

which scores in a few seconds instead of retraining, and should reprint the
minimum-percentile row `1.096 / 1.233 / 1.526 / 1.825 / 2.024 / 2.258` together
with `HELDOUT_SCORE 1.477473`.

Bit-exactness depends on torch version and thread count; see `VERIFICATION.md`.
The grid ratios are far more robust than the sixth decimal of the score.
