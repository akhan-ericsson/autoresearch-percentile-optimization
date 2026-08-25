# Independent verification log

An end-to-end reproduction attempt run on a **single-core** container
(1 CPU / 1 thread, torch 2.13.0, CVXPY 1.9.2), against the campaign's own
`prepare.py` and `qft_reference.py`. The campaign itself ran on a 12-core
machine.

## 1. The evaluator — verified exactly

```
$ python prepare.py
grid: 17 cells  |  K in (1, 2, 4, 6, 8, 10)  |  250 drops/K  |  budget 10.0s
```

| Claim | Result |
|---|---|
| `B = 7` cells | confirmed |
| 17 grid cells (6 `min` + 11 `Kq>1`) | confirmed |
| why 17 and not 6x3 = 18 | at `K=1` (7 users) `p10` collapses onto `min`: `settings_for(1) == [('min',1), ('p25',2)]` |
| 250 pinned drops per K, `TEST` seeds 5000+K | confirmed |
| inference budget 10.0 s | confirmed |
| 7 wrapped hexagonal cells, 2R = 2000 m | confirmed (tessellation asserted at 2R sqrt(7)) |
| path loss d0 = 0.392, alpha = 3.76; Rayleigh fading; 43 dBm | confirmed |

Full-power SLqP denominators span **0.0577 – 9.8896 Mbps = 171x**, which is why
the metric is a per-cell ratio rather than a raw mean.

## 2. Theorem 1 — verified against an independent solver

The trained model's minimum-percentile row, and the CVXPY QFT reference
computed on the *identical* pinned drops:

| K | 1 | 2 | 4 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|
| model `min` | 1.096 | 1.233 | 1.526 | 1.825 | 2.024 | 2.258 |
| QFT `min` | 1.0959 | 1.2329 | 1.5257 | 1.8241 | 2.0223 | 2.2532 |
| model / QFT | 100.0% | 100.0% | 100.0% | 100.1% | 100.1% | 100.2% |

The model row is also character-for-character the campaign's own pinned gate
value (`the min row MUST reprint 1.096/1.233/1.526/1.825/2.024/2.258`).

At `Kq = 1` the model's output is the exact balancing fixed point, so the
model very slightly *exceeds* QFT at large K — QFT is iterative and stops at 10
iterations. This is the expected direction.

## 3. The reference bar — recomputed from scratch

Running `qft_reference.py` over all 17 cells x 250 pinned drops:

```
QFT_SCORE (mean over 17 cells) = 1.485645        [campaign / program.md: 1.4850]
```

Deviation **+0.00065 (+0.04%)**.

Per-cell timings are worth recording, because they bear on the amortisation
argument: the `K=10 min` cell alone took **685 s** for QFT, against **3.3 s**
for the model's entire 17-cell grid pass.

## 4. Exp 63's deficit law — independently reproduced

The log records QFT ahead of the student on all eleven `Kq>1` cells, with the
deficit a monotone function of `Kq`: *0.22% at Kq=2 rising to 2.27% at Kq=18*.
Measured here:

| | log | measured |
|---|---|---|
| `Kq = 2` deficit | 0.22% | **0.23%** |
| `Kq = 18` deficit | 2.27% | **2.06%** |

## 5. The champion score — reproduced to 0.06%

| Run | Score |
|---|---|
| campaign, experiment 81 (12-core) | **1.477473** |
| this reconstruction (1-core) | **1.476605** |
| difference | **-0.00087 (-0.059%)** |

Outside the campaign's ±0.0005 band, which was calibrated by repeated
*identical* runs on *their* hardware.

### Controlled test of the one un-flagged revert

The exp-88 teacher change (best iterate -> last iterate) is the only revert with
no documented flag. Isolating it, all else equal on this machine:

| | Score |
|---|---|
| A — last-iterate teacher (= experiment 81) | 1.476605 |
| B — best-iterate teacher (= experiment 88) | 1.475953 |
| effect of the revert (A − B) | **+0.00065** |

The revert moves the score **toward** experiment 81, and it independently
reproduces the behaviour the log documents. Exp 89's header records the
pre-exp-88 teacher sitting below its own anchor *"by −0.019 at K=4 up to −0.090
at K=10"*; measured here with the revert applied:

| K | 4 | 6 | 8 | 10 |
|---|---|---|---|---|
| teacher − anchor | −0.015 | −0.036 | −0.045 | **−0.089** |

The revert is therefore correct, and the residual −0.00087 is **not** attributable
to it.

## Conclusion

Everything that is deterministic reproduces: the evaluator, the physics, the
grid, the exactness theorem, the reference bar, and the per-cell deficit law.
The training-path score lands 0.06% low, which is consistent with a single-core
run of a 2000-step optimisation differentiated through a 40-pass fixed point —
floating-point reduction order differs with thread count and compounds.

**Re-run `verify.py` on the campaign's own hardware before publishing.** If it
prints `1.477473`, the reconstruction is exact. If it prints ≈`1.4766`, the
offset was this container's single core, and the git-recovered `train.py` should
be shipped instead.
