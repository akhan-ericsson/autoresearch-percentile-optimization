# `autoresearch/` — the framework

The reusable half of this repository: the protocol, the charter that governed
this campaign, and the seed script the agent started from.

Start with **[`PROTOCOL.md`](PROTOCOL.md)**. It describes the three-file loop,
the five safeguards, how to govern exploration breadth, and how to write a
charter an unattended agent will not reinterpret.

## Contents

| File | Role |
|---|---|
| `program.md` | The research charter from this campaign — goal, metric, reference bars, inference contract, exploration protocol. The most directly reusable artifact here; adapt it to your problem. |
| `log.csv` | The machine-readable trace behind the progress figure: one row per experiment. |
| `PROTOCOL.md` | The protocol, written up. |

The full narrative changelog is not a separate file — it lives in the docstring
of `train.py` (in `../verify/`), which is itself a demonstration of the
append-only log discipline the protocol asks for.

## Running your own campaign

1. Write `prepare.py` for your problem: it must generate a pinned held-out set
   from fixed seeds, score a candidate, return one scalar, and enforce whatever
   contract your deployment implies. Then never touch it again.
2. Write `program.md`. Be explicit and be firm. See `PROTOCOL.md`.
3. Write a seed training script that runs end to end and scores *something*,
   however weak. The agent needs a working baseline, not a good one. The seed
   used here is the first commit of `../verify/train.py` in the campaign's
   version history.
4. Put all three under version control and point an agent at the loop, with
   commit-on-improve and revert-on-fail.
5. Audit the log, not the code.

## What to expect

In this campaign, ~80 experiments over ~26 hours took a learned power-control
model from 92.2% to 99.5% of a converged classical reference on a strongly
NP-hard problem.

Two findings are worth knowing in advance. First, the largest gains were
**diagnoses, not sweeps**: the single biggest improvement came from noticing
that the objective's gradient magnitude scaled with the percentile index and
normalising it away. Second, where a strong classical solution exists, the
search tends to **converge toward variants of it** — a phenomenon reported
independently by other agentic-search frameworks in this domain. Budget for
that: seed the classical solution as the zero point rather than hoping the
agent rediscovers it.
