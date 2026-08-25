# The protocol

Three files, strictly separated. An agent edits exactly one of them.

```
prepare.py    IMMUTABLE EVALUATOR   the agent may import it, never edit it
train.py      THE ONLY MUTABLE FILE model + loss + training loop + changelog
program.md    RESEARCH CHARTER      written by the human; goal, metric, protocol
```

## The loop

Each cycle, unattended:

1. The agent reads `program.md` and the changelog accumulated in `train.py`.
2. It states **one** hypothesis and a **pre-registered falsifier** — the
   concrete observation that would disprove it — before running anything.
3. It edits `train.py`.
4. The inner loop runs: one complete fixed-budget training run (here, 2000 Adam
   steps, ~1–2 min on CPU).
5. `prepare.py` scores the result on the pinned held-out grid.
6. If the score improves beyond the noise band, `git commit`. Otherwise
   `git revert`. Either way the outcome — including whether the falsifier fired
   — is appended to the changelog before the next cycle begins.

## The five safeguards

These are what make the output trustworthy rather than merely impressive. Each
earns its place.

**1. Hash-pin the evaluator.** Verify `prepare.py`'s SHA-256 every iteration.
This removes, by construction, the characteristic failure of self-improving
systems: making the test easier instead of the model better. An agent that
cannot edit the judge cannot game the metric.

**2. Enforce an inference contract.** Here: a 10-second budget for inference
over the whole grid, a no-test-time-fitting tripwire, and output-shape guards.
A violation raises an exception rather than returning a score, so it cannot be
banked. Without this, an agent can satisfy the metric by smuggling per-instance
optimisation into "inference" — which defeats the point of learning an
amortised model in the first place.

**3. Pre-register a falsifier for every experiment.** Requiring the agent to
say, in advance, what result would count *against* its hypothesis is what turns
the log from a list of scores into a sequence of interpretable findings —
including the negative ones, which in this campaign outnumbered the positive
ones and were more informative.

**4. Calibrate a noise band.** Run the same configuration repeatedly and measure
the spread (here ±0.0005). Any "improvement" inside the band is not an
improvement. Without this an agent will happily bank drift.

**5. Score sample-matched on identical pinned realizations.** Comparing a
candidate against a reference on *different* draws imports realization variance
into every decision. A mid-campaign audit in this project found exactly this
violation and it retroactively changed the reading of several earlier results.

## Governing exploration

Breadth by charter, not by chance. Cap the number of distinct architecture
families (here: six). Give each a protected grace window before pruning it back
to the incumbent, then spend the remaining budget on depth in the most
promising one.

This cap was itself learned. An earlier campaign opened seventeen families with
a single experiment each and tuned none of them; the result was broad, shallow
and worthless. The companion failure was scope: one model straddling a
qualitative shift in the optimal policy. Narrowing the target band fixed it.

## Writing the charter

Write `program.md` in firm, authoritative, unambiguous language, and state not
only what the agent should attempt but what it **must not** do. An agent running
unattended for tens of hours will act on whatever latitude the charter leaves
it, and hedged phrasing invites reinterpretation of the protocol precisely when
no human is present to correct it. Express the family cap, the falsifier
requirement and the prohibition on editing the evaluator as obligations, not
suggestions.

## Choosing the models

In this campaign the labour divided as follows, and the split is worth copying:

| Role | Model |
|---|---|
| Set up the campaign — evaluator, seed training script, charter | Claude Opus 5 |
| Harness, and all code writing | Claude Code |
| Drive the outer loop | Claude Sonnet 5 |

The initial design demands the stronger model. Once the protocol is fixed, the
iterative loop is well served by a faster one.

## What makes a problem a good fit

You need three things. If any is missing, the loop has nothing to grip:

* a **scalar figure of merit** that genuinely captures the engineering goal;
* a **fast, deterministic simulator** that can serve as an impartial judge —
  fast enough that a full inner loop costs minutes, not hours;
* a **mutable algorithmic artifact** worth improving.

Radio resource management supplies all three, which is why it was the target
here. So do many other engineering domains.
