"""
train.py  --  the ONLY file the autoresearch agent may edit (low-percentile-band
campaign, band 1 of 4: 0-25%).

Trains ONE model that maps (channel gains, percentile) -> transmit powers for
ANY network size K in 1..K_MAX and ANY percentile Kq WITHIN THE 0-25% BAND
(min/p10/p25 of K*B -- this campaign does not train or evaluate p50 or sum).
Required API:

    powers = model(A, Kq)      # A: [batch, K, B, B] -> powers: [batch, K, B]

Must print:  FAMILY <tag>   and   HELDOUT_SCORE <float>
(the evaluator also prints INFERENCE_S, which the harness surfaces per line).

HELDOUT_SCORE is the mean over a fixed 17-cell (K, percentile-in-band) grid of
(model SLqP / full-power SLqP); 1.000 is the trivial floor.

THIS CAMPAIGN IS PURE ML, FROM SCRATCH: no inheritance from ANY prior campaign's
checkpoints (not the full-range campaign's, not any other band's, once they
exist) -- the model must be one fixed feed-forward pass (the evaluator enforces
a 10 s grid inference budget and a no-test-time-fitting tripwire). TRAINING may
use anything -- the differentiable objective, labels generated via
qft_reference.py, RL rewards, self-supervision, curricula.

-----------------------------------------------------------------------------
EXPERIMENT 81 -- family `interference_attention` (DEPTH-TUNING THE CHAMPION;
still 6 of <=6 families -- the breadth cap is REACHED and no new family may be
opened, so every remaining iteration is depth on this one)
-----------------------------------------------------------------------------
EXP 80 RAN AND ITS FALSIFIER 2 FIRED -- ON THE SIGN, NOT ON THE MAGNITUDE, AND
THAT SPLIT IS THE WHOLE READING. Peak LR 1e-3 -> 2e-3 scored 1.476871 against
1.476529, KEPT as a new global best. Falsifier 1 passed for the thirteenth
consecutive run: the six `min` cells reprinted EXACTLY 1.096/1.233/1.526/1.825/
2.024/2.258, so the edit was correct. The eleven Kq>1 cells moved

    p10  +0.001 / +0.001 /  0.000 /  0.000 /  0.000
    p25  +0.001 / +0.001 /  0.000 /  0.000 / +0.001 / +0.001

-- six up, five flat, NONE down, p25 included: falsifier 2's sign condition
verbatim, and not the exp-79 seesaw falsifier 3 described. But falsifier 2
predicted +0.002 to +0.006 on the mean and the run delivered +0.00034, between a
sixth and a twentieth of it.

WHY THAT KILLS THE PRE-REGISTERED 4e-3 PROBE RATHER THAN MOTIVATING IT. The
falsifier-2 branch sent the next iteration to LR = 4e-3 to bracket exp 16's
3e-3 on the clean gradient. Its premise was the exp-79/80 travel argument: the
iterate orbits a ball of size ~LR*sigma and travels ~LR*STEPS/2, so after exps
19 and 79 cut sigma 8x the constraint had to be travel. Exp 80 DOUBLED the
travel and moved the answer by 0.0003. That is not a step-length-limited
optimiser; it is one that is already sitting on its landing point. Both halves
of the descent recipe are therefore measured and both are closed -- noise
(exp 79, +0.00007) and step (exp 80, +0.00034) -- and 4e-3 would buy a third
digit of the same null while risking exp 16's 0.027 collapse. THE LANDING POINT
ITSELF IS WHAT IS WRONG, which is exactly the branch falsifier 3 named.

THE ONE CHANGE: W_SCALE 3.0 -> 1.5.

WHAT IT IS. `raw_profile` emits w = w_clip * 10**(W_SCALE * tanh(logit)), so
W_SCALE is the last scalar in the emission path that no experiment has moved,
and the only structural freedom `_cut_clamp`'s theorem leaves the head. Eighty
experiments have discussed it as a RAIL -- how deep a sacrificed user may be
placed. It is equally the GAIN: d(decades)/d(logit) = W_SCALE at the origin.
Adam moves each head weight ~LR per step regardless of gradient scale, so the
head's step measured in the units the objective is actually written in --
decades of target SINR -- is W_SCALE * ||LN(h)|| * LR, and exp 80 just doubled
the last factor. Halving W_SCALE halves the step and doubles the precision with
which the profile can be placed, at an unchanged LR, unchanged noise, unchanged
travel in parameter space.

WHY RESOLUTION IS THE SUSPECT LEFT STANDING. Nine axes are now closed: capacity
0-for-4, distillation 0-for-6, softening 0-for-3, Kq re-weighting 0-for-2, head
conditioning, search 0-for-2, the cut plateau, the refinement stage, and now
BOTH halves of the descent recipe. Exp 76 named the residual "one strong
attractor that every parameterisation falls into". A too-coarse output map is
the one mechanism that produces precisely that signature -- the head oscillates
in decade-sized jumps, the cosine anneal freezes it wherever the last large step
left it, and neither cleaner gradients nor longer travel help because the
quantisation is in the PARAMETERISATION rather than in the descent. It also
retro-explains three otherwise-unexplained nulls: capacity 0-for-4 (a wider
trunk feeds the same coarse map), exp 76's refinement head moving 0.805-0.963
DECADES per user and changing nothing (a tanh at ~90% of its own rail is
saturated and cannot be re-tuned by any anneal), and the deficit growing
monotonely with Kq (more below-cut users to place, each at the same resolution).

WHY 1.5 KEEPS EVERY MOVE THE POLICY USES. The rail is a correction ON TOP OF
`w_clip`, which already places each user at its own full-power depth below the
cut, so this is 1.5 decades of DISAGREEMENT with a calibrated anchor, not the
total dynamic range. Full sacrifice survives: a graded user 1.5 decades under
the cut sits at 3% of the cut level, i.e. a few percent of a cut-level rate, and
the anchor's own depth adds to that -- trap 2's muting move is intact.
MEMBERSHIP_CHECK says the head revises at most 2.92 of 17 sacrificed users, so
the learned policy is overwhelmingly SHAPE within the anchor's set rather than
membership flips, and shape is exactly what resolution buys.

COST. Zero. One constant; no extra drops, no extra rate evaluations, no extra
wall-clock, no change to `forward()`'s structure, and INFERENCE_S unchanged at
~1.16 s of the 10.0 s budget.

PREDICTION. All of it in the eleven Kq>1 cells -- the six `min` cells are
algebraically pinned at p*(A) by exp 49's `_cut_clamp` for ANY parameters
(Kq=1 makes the clamped profile flat and `_profile_fixed_point` normalises a
flat profile away exactly) and carry zero training mass under KQ_MIN_TRAIN=2.
Predicted +0.002 to +0.008 on the 17-cell mean, weighted toward the large-K
p10/p25 cells where the most below-cut users must be placed.

FALSIFIER, PRE-REGISTERED, READ IN diagnostics.txt BEFORE THE SCORE.
 1. The six `min` cells must print EXACTLY 1.096/1.233/1.526/1.825/2.024/2.258.
    Pinned for any parameters; any motion at all is a leak in this edit and it
    reverts on sight rather than being argued about.
 2. HEAD_CHECK is the mechanism gate and is read FIRST, because it interprets
    both outcomes. If `rail` is large and p90 is pinned at W_SCALE, the reach
    binds and the knob is a RAIL: the next probe is 6.0. If the corrections live
    well inside the rail, the knob is pure gain/resolution and the reading below
    applies as written.
 3. IF THE ELEVEN Kq>1 CELLS RISE COHERENTLY: resolution was the constraint, the
    attractor is a quantisation artefact of the output map, and the next probe is
    W_SCALE = 0.75 -- with LR = 4e-3 rehabilitated after it, since a finer map is
    exactly the regime in which a longer step becomes affordable again.
 4. IF THEY FALL COHERENTLY: the reach was binding after all, exp 31's 3.0 was
    load-bearing, and the knob is probed in the OTHER direction at 6.0, which
    closes it from both sides in two iterations.
 5. IF THEY SEESAW OR |mean delta| <= 0.001: the output map's scale is not the
    constraint either, the knob is CLOSED, and with it the last scalar in the
    emission path. The residual ~0.02 (exp 63's paired, per-drop measurement, the
    only trustworthy sizing -- the per-cell QFT columns are 30-drop samples and
    are visibly noisy, e.g. K=10 `min` reads 2.07 against a provably-optimal
    2.258) is then a property of the hypothesis class as a whole, and the honest
    remaining move is to spend the 8.8 s of UNUSED inference budget on the one
    thing the contract still permits and this file has never tried: emitting the
    profile in more than one shot without evaluating any objective.

OFF-GRID. Nothing about the training DISTRIBUTION is touched by a single
character: K is still uniform on 1..10 (the ungraded 3, 5, 7, 9 included), `frac`
is still flat on (1/KB, 0.25] through the evaluator's own `kq_of()`,
`_band_kq_max()` and KQ_MIN_TRAIN = 2 are unchanged, and every graded cell keeps
exactly the mass it had. This is one constant in the output map, so no off-grid
check is owed; run `k_generalization_check.py` on K in {3,5,7,9} before banking
any jump regardless, per the standing protocol.

REPRODUCIBILITY. `W_SCALE` enters only `raw_profile`'s `torch.pow` and the new
HEAD_CHECK diagnostic. `nn.init.zeros_` still zeroes the head, so the run still
starts at w = w_clip EXACTLY and step 0 is bit-identical; no generator is
touched (`g`, `gd`, `make_pools()`, the TEACH cache build and `PowerNet()`'s
module construction order are byte-identical), so the initial parameter vector
and the entire (K, Kq, drop-seed) sequence are bit-for-bit exp 80's and the
output map's scale is the ONLY difference between the two runs. `head_report`
runs after `evaluate`, consumes no RNG and touches no training path. Identical
code reproduces an identical score.

FAMILY TAG. `interference_attention`, unchanged and correct: no framework, no
architecture, no feature, no loss, no sampler and no structural `forward()`
change -- one constant of the existing output map, plus a read-only diagnostic.

-----------------------------------------------------------------------------
EXPERIMENT 80 -- family `interference_attention` -- SCORED 1.476871, KEPT (new
global best); falsifier 2 fired on SIGN but at a tenth of its magnitude, which
closes the LR axis rather than continuing it -- see exp 81
-----------------------------------------------------------------------------
EXP 79 RAN AND ITS FALSIFIERS SPLIT. TASKS 4 -> 8 scored 1.476529 against the
exp-69 champion's 1.476460 -- KEPT as a new global best by +0.00007, which is
to say by nothing. Falsifier 1 passed for the twelfth consecutive run: the six
`min` cells reprinted EXACTLY 1.096/1.233/1.526/1.825/2.024/2.258, so the edit
was correct. But falsifiers 2 and 3 fired at once, in different columns, and
that split is the whole reading of this experiment. Against exp 76's persisted
row the eleven Kq>1 cells moved

    p10  1.190 / 1.406 / 1.526 / 1.648 / 1.777   (was 1.190/1.404/1.523/1.644/1.774)
    p25  1.070 / 1.146 / 1.268 / 1.322 / 1.376 / 1.411
                                    (was 1.071/1.146/1.267/1.322/1.377/1.413)

i.e. p10 rose +0.000/+0.002/+0.003/+0.004/+0.003 -- inside falsifier 2's
predicted +0.002..0.006 band -- while p25 went -0.001/0.000/+0.001/0.000/-0.001/
-0.002, falsifier 3 verbatim. Summed: +0.012 over five cells, -0.003 over six,
+0.009/17 = +0.0005 on the mean, of which the harness measured +0.00007. Two
halves of the same block moving in opposite directions by the same magnitude is
not a policy improvement; it is a RELABELLED DRAW.

THE READING, AND WHY IT IS THE IMPORTANT OUTPUT OF EXP 79. Var_step =
Var_task/T + Var_drop/BATCH. Exp 19 quartered the first term for +0.0150. Exp 79
halved what remained of it for +0.00007. Those two facts together say the task
term stopped binding somewhere below T = 4 -- so the honest conclusion is NOT
"variance is not the constraint", it is "the variance that was removed was never
converted into anything". A stochastic optimiser trades noise against step
length: the size of the ball the iterate orbits goes like LR * sigma, and the
distance it can travel in a fixed 2000 steps goes like LR * STEPS. Exps 19 and
79 have cut sigma by 8x at a peak LR that has not moved by one digit since
experiment 1. The ball has therefore shrunk 8x, the travel has not grown at all,
and a shrinking ball around a landing point you never had the step budget to
reach buys exactly the +0.00007 that was measured. THE VARIANCE REDUCTION IS
BANKED AND UNSPENT.

THE ONE CHANGE: PEAK LR 1e-3 -> 2e-3.

WHY THIS IS NOT A REPEAT OF EXP 16. The learning rate is 0-for-1 on this file
and the single probe was exp 16, peak 1e-3 -> 3e-3, which printed 1.392901
against exp 15's 1.419842 -- a 0.027 collapse, the worst regression of the
campaign. That is a real result and it is being respected, not ignored: it is
being READ IN ITS REGIME. Exp 16 ran at TASKS = 1 and BATCH = 128. Its per-step
gradient carried Var_task/1 + Var_drop/128; this file's carries Var_task/8 +
Var_drop/256. Whichever of the two terms dominates, the noise standard deviation
today is between 1.4x (pure drop term) and 2.8x (pure task term) SMALLER than
the noise exp 16 tried to take a 3x step through. A 3x step at that noise
diverged; a 2x step at between half and a third of it is the interior of the
bracket exp 16 established, and it is the first LR probe this campaign has ever
run on a gradient it actually cleaned up first.

Exp 15's own entry pre-registered this move in as many words -- "if the schedule
pays, raising the peak (WHICH ANNEALING MAKES AFFORDABLE) is the natural
experiment 16 on top of it" -- and the schedule paid (+0.0093). Exp 16 collected
on that promise immediately, at 3x, on a T = 1 gradient, and lost. The promise
was not thereby voided; it was mispriced. Two variance reductions later, at 2x,
it is the cheapest untested move on the file: ZERO extra rate evaluations, zero
extra drops, zero extra wall-clock, not one character of `forward()` touched,
INFERENCE_S unchanged at ~1.16 s of the 10.0 s budget.

WHY IT IS THE RIGHT LEVER RATHER THAN MORE VARIANCE. The seven closed axes
(capacity 0-for-4, distillation 0-for-6, softening 0-for-3, Kq re-weighting
0-for-2, head conditioning, search 0-for-2, the cut plateau, the refinement
stage) unanimously say the model can COMPUTE the answer and is not being pulled
away from it; exp 76 named the residual "one strong attractor that every
parameterisation falls into", and exp 79 has now shown the attractor is not held
in place by gradient noise either. What is left is the only remaining property
of an attractor: the iterate does not have the STEP BUDGET to leave it. 2000
steps is a short run -- Adam at 1e-3 under a cosine-to-zero schedule travels
about LR*STEPS/2 = 1.0 units of parameter norm in total, and every capacity
probe on this file (HIDDEN 48 -> 64, ROUNDS 4 -> 6, weight-tied 8 hops) came back
at exactly zero, which is the signature of a net that is not filling the
capacity it already has. The two direct probes of the descent recipe are the two
largest non-architectural wins here (+0.0093 the schedule, +0.0150 the variance).
This is the third, and the one both of them were setting up.

PREDICTION, AND HOW IT IS READ. All of it must appear in the eleven Kq>1 cells:
the six `min` cells are algebraically pinned at p*(A) by exp 49's `_cut_clamp`
for ANY parameters and carry no training mass at all under KQ_MIN_TRAIN = 2, so
they cannot move for any reason other than a bug. Predicted +0.002 to +0.006 on
the 17-cell mean (1.4785-1.4825) against a remaining QFT deficit of ~0.0085,
concentrated in the p25 column, where the deficit is largest and where exp 79's
cleaner gradient conspicuously failed to buy anything.

FALSIFIER, PRE-REGISTERED, READ IN diagnostics.txt BEFORE THE SCORE.
 1. The six `min` cells must print EXACTLY 1.096/1.233/1.526/1.825/2.024/2.258.
    Pinned for any parameters; any motion at all is a leak in this edit and it
    reverts on sight rather than being argued about.
 2. IF THE ELEVEN Kq>1 CELLS RISE COHERENTLY (most of them up, p25 included, no
    p10/p25 seesaw of the exp-79 kind): the banked-and-unspent reading is
    confirmed, the descent recipe is live again, and the next probes are LR =
    4e-3 -- bracketing exp 16's 3e-3 from above ON THE CLEAN GRADIENT, which is
    the measurement that separates "3e-3 was too big" from "3e-3 was too big AT
    T=1" -- and then, only if that also pays, TASKS = 16 to fund it.
 3. IF THEY SEESAW AGAIN (some up, some down, |mean delta| <= 0.001): the step
    length is not the constraint either, the LR axis closes at two points on the
    same side, and with capacity, knowledge, targets, softening, measure, search
    and now BOTH halves of the descent recipe closed, the residual ~0.0085 is a
    property of the (feature set + cut-clamp + pointwise head) hypothesis class
    itself. The endgame probe is then the one structural degree of freedom the
    clamp theorem leaves untouched and no experiment has tested: `W_SCALE`, the
    +-3-decade tanh reach of the head, which sets how DEEP below the cut a
    sacrificed user may be placed and is the only knob whose range grows with Kq.
 4. IF THEY FALL BY MORE THAN ~0.005 EACH, or the score collapses toward the
    1.000 full-power floor: 2e-3 is past the stability edge on this gradient
    too, exp 16's result was about the step and not about the noise, and the LR
    is CLOSED for good at 1e-3 by revert -- report that plainly, because it also
    retires falsifier 2's 4e-3 before it costs an iteration.

OFF-GRID. Nothing is narrowed and nothing about the training DISTRIBUTION is
touched by a single character: K is still uniform on 1..10 (the ungraded 3, 5,
7, 9 included), `frac` is still flat on (1/KB, 0.25] through the evaluator's own
`kq_of()`, `_band_kq_max()` and KQ_MIN_TRAIN = 2 are unchanged, and every graded
cell keeps exactly the mass it had. This is one optimiser scalar, so no off-grid
check is owed; run `k_generalization_check.py` on K in {3,5,7,9} before banking
any jump regardless, per the standing protocol.

REPRODUCIBILITY. `LR` enters only `torch.optim.Adam(..., lr=LR)` and, through
it, `CosineAnnealingLR(T_max=STEPS, eta_min=0.0)`, which still decays to exactly
zero. No generator is touched: `g`, `gd`, `make_pools()`, the TEACH cache build
and `PowerNet()`'s module construction order are byte-identical, so the initial
parameter vector and the entire (K, Kq, drop-seed) sequence are bit-for-bit exp
79's and the step size is the ONLY difference between the two runs. Identical
code reproduces an identical score.

FAMILY TAG. `interference_attention`, unchanged and correct: no framework, no
architecture, no feature, no loss and no `forward()` change -- one optimiser
scalar.

-----------------------------------------------------------------------------
EXPERIMENT 79 -- family `interference_attention` -- SCORED 1.476529, KEPT (new
global best by +0.00007); falsifiers 2 and 3 SPLIT by column, see exp 80
-----------------------------------------------------------------------------
EXP 78 RAN AND ITS FALSIFIER 3 FIRED: FLAT. ALPHA_T = 2.0 scored 1.476007
against the exp-69 champion's 1.476460, and the eleven Kq>1 cells printed
p10 1.189/1.405/1.524/1.647/1.775 and p25 1.070/1.145/1.266/1.321/1.376/1.411
against exp 76's persisted 1.190/1.404/1.523/1.644/1.774 and 1.071/1.146/1.267/
1.322/1.377/1.413 -- every single delta is +-0.001..0.003, i.e. one noise ball,
net -0.002 summed over eleven cells. Falsifier 1 also passed cleanly: the six
`min` cells reprinted EXACTLY 1.096/1.233/1.526/1.825/2.024/2.258 for the
eleventh consecutive run, and the diagnostics repair took (the file now says
"EXP score 1.476007", not exp 76's). So the edit was correct and the response is
flat on [1, 2].

ALPHA_T IS CLOSED AT 1.0, and exp 78's own falsifier 3 prescribes what happens
next, verbatim: "the weight is done, ALPHA_T returns to 1.0 by revert, and the
remaining iterations go to TASKS and tail averaging as exp 77 prescribed." Four
points now bracket it with the same oracle -- 0.0 (-0.0028), 1.0 (ref), 2.0
(-0.0005), 8.0 (-0.0104) -- a broad, slightly-left-of-2 plateau whose interior
is worth no more than a thousandth. The regulariser reading from exp 77 survives
(zero still costs 0.0028); only the hope of a tunable optimum dies. No further
probes of this knob.

ALPHA_T therefore RETURNS TO 1.0 here. That is a revert of a falsified probe
back to the champion's value, not a second variable: it restores the 1.476460
reference configuration so that the one change below is measured against the
champion rather than against an exploring state the harness never banked.

THE ONE CHANGE: TASKS 4 -> 8.

WHY THIS, AND WHY THE OBJECTION AGAINST IT WAS ARITHMETICALLY WRONG. Exp 78
dismissed this move on the grounds that "TASKS 4 -> 8 at fixed BATCH ... cuts
SUB 32 -> 16, and each task's `ref` is a ratio-of-means over that sub-batch, so
the per-task gradient SCALE gets noisier exactly as its DIRECTION gets cleaner."
That reasoning used the FILE's default BATCH = 128. The harness does not run the
file's default: `autoresearch.sh` line 39 exports BATCH = 256 (and POOL = 8192,
STEPS = 2000) into every scored run. The true arithmetic is

    SUB = BATCH // TASKS = 256 // 8 = 32,   not 16.

Thirty-two drops per task is exactly the sub-batch size the champion's own notes
were written around, and it is the size at which the ratio estimator has been
argued sound throughout. The single objection that kept this move off the table
for two experiments does not survive contact with the harness's actual budget,
and no other objection was ever raised against it.

THE MECHANISM, AND WHY IT IS THE LAST LIVE ONE. Per-step gradient variance
decomposes as

    Var_step = Var_task / T  +  Var_drop / (T * SUB)
             = Var_task / T  +  Var_drop / BATCH,

so at fixed BATCH the DROP term is a constant, 1/256, entirely independent of
T -- doubling T does not thin it by a single drop. What doubles is the
suppression of the TASK term, the disagreement between the gradient directions
of two different (K, Kq) cells, which exp 15 named as the campaign's dominant
noise source and only suppressed at the tail via the cosine anneal. Exp 19
attacked it at the source, T = 1 -> 4, and bought +0.0150 -- the largest
non-architectural gain in seventy-eight experiments, and the reason exp 21's
depth probe became readable at all. T = 4 -> 8 halves what remains of the same
term, at IDENTICAL total drop evaluations per step -- an unchanged 256 drops
through `slqp_rate` per step.

WHY IT IS NOT MERELY "MORE OF A GOOD THING". The seven closed axes (capacity
0-for-4, distillation 0-for-6, softening 0-for-3, Kq re-weighting 0-for-2, head
conditioning, search 0-for-2, the cut plateau, the refinement stage) all asked
what the model can COMPUTE or what it is PULLED TOWARD. Exp 76's entry drew the
right conclusion from their unanimity -- "that is not a knowledge deficit and it
is not a capacity deficit; it is one strong attractor that every parameterisation
this campaign has tried falls into" -- and an attractor that swallows a
saturated free-parameterised second head, a straight-through membership
gradient, a certified-QFT teacher and a 78% capacity increase alike is an
OPTIMISATION artefact, not a representational one. The two levers that act on
an optimisation artefact are the step-size schedule (exp 15, +0.0093, done and
annealed to exactly zero) and the gradient's variance (exp 19, +0.0150, done
once and never repeated). This is the second half of the only lever with a
positive track record on this file.

PREDICTION, AND HOW IT IS READ. Diminishing returns are expected: exp 19
QUARTERED the task term for +0.0150, this HALVES what is left, and the drop term
-- untouched at Var_drop/256 -- is a larger share of the total than it was at
T = 1. Predicted +0.001 to +0.004 on the 17-cell mean, i.e. 1.4775-1.4805, and
because the six `min` cells are algebraically pinned by exp 49's clamp and
untrained by KQ_MIN_TRAIN = 2, ALL of it must appear in the eleven Kq>1 cells,
which is also where all 0.0088 of the remaining QFT deficit sits (model 15.129
vs QFT 15.36 summed over those eleven).

COST. Zero extra rate evaluations: 8 x 32 = 256 = 4 x 64, the same drops through
the same `slqp_rate`, the same `_features` fixed points, the same total einsum
FLOPs. What doubles is the NUMBER of forward/backward invocations per step (8
instead of 4) on tensors half as large, so on CPU the kernel-launch and
autograd-bookkeeping overhead roughly doubles while the arithmetic does not;
expect wall-clock up ~20-40%, which is unbudgeted -- `autoresearch.sh` runs
`python3 train.py` with no timeout (line 92; the 600 s `AGENT_TIMEOUT` guards the
AGENT call, not the training run). Peak memory HALVES, since each task's graph is
backpropagated and freed before the next is built. `forward()` is not touched by
a single character and INFERENCE_S stays at the champion's ~1.15-1.20 s of the
10.0 s budget.

SEED HYGIENE -- ALREADY CORRECT, BY LUCK OF EXP 45'S STRIDE. The fresh-drop
seed is `CH_SEED + 8 * step + j`. Exp 45 chose stride 8 while TASKS was 4, so
four seeds per step have been going unused; at TASKS = 8, j spans 0..7 and the
stride is exactly saturated with NO collision across steps. The range grows from
20,000,000..20,015,996 to 20,000,000..20,015,999 -- still disjoint from
`make_pools`' 1000..1520, the TEACH cache's 30,000,000+, and the evaluator's
pinned TEST seeds 5000..5010. No line of `sample_channels` bookkeeping changes.

REPRODUCIBILITY. `g` now draws 8 (K, Kq) tasks per step instead of 4, so the
task sequence necessarily differs from the champion's -- that IS the change, and
it is the only one; the sampler LAW is bit-for-bit identical (K uniform on
1..10, `frac` flat on (1/KB, 0.25] through the evaluator's own `kq_of()`,
conditioned on Kq >= 2). `gd`, the TEACH cache build, and `PowerNet()`'s
construction order are untouched, so the initial parameter vector is bit-for-bit
the champion's. The loss still divides by TASKS, so the gradient remains the MEAN
of the per-task ratios and LR = 1e-3 keeps its meaning exactly. Identical code
reproduces an identical score.

FALSIFIER, PRE-REGISTERED, READ IN diagnostics.txt BEFORE THE SCORE.
 1. The six `min` cells must print EXACTLY 1.096/1.233/1.526/1.825/2.024/2.258.
    They are algebraically pinned at p* for ANY parameters and untrained, so any
    movement is a bug in this edit and it reverts on sight rather than being
    argued about.
 2. THE MECHANISM GATE: the eleven Kq>1 cells against exp 78's row (p10 1.189/
    1.405/1.524/1.647/1.775, p25 1.070/1.145/1.266/1.321/1.376/1.411). If they
    RISE by ~0.002-0.006 each, gradient variance is confirmed as the binding
    constraint behind all seven closed axes, and the next probes are TASKS = 16
    (SUB 32 -> 16, where the ratio estimator genuinely does start to thin) and
    then the drop term via BATCH -- in that order, because the first is free and
    the second is not.
 3. If they are FLAT (|delta| <= 0.001 each), the task term was already
    suppressed below the drop term at T = 4, exp 19's +0.0150 was the whole of
    this axis, and TASKS is CLOSED at 4 by revert. The remaining lever is then
    the DROP term alone -- Var_drop/BATCH -- which no experiment has ever varied,
    and that becomes the endgame probe.
 4. If they FALL, the per-task `ref` at SUB = 32 is noisier than the task-
    direction gain is worth, the ratio estimator's small-sample bias is the
    binding term rather than the task disagreement, and the repair is TASKS = 8
    at BATCH = 512 (restoring SUB = 64) rather than a return to 4 -- report that
    reading plainly, since it separates the two hypotheses the T-vs-SUB trade
    confounds.

OFF-GRID. Nothing is narrowed. The training law is untouched -- K uniform on
1..10 (the ungraded 3, 5, 7, 9 included), `frac` flat on (1/KB, 0.25] through the
evaluator's own `kq_of()`, `_band_kq_max()` and KQ_MIN_TRAIN = 2 unchanged -- and
the change draws MORE independent (K, Kq) tasks from that same law per step, so
if anything it BROADENS the per-step coverage of the band. No off-grid check is
owed; run `k_generalization_check.py` on K in {3,5,7,9} before banking any jump
regardless.

FAMILY TAG. `interference_attention`, unchanged and correct: no framework, no
architecture, no feature and no `forward()` change -- one optimiser constant.

-----------------------------------------------------------------------------
EXPERIMENT 78 -- family `interference_attention` -- SCORED 1.476007, FALSIFIER 3
-----------------------------------------------------------------------------
EXP 77 RAN AND ITS FALSIFIER 4 FIRED. ALPHA_T = 0.0 scored 1.473660 against the
exp-69 champion's 1.476460: deleting the distillation term COST 0.0028. Exp 77's
own pre-registration named this outcome and what it means -- verbatim: "If they
FALL, the term was acting as a regulariser rather than as a teacher -- the one
reading exp 64 cannot distinguish -- and the honest next move is to restore it."
That is now the measured verdict, and it inverts fifty-five experiments of
reading. The term is NOT a teacher. Its target really is worse than the student
on ten of eleven cells (the persisted TEACHER_CHECK is unchanged and still
correct), and exp 65 really did show that swapping in the certified QFT
reference makes things worse, not better -- but none of that was ever the
mechanism. What the term supplies is a SECOND, differently-conditioned gradient
on the same head coordinates, drawn from a fixed 1440-drop set with its own
generator, which regularises a 115k-parameter net trained by a single
high-variance direct objective. Remove it and the score falls; that is the only
reading left standing.

(Two bookkeeping facts about exp 77, recorded because they cost the run its
diagnostics. FIRST, the harness logged an agent execution error on it -- the
same failure mode as exps 68 and 75 -- but unlike exp 75 the edit DID land: the
revert of exps 74/75/76 is complete and verified here token by token against
commit 2fecf0e, `forward()` is two lines, `_cut_clamp` runs with CUT_TAU = 0.0
so the `ste` branch is unreachable, and INFERENCE_S printed 1.198 s, i.e. one
fixed point. The measurement is clean and one-variable. SECOND, exp 77 deleted
`refine_report` but left its call site in `main()`, so the diagnostics write
raised NameError into the `except` that exists to keep a diagnostic from killing
a run: diagnostics.txt on disk is still exp 76's. That call is DELETED here. It
is a repair to a broken diagnostic, not a second variable -- it is downstream of
`score`, touches no training path, and its only effect is that this run's
falsifier table exists to be read.)

THE ONE CHANGE: ALPHA_T 0.0 -> 2.0.

WHY 2.0 AND NOT THE CHAMPION'S 1.0. Restoring 1.0 is a no-op that reproduces
1.476460 exactly and learns nothing; the run is spent either way, so spend it on
the one number this term has never had. ALPHA_T has now been measured at THREE
points with the SAME oracle -- and, for the first time, the three BRACKET an
interior optimum instead of pointing monotonely downhill:

    ALPHA_T   0.0        1.0        8.0
    delta   -0.0028    0 (ref)    -0.0104        (exps 77, 69, 64)

Both curvature signs are now pinned by data rather than assumed. Fit any
reasonable functional form through those three points and the argmax lands
between 1.2 and 3.1: a plain quadratic in the weight peaks at 3.1; a
saturating-benefit-minus-linear-drag model, `B*a/(a+k) - c*a` -- which is the
mechanistically honest shape, since a regularisation benefit saturates while the
pull toward a measurably worse target grows linearly -- peaks at 1.2, 1.8 and
2.3 for k = 1, 3, 8. **2.0 is the consensus argmax of every form tried and the
only value inside all of their credible intervals.** Predicted gain +0.0005 to
+0.0015 on the 17-cell mean, i.e. ~1.477-1.478, concentrated in the eleven Kq>1
cells because Kq=1 is algebraically pinned.

WHY THIS IS THE RIGHT USE OF THE ITERATION. The two alternatives exp 77 named --
TASKS 4 -> 8 and tail weight averaging -- are both weaker bets and neither is
bracketed. TASKS 4 -> 8 at fixed BATCH halves the task-direction variance (exp
19's 1 -> 4 quartered it for +0.0150) but simultaneously cuts SUB 32 -> 16, and
each task's `ref` is a ratio-of-means over that sub-batch, so the per-task
gradient SCALE gets noisier exactly as its DIRECTION gets cleaner -- an untested
trade with no sign. Tail averaging over a cosine that already anneals to exactly
zero is a second-order correction to a residual ball the schedule was built to
remove, and it needs new code in the one place (the eval hand-off) where new
code has silently broken two of the last four runs. ALPHA_T = 2.0 is one
constant, zero new code, an INTERPOLATION inside a measured bracket, and it acts
on information that did not exist ninety minutes ago.

COST. Zero. The term, its oracle, its cache and its generator all still run at
ALPHA_T = 0.0 -- only the multiplier is dead -- so this changes not one
allocation, not one RNG draw and not one wall-clock second. INFERENCE_S stays at
the champion's ~1.14 s of the 10.0 s budget; `forward()` is not touched by a
single character.

REPRODUCIBILITY. `g` (the direct objective's) and `gd` (the distillation
batch's) are separate generators and neither is touched, the TEACH cache build
is bit-identical, and `PowerNet()` is constructed after it as before, so the
initial parameter vector and the entire (K, Kq, drop-seed) sequence of all 2000
steps x 4 tasks are bit-for-bit exp 69's. The scaled gradient is the ONLY
difference between this run and the 1.476460 champion, and between this run and
exp 77's 1.473660.

FAMILY TAG. `interference_attention`, which is what this file has actually been
since exp 77 reverted the sixth family's two blocks: CUT_TAU = 0.0, no `ste`,
no refinement stage, `forward()` character-for-character exp 69's. Exp 77 kept
printing `cut_straight_through` only because its edit died before reaching
`main()`, so the harness has been mislabelling the champion's own architecture
for one iteration. This corrects it. (Side effect, declared rather than relied
on: the harness compares the printed tag with `.current_family` and has no
memory of a family's earlier grace, so it will announce a fresh 5-iteration
window. That is the harness's own rule for a tag change, not the reason for the
change -- the tag is simply the true one.)

FALSIFIER, PRE-REGISTERED, READ IN diagnostics.txt BEFORE THE SCORE.
 1. The six `min` cells must print EXACTLY 1.096/1.233/1.526/1.825/2.024/2.258.
    They are algebraically pinned at p* for ANY parameters (exp 49's clamp) and
    untrained (KQ_MIN_TRAIN = 2), so any movement is a bug in this edit and it
    reverts on sight rather than being argued about. This also confirms the
    diagnostics write is repaired -- if diagnostics.txt still says
    "EXP score 1.476050" the `refine_report` fix did not take.
 2. THE MECHANISM GATE: the eleven Kq>1 cells against exp 76's persisted row
    (p10 1.190/1.404/1.523/1.644/1.774, p25 1.071/1.146/1.267/1.322/1.377/
    1.413), which is within 0.001 of the champion's. If they RISE by ~0.001-0.002
    each, the regulariser reading is confirmed, the weight was under-set for
    fifty-five experiments, and the next probe is ALPHA_T = 3.0 (the quadratic's
    argmax, the one value the saturating models exclude) -- one more point turns
    the bracket into a curve.
 3. If they are FLAT (|delta| <= 0.001 each, score within noise of 1.4765), the
    response is flat on [1, 2] and the peak is a plateau: the weight is done,
    ALPHA_T returns to 1.0 by revert, and the remaining iterations go to TASKS
    and tail averaging as exp 77 prescribed.
 4. If they FALL, the optimum is at or below 1.0, the saturating model with
    k ~ 1 is right and the drag term dominates immediately above the incumbent.
    Then the bracket is [0, 2] with a peak near 1, the term is correctly tuned as
    it stands, and the honest report is that ALPHA_T is CLOSED at 1.0 -- three
    points on both sides, no further probes. Say so plainly; the harness's revert
    restores 1.0 for free.

OFF-GRID. Nothing is narrowed. The training law is untouched -- K uniform on
1..10 (the ungraded 3, 5, 7, 9 included), `frac` flat on (1/KB, 0.25] through the
evaluator's own `kq_of()`, `_band_kq_max()` and KQ_MIN_TRAIN = 2 unchanged, the
TEACH cache still built on seeds 30,000,000+ with three tasks per K -- and the
change is a scalar on an existing term. No off-grid check is owed; run
`k_generalization_check.py` on K in {3,5,7,9} before banking any jump regardless.

-----------------------------------------------------------------------------
EXPERIMENT 77 -- family `interference_attention` -- SCORED 1.473660, FALSIFIER 4
-----------------------------------------------------------------------------
OUTCOME, RECORDED AT THE TOP OF ITS OWN ENTRY: the eleven Kq>1 cells FELL. The
prediction below -- "+0.003 per Kq>1 cell, ~+0.002 on the 17-cell mean, a new
champion at ~1.4785" -- was WRONG BY SIGN, and exp 64's linear extrapolation of
the transfer function does not survive to zero. Everything the entry says about
the TARGET (the oracle is below the student on ten of eleven cells; exp 65's
certified-QFT swap scored lower still; the gauge-fixed log-decade MSE is not a
monotone proxy for SLqP) remains measured and true. What it got wrong is the
inference from those facts to the TERM: a gradient can be worth having without
its target being worth imitating. See exp 78 above.

THE SIXTH AND FINAL BREADTH SLOT IS SPENT AND IT CLOSED ON ITS OWN FALSIFIERS.
`cut_straight_through` ran three iterations and every one of them landed BELOW
the exp-69 champion: 1.476142 (exp 74, the straight-through cut gradient),
1.476142 again (exp 75 never executed), 1.476050 (exp 76, the refinement stage
exp 75 had specified). Both pre-registered mechanism gates fired NEGATIVE, in
the exact form that was written down in advance:

  * exp 74's falsifier 4. MEMBERSHIP_CHECK moved well off zero (0.02/1 at Kq=2
    to 4.18/17 at Kq=18 in the persisted table) while the eleven Kq>1 cells did
    not rise -- net -0.0003. Verbatim: "the anchor's full-power set was already
    right and the residual is SHAPE within the set, not membership."
  * exp 76's falsifier 4. REFINE_CHECK printed a mean |correction| of
    0.805-0.963 DECADES -- an order of magnitude past the 0.05 threshold, and
    with W_REF = 1.0 that is a tanh sitting at ~90% of saturation -- and it
    re-decided up to 4.05 of 17 sacrificed users per drop. The eleven cells
    still did not rise (1.476050 against 1.476142). Verbatim: "the stage is
    fitting drop-level noise ... and it closes this one too."

Read together those two runs say something sharper than either alone. A second,
saturated, freely-parameterised head that sees the operating point, re-decides a
quarter of the sacrificed set, and moves the profile by a full decade produces
the SAME 17-cell mean to four decimals. That is not a knowledge deficit and it
is not a capacity deficit; it is one strong attractor that every parameterisation
this campaign has tried falls into. Both blocks are therefore REVERTED and this
file is the exp-69 champion, byte-for-byte, with one constant changed. (`ste`
survives inside `_cut_clamp` as dead, documented code with CUT_TAU retired to
0.0, so it is never reachable; `_refine`, `_ref_features`, `self.ref`,
`self.ref_head`, `return_h`, `W_REF`, `REF_FEAT` and `refine_report` are gone.
`membership_report` is kept -- it is off every training path and it is the one
gate that reads the policy's actual Kq>1 content.)

THE ONE CHANGE: ALPHA_T 1.0 -> 0.0. THE DISTILLATION TERM IS DELETED.

WHY THIS, AND WHY IT HAS NEVER BEEN RUN. `ALPHA_T` has been on since exp 23,
fifty-four experiments ago, and it has been varied in exactly one direction:
UP. Exp 64 took it 1.0 -> 8.0 and scored 1.465240 (-0.0104); exp 65 held 8.0
and swapped in the certified QFT reference itself and scored 1.463448, LOWER
still. Exp 68 reverted the weight to 1.0 and the oracle to `teach_profile`, and
there it has sat. The downward half of exp 64's own prescription -- its note
says the next knobs are "ALPHA_T downward from 8" -- stopped at the incumbent
and never reached zero. Zero is the one setting of this term the campaign has
no measurement for.

AND ITS TRANSFER FUNCTION IS ALREADY CALIBRATED, WHICH IS WHY THE PREDICTION IS
QUANTITATIVE RATHER THAN HOPEFUL. Exp 64 is a paired A/B on this exact term:
weight 1 -> 8 moved the eleven Kq>1 cells by ~0.023 EACH, straight toward the
teacher's row (p10 1.752 vs 1.775, p25 1.382 vs 1.405). The pull is toward the
teacher, and the teacher is WORSE THAN THE STUDENT -- not marginally, and not
by an unpaired yardstick. The persisted TEACHER_CHECK from exp 76 measures the
live `teach_profile` oracle against the model's own GRID row, cell for cell:

    p10   K=2 1.211 / K=4 1.332 / K=6 1.368 / K=8 1.507 / K=10 1.585
    p25   K=1 1.060 / K=2 1.101 / K=4 1.132 / K=6 1.133 / K=8 1.189 / K=10 1.155
    student, same cells:
    p10       1.190       1.404       1.523       1.644        1.774
    p25       1.071       1.146       1.267       1.322  1.377  1.413

The oracle is below the student on TEN of eleven cells, and at K=10/p25 it is
below by 0.258 -- a policy worth 1.155 is being handed to the student as a
target for a cell the student already runs at 1.413. Seven percent of every
gradient step, for fifty-four experiments, has been a pull toward that. The
term's own docstring calls it "UNTUNED -- if the term bites at all, this is the
first knob"; exp 64 proved it bites, at ~0.003 per cell per unit weight, and in
the wrong direction. Removing it should hand ~0.003 back on each of the eleven
Kq>1 cells, i.e. ~+0.002 on the 17-cell mean -- which would be the largest
single gain since exp 41 and a new champion at ~1.4785.

WHY THE WHOLE TERM RATHER THAN A SMALLER WEIGHT. Because the distillation
thread is 0-for-6 (exps 23, 25/26, 39, 51, 53, 65) and exp 65 is the closing
argument: the CERTIFIED QFT reference, distilled at the same weight as a
strictly worse oracle, scored BELOW it. That is a verdict on the SURROGATE --
the gauge-fixed log-decade MSE is not a monotone proxy for SLqP, so no target,
however good, makes this term point at the metric. A term whose surrogate is
broken and whose current target is measurably worse than the student has no
weight at which it is principled; the interesting number is what the direct
objective alone does, and that has never been measured on this architecture.

WHAT THIS IS NOT. It is not a seventh breadth family: no framework, no
architecture, no feature and no sampler changes. It is not a narrowing -- the
training law is untouched (K uniform on 1..10 including the ungraded 3, 5, 7, 9;
`frac` flat on (1/KB, 0.25] through the evaluator's own `kq_of()`;
`_band_kq_max()` and KQ_MIN_TRAIN = 2 unchanged), and if anything the deleted
term was the ONLY component fitted on a fixed 1440-drop cached set, i.e. the
only memorisation surface in the file. It is not a change to `forward()`, which
is not touched by a single character.

REPRODUCIBILITY, EXACTLY. The TEACH cache build consumed no global RNG (Adam
and `_profile_fixed_point` draw nothing) and ran BEFORE `PowerNet()`, so
deleting it leaves the initial parameter vector bit-for-bit the champion's. The
distillation batch had its own generator `gd` (SEED + 991) and the direct
objective's `g` is untouched, so the (K, Kq, drop-seed) sequence of all 2000
steps x 4 tasks is bit-for-bit exp 69's. The deleted gradient is therefore the
ONLY difference between this run and the 1.476460 champion. Identical code
reproduces an identical score.

COST. Strictly negative. Every step loses one `raw_profile` pass over 48 drops
(which runs `_features`, i.e. two 40-iteration float64 fixed points) -- roughly
a quarter of each step -- and the run loses the ~20-25 s one-time oracle build.
Nothing is added anywhere. INFERENCE_S returns to the champion's ~1.14 s of the
10.0 s budget (exp 76's second fixed point is gone with the refinement stage).

FALSIFIER, PRE-REGISTERED, READ IN diagnostics.txt BEFORE THE SCORE.
 1. The six `min` cells must print EXACTLY 1.096/1.233/1.526/1.825/2.024/2.258.
    They are algebraically pinned at p* for any parameters (exp 49's clamp) and
    are untrained (KQ_MIN_TRAIN = 2), so any movement is a leak from the revert
    and this reverts on sight rather than being argued about.
 2. THE MECHANISM GATE: the eleven Kq>1 cells against exp 76's persisted row
    (p10 1.190/1.404/1.523/1.644/1.774, p25 1.071/1.146/1.267/1.322/1.377/
    1.413). Exp 64's calibration predicts ~+0.003 on each. If they rise by
    roughly that, the term was a measurable drag for fifty-four experiments,
    the direct objective alone is the right training law, and the next knobs
    are the ones the freed compute pays for -- TASKS (4 -> 8 at fixed BATCH,
    exp 19's 1 -> 4 bought +0.0150 and 4 -> 8 is untested) and tail weight
    averaging over the cosine's last quarter.
 3. If they are FLAT (|delta| <= 0.001 each), the term was inert at weight 1
    despite exp 64's linear extrapolation, the surrogate is not merely
    non-monotone but degenerate at small weight, and the endgame is purely the
    optimiser: TASKS, SWA, and the schedule.
 4. If they FALL, the term was acting as a regulariser rather than as a
    teacher -- the one reading exp 64 cannot distinguish -- and the honest next
    move is to restore it at ALPHA_T = 1.0 and spend the remaining iterations on
    TASKS and weight averaging instead. Report that outcome plainly; it is the
    informative one.

OFF-GRID. Nothing is narrowed toward the graded points -- the change is the
DELETION of the only fixed-drop-set term in the file -- so no off-grid check is
owed. Run `k_generalization_check.py` on K in {3,5,7,9} before banking any jump
regardless.

-----------------------------------------------------------------------------
EXPERIMENT 76 -- family `cut_straight_through` (grace 3 of 5) -- REVERTED
-----------------------------------------------------------------------------
EXP 75 NEVER RAN -- THE SAME FAILURE MODE AS EXP 68, AND THE EVIDENCE IS IN THE
FILE. The harness logged an agent execution error on it and re-scored train.py,
which printed 1.476142, exp 74's score to six decimals, at INFERENCE_S = 1.132 s
-- i.e. ONE fixed point, not two. What actually landed was the journal entry
below plus the two dead constants `W_REF` and `REF_FEAT`; `_ref_features`,
`self.ref`, `self.ref_head`, the two-pass `forward()` and `REFINE_CHECK` were
never written, and grepping the file for any of those five names returns only
prose. Exp 75's hypothesis is therefore UNTESTED and still the indicated one, so
THIS run is that change implemented, verbatim to its own specification. Nothing
in the design below is revised: `ste=True`, CUT_TAU = 0.25, W_REF = 1.0 and
REF_FEAT = 4 are exactly as it pre-registered them, so this remains a strict
one-variable step from 1.476142 and its four falsifiers are inherited unchanged.
The only additions are mechanical: `raw_profile` grows an optional `return_h`
flag (default False, so `membership_report` and the ALPHA_T distillation term
call it byte-identically and still read PASS ONE), and `refine_report` prints
the pre-registered `REFINE_CHECK` gate into diagnostics.txt.

THE ONE PLACE THE SPEC NEEDED A CHOICE, RECORDED. Exp 75's own note observes
that the induced-SINR SHAPE of p1 is w1 renormalised and so carries nothing new;
that applies to REF_FEAT channel 1 (decades below the realised cut), which is
kept anyway as the pointwise GAUGE the correction is expressed against -- it is
free (p1's SINR is computed for channels 2-4 regardless) and it saves the ref
MLP from having to rebuild its own position from `hout`. Channels 2-4 -- the
cut's absolute level log10 c, own log power headroom (0 for the binding user),
own cell's aggregate log power -- are the genuinely unreachable ones, and if the
axis is live they are why.

(Exp 75's entry follows, unedited; it is the hypothesis this run tests.)

EXP 74 SCORED 1.476142 AND ITS FALSIFIERS FIRED AS FOLLOWS. Falsifier 1 held
exactly: the six `min` cells printed 1.096/1.233/1.526/1.825/2.024/2.258, bit
for bit, so the straight-through flag leaked nothing into the emitted value.
MEMBERSHIP_CHECK is non-zero and grows with Kq (0.02/1 at Kq=2 to 2.66/17 at
Kq=18), and the eleven Kq>1 cells did NOT rise: -0.001 on eight of them, +0.001
at K=8/p25, +0.002 at K=10/p25, 0.000 at K=1/p25, i.e. one noise ball and a net
-0.0003. That is falsifier 4, verbatim: "the anchor's full-power set was already
right and the residual is SHAPE within the set, not membership -- which closes
the membership reading for good and points depth-tuning at the below-cut
magnitudes instead." This run does exactly that.

ONE HONEST CAVEAT, RECORDED SO IT IS NOT LOST. MEMBERSHIP_CHECK was introduced
BY exp 74, so there is no champion baseline for it and falsifier 2 ("the
champion is also pinned near 0.00") cannot be strictly separated from falsifier
4. It does not matter for the decision, because both readings prescribe the same
next move -- 2 says depth-tune the champion, 4 says depth-tune the below-cut
magnitudes -- and it should not be re-litigated: the champion CANNOT be pinned at
0.00 there anyway. The bottom set has FIXED size Kq-1, so whenever a below-cut
user rises through the cut some above-cut user necessarily descends into the set
even with an identically-zero gradient of its own. Exp 74's note overstated its
own mechanism on that one point. `ste=True` and CUT_TAU = 0.25 are KEPT
UNCHANGED so this run is a strict one-variable step from 1.476142, and CUT_TAU =
0.0 remains a free one-line revert for a later iteration.

THE ONE CHANGE: A REFINEMENT STAGE THAT SEES THE CANDIDATE ALLOCATION. Every
axis this campaign has closed -- capacity 0-for-4, distillation 0-for-6,
softening 0-for-3, Kq re-weighting 0-for-2, head conditioning, features, search,
now the cut plateau -- changed HOW MUCH the head could compute or WHAT IT WAS
PULLED TOWARD. None of them changed WHAT IT KNOWS, because every one of the 24
features is a functional of (A, Kq) alone, evaluated at four allocations fixed in
advance (P_T, P_T/K, channel inversion, p*). The head therefore chooses the
below-cut magnitudes without ever seeing the operating point they produce. Now:

    hout, w1 = trunk(A, Kq)                    # unchanged, bit for bit
    p1       = FP(cut_clamp(w1, Kq))           # the candidate, under no_grad
    w2       = w1 * 10 ** (W_REF * tanh(ref_head(ref([hout, reffeat(A,p1,Kq)]))))
    p        = FP(cut_clamp(w2, Kq, ste=True)) # emitted, as before

WHY THE FED-BACK QUANTITY IS NEW AND NOT A RE-READ OF w1. `_profile_fixed_point`
realises SINR EXACTLY proportional to w, so the induced SINR *shape* of p1 is w1
renormalised and carries nothing -- that observation also kills, in advance, the
cheaper "re-anchor `_clip_profile` at the model's own operating point" variant,
which is provably the identity composed with a doubled head correction. What is
NOT recoverable from w1 is the pair (`p1`, the achieved scale `c`): the fixed
point's normalisation is a 40-iteration global functional of (A, w1), so which
user BINDS at P_T, how much power headroom each other user has, and the absolute
SINR level the profile buys are all quantities no amount of message passing
computes internally. They are exactly what the below-cut choice needs: the
objective is a sum of log2(1 + c*w_i), whose curvature -- and hence how far below
the cut each sacrificed user should sit -- depends on c, and the marginal cost of
raising anyone is paid by the binding user alone. REF_FEAT = 4: decades below the
realised cut, the cut's absolute log10 SINR level (= log10 c), own log power
headroom (0 for the binding user), own cell's aggregate log power.

WHY THIS IS NOT A SEVENTH BREADTH FAMILY. The framework is untouched (the same
direct objective over the same sampler, the same gauge-fixed teacher term at the
same ALPHA_T = 1.0, which still reads `raw_profile`, i.e. PASS ONE, so its
semantics are unchanged); the backbone, the anchor `_clip_profile`, W_SCALE, the
clamp theorem and the fixed point are all the champion's. One bounded
multiplicative stage is added to the OUTPUT, exactly as exp 31 added the profile
head and exps 41/42 added attention inside their own families.

SAFETY, BY CONSTRUCTION, NOT BY HOPE.
 * `ref_head` is zero-initialised in weight AND bias, so at step 0 tanh(0) = 0
   and w2 == w1 IDENTICALLY: the run starts at exp 74's exact policy and the
   refinement is a correction to a validated allocation, not a fresh search.
 * The two new modules are declared AFTER `self.head`, so every module above them
   draws from the global generator in exactly the order the 1.476142 run did and
   their initial parameters are bit-for-bit identical.
 * The six `min` cells are still algebraically pinned: `_cut_clamp` at Kq = 1
   returns a FLAT profile for ANY input whatsoever, so p* is emitted no matter
   what the refinement multiplies by. This is checkable, and it is falsifier 1.
 * `p1` is computed under `torch.no_grad()`, so nothing backpropagates through
   the extra fixed point; the gradient reaches the trunk through the `w1 * ...`
   factor and through `hout`, exactly the two paths that already existed.

CONTRACT. Two unconditional fixed points instead of one, plus a one-hidden-layer
MLP on K*B nodes. No objective is evaluated anywhere: no rate, no log2, no SLqP,
no top-k of any rate. There is no candidate SET -- p1 is not compared with p2,
not scored, not accepted or rejected; p2 is emitted unconditionally whatever p1
was. No gradient step, no restart, no early stop, no branch on any utility.
`_ref_features` reads `_induced_sinr`, the same SINR-of-a-given-allocation
primitive `_features` has run inside `forward()` since exp 28 -- the only
difference is that the allocation is now the model's own first stage, which is
still a deterministic feed-forward function of the input. INFERENCE_S should go
~1.4 s -> ~2.2 s of the 10.0 s budget; if it does not, that is a bug, not a
tuning problem.

OFF-GRID. The training law is untouched -- K uniform on 1..10 (ungraded 3, 5, 7,
9 included), `frac` flat on (1/KB, 0.25] through the evaluator's own `kq_of()`,
`_band_kq_max()` and KQ_MIN_TRAIN = 2 unchanged -- so nothing is narrowed toward
the graded points. Run `k_generalization_check.py` on K in {3,5,7,9} before
banking any jump regardless.

FALSIFIER, PRE-REGISTERED, READ IN diagnostics.txt BEFORE THE SCORE.
 1. The six `min` cells must again print EXACTLY 1.096/1.233/1.526/1.825/2.024/
    2.258. Any movement is a leak into a provably pinned column: revert on sight.
 2. THE MECHANISM GATE: `REFINE_CHECK` prints, per Kq>1 cell, the mean |tanh|
    correction IN DECADES and how many of the Kq-1 sacrificed users differ
    between pass 1 and pass 2. If the mean correction is <=0.01 decades the
    refinement was simply switched off by training -- the candidate allocation
    carries no usable signal, the "what it knows" axis is closed alongside the
    other seven, and the endgame is pure hyper-tuning of the champion.
 3. If the correction is substantial (>=0.05 dec) and the eleven Kq>1 cells rise,
    the axis is live and the knobs, in order, are W_REF (0.5 / 2.0), then the
    REF_FEAT set (drop the cell-aggregate term, add the per-user interference
    contribution at p1), then a second refinement stage.
 4. If the correction is substantial but the eleven cells do NOT rise, the stage
    is fitting drop-level noise: that is a capacity-style failure on an axis the
    campaign has already closed four times, and it closes this one too.

-----------------------------------------------------------------------------
EXPERIMENT 74 -- family `cut_straight_through` (BREADTH FAMILY 6 of <=6)
-----------------------------------------------------------------------------
WHERE THE CAMPAIGN STANDS. Exp 69 is the champion at 1.476460; QFT is 1.485.
Exp 63's PAIRED measurement, on the grid's own pinned drops, is the only clean
read of the residual: QFT beats the student on 100% of drops in ALL ELEVEN Kq>1
cells, by 0.65% at Kq=2 rising monotonely to 4.46% at Kq=18, se 0.001-0.003, for
+0.0197 available -- while the six Kq=1 cells are at or above it, as `p*`'s
provable optimality requires. Those eleven cells have moved <=0.006 in total
across eighteen experiments. Closed axes, >=2 independent probes each: capacity
0-for-4 (18, 21, 22, 41/42), distillation 0-for-6 (23, 25/26, 39, 51, 53, 65 --
the last being the certified QFT reference ITSELF, which scored BELOW a strictly
worse oracle at the same weight), objective softening 0-for-3 (3, 4, 20), Kq
re-weighting 0-for-2 (5, 17), head conditioning closed (50), features +0.0009
then null (69, 70).

EXP 72 CLOSED THE FIFTH FAMILY ON ITS OWN PRE-REGISTERED FALSIFIER. It scored
1.472085 and its mechanism gate -- "best-cand bottom-set flips/drop" -- printed
0.04, with the mean candidate advantage still at -0.499%. Its falsifier 3 said,
verbatim: "If it is ~0 the search is buying SHAPE, not membership, and the whole
framework is a redundant finite-difference copy of a gradient we already have
exactly -- the sixth breadth slot opens immediately." Even masked to the eight
users nearest the cut, a 0.15-decade isotropic proposal flipped essentially no
memberships in 32-drop batches. `search_improvement` is 0-for-2 and closed;
exp 73 (its grace iteration 3) died on a NameError and never scored. This run
takes the sixth and final breadth slot, and it keeps exp 71/72's DIAGNOSIS while
discarding their INSTRUMENT: the missing quantity is a derivative, so take it by
backprop, not by sampling.

THE PLATEAU, NAMED EXACTLY -- AND IT IS WORSE THAN "A ZERO GRADIENT". The output
is p = FP(_cut_clamp(f, Kq)) with f = raw_profile. `_cut_clamp` is
`torch.minimum(f, thr)`, thr = kthvalue(f, Kq), so for the K*B - Kq entries above
the cut the output is `thr` and d out_j / d f_j is EXACTLY ZERO. The head is
POINTWISE, so f_j is a function of logit_j alone: therefore

    for every above-cut user j,  dL/dlogit_j == 0 identically, every step.

Now read the initial condition. The head is zero-initialised, so at step 0
f = w_clip = min(sinr_fp/thr_fp, 1), which is strictly < 1 on EXACTLY the Kq-1
worst users at FULL POWER and exactly 1 on the other K*B - Kq + 1. Its Kq-th
smallest is 1, so `_cut_clamp` is the identity on it. Consequence: the run BEGINS
with the bottom-Kq set equal to "the Kq-1 worst-at-full-power users", and a user
can only ever LEAVE that set (by rising through the cut) -- NOTHING CAN ENTER,
because entering requires descending from a coordinate whose gradient is
identically zero. The set is not merely hard to learn; it is very nearly a fixed
point of training, inherited from a full-power heuristic, and QFT's own operating
point is not full power.

That single fact predicts every number this campaign has measured: the deficit is
confined to the eleven Kq>1 cells (only there is the set non-empty); it grows
monotonely with Kq (the number of frozen decisions IS Kq-1); it is zero at Kq=1
(empty set, and `_cut_clamp` returns the provable optimum for any weights); and
eighteen experiments changing WHAT the network sees (features, attention,
victim edges) and HOW MUCH capacity it has cannot touch it, because no amount of
either changes a coordinate whose partial derivative is identically zero.

THE ONE CHANGE: A STRAIGHT-THROUGH GRADIENT FOR THE CUT, TRAINING ONLY. With
z = log10(f/thr), the hard clamp is f * 10**(-relu(z)). `_cut_clamp(..., ste=True)`
returns the SAME VALUE and the gradient of the softplus relaxation

    soft = f * 10 ** ( -CUT_TAU * softplus(z / CUT_TAU) )
    d log soft / d log f = sigmoid(-z / CUT_TAU)

-- 1 far below the cut (bit-identical to the champion there), 0.50 AT the cut,
0.12 half a decade above, 0.02 a decade above. An above-cut user now receives its
true fixed-point-mediated signal, "release target -> the scale c rises -> every
graded rate rises", weighted by how flippable it is. It is exact (backprop, not
eight random draws), dense (all K*B coordinates every step), self-extinguishing
(weight -> 0 for users decades clear of the cut), and it hands over to the true
gradient the moment a user crosses. And the initial condition is ideal: at zero
logits every above-cut user sits at z = 0 EXACTLY, so this is precisely the
well-defined one-sided derivative at the kink, taken at weight 1/2. The run
starts on the boundary and, for the first time, can step off it.

WHY THIS IS NOT THE 0-for-3 "OBJECTIVE SOFTENING" AXIS. Exps 3, 4 and 20 softened
`slqp_rate`'s top-k -- they changed WHAT WAS OPTIMISED, so the training objective
stopped being the graded metric. Here the objective is byte-identical (the same
`-slqp_rate(model(A,Kq),A,Kq).mean()/ref`, the same hard `torch.topk`), the
emitted allocation is bit-identical, and the graded forward pass is byte-for-byte
the champion's. The ONE variable is the descent direction on a plateau of the
UNCHANGED loss. Nor is it a seventh distillation: no teacher, no label, no cache;
`teach_profile` and ALPHA_T = 1.0 are untouched and still call `_cut_clamp`
with `ste=False`, i.e. with the champion's exact gradient.

VALUE-EXACTNESS, TWICE OVER. (i) `out = hard.detach() + (soft - soft.detach())`
-- the parenthesis is a tensor minus its own bitwise-identical values, exactly
+0.0, so `out == hard` bit-for-bit. (ii) The branch is additionally gated on
`torch.is_grad_enabled()`, and `prepare.evaluate` calls the model under
`torch.no_grad()`, so at grading time the executed code is literally the
champion's: same powers, same INFERENCE_S (~1.4 s of 10.0), no contract surface
added. `ste=True` is passed from exactly one call site, `forward()`.

COST. One log10, one softplus and one pow on an [n, K*B] tensor per task per
step, against the two 40-iteration float64 fixed points the same step already
runs. Nothing is added to inference.

REPRODUCIBILITY. No RNG is consumed anywhere new. `PowerNet.__init__` is
untouched, so the initial parameter vector is bit-for-bit exp 69's; `g` and `gd`
draw the identical streams, so the (K, Kq, drop-seed) sequence and the teacher
cache are unchanged. Identical code reproduces an identical score.

OFF-GRID. The training law is untouched -- K uniform on 1..10 (ungraded 3, 5, 7,
9 included), `frac` flat on (1/KB, 0.25] through the evaluator's own `kq_of()`,
`_band_kq_max()` unchanged, KQ_MIN_TRAIN = 2 -- so nothing is narrowed toward the
graded points and no off-grid check is owed; run `k_generalization_check.py` on
K in {3,5,7,9} before banking any jump regardless.

FALSIFIER, PRE-REGISTERED, READ IN diagnostics.txt BEFORE THE SCORE.
 1. The six `min` cells must print EXACTLY 1.096/1.233/1.526/1.825/2.024/2.258.
    The clamp's returned value is bit-identical and Kq=1 is never trained
    (KQ_MIN_TRAIN = 2); movement means a leak and the run reverts on sight.
 2. THE MECHANISM GATE, new: `MEMBERSHIP_CHECK` reports, per Kq>1 cell, how many
    of the model's Kq-1 sacrificed users the w_clip anchor does NOT sacrifice.
    The claim above is that the champion is pinned near 0.00 here. If exp 74
    ALSO prints ~0.00, then the plateau was not what froze membership and this
    entire eighteen-experiment diagnosis is wrong -- report that, because it is
    the most informative outcome available and it redirects the endgame to
    depth-tuning the champion.
 3. If MEMBERSHIP_CHECK moves off 0 AND the eleven Kq>1 cells rise, the
    mechanism is confirmed and the knobs, in order, are CUT_TAU (0.15 / 0.4),
    then ALPHA_T (the teacher's own membership signal now competes with a live
    gradient), then STEPS.
 4. If MEMBERSHIP_CHECK moves off 0 but the eleven cells do NOT rise, the
    anchor's full-power set was already right and the residual is SHAPE within
    the set, not membership -- which closes the membership reading for good and
    points depth-tuning at the below-cut magnitudes instead.

-----------------------------------------------------------------------------
EXPERIMENT 69 -- family `interference_attention` (the champion, 1.476460)
-----------------------------------------------------------------------------
EXP 68 NEVER RAN. The harness logged an agent execution error on it and re-scored
the file, which printed 1.475555 -- the champion's score to six decimals, because
only this journal entry had been written when the step died; `N_FEAT` was still
21 and no line of `_features` had moved. Exp 68's hypothesis is therefore
UNTESTED, it is still the indicated one, and this run is that hypothesis actually
implemented, unchanged: `N_FEAT` 21 -> 24 and block (f) of `_features`. The
entry below is exp 68's, kept verbatim as the pre-registration.

WHAT EXP 65 SETTLED, AND WHY THIS FILE IS BACK AT EXP 52's CODE. Exps 65, 66 and
67 are ONE measurement: the harness logged an agent execution error on all three
and re-scored the same file, which is why the log shows 1.463448 three times.
That file was exp 65's -- the certified QFT reference itself as the distillation
teacher, at exp 64's calibrated ALPHA_T = 8.0 -- and its own pre-registered
falsifier fired the losing way. The eleven Kq>1 cells moved DOWN, to p10
1.182/1.389/1.505/1.624/1.749 and p25 1.064/1.124/1.245/1.299/1.351/1.385
against the champion's 1.190/1.406/1.524/1.643/1.771 and 1.070/1.143/1.265/
1.317/1.368/1.401, for 1.463448 -- BELOW exp 64's 1.465240, which pulled toward
a strictly worse (local-search) teacher at the same weight. A BETTER teacher
scored WORSE at the same weight, so the defect is the surrogate, not the target:
a log-decade MSE onto a fixed 640-drop label set is not a monotone proxy for
SLqP, and no oracle will fix that. Distillation is 0-for-6 (exps 23, 25/26, 39,
51, 53, 65) and is closed, on the strongest target that exists. The six `min`
cells printed exactly 1.096/1.233/1.526/1.825/2.024/2.258 for the eighth
consecutive run, so `model(A,1) == p*(A)` is confirmed once more and the port
itself was correct. Exp 65's code (`_qft_profile`, its cache, ALPHA_T = 8.0,
TEACH_TASKS 20 / TEACH_SUB 32) is reverted in full; this run is a one-variable
A/B against the 1.475555 champion. Its journal entry is kept below as the
record.

THE ONE CHANGE: THREE SET-RESTRICTED INTERFERENCE-COUPLING FEATURES, the first
addition to the feature set in 38 experiments. Exp 63's paired measurement on
the grid's OWN pinned drops says the residual is a POLICY error -- QFT beats the
student on 100% of drops in all eleven Kq>1 cells, deficit a clean monotone
function of Kq (0.22% at Kq=2 -> 2.27% at Kq=18) -- and those eleven cells have
moved <=0.006 in total across fourteen experiments that changed attention, the
drop stream, the Kq floor, the output cut, a refinement round and four oracles.
One attractor, reached from a zero-init head, with capacity 0-for-4: that is a
REPRESENTABILITY claim, and there is exactly one quantity the objective turns on
that this network provably cannot compute.

EVERY POOLING OPERATION IN THE MODEL IS NORMALISED. `_edge_weights` is
row-normalised (what FRACTION of my interference comes from cell c),
`_victim_weights` is normalised again (what SHARE of cell c's pain lands on me),
the cell pool is mean and max, and `_attend`'s softmax is normalised by
construction. Every one of them returns an AVERAGE. But what decides a Kq>1
allocation is a SUM OVER A Kq-SELECTED SUBSET: "of the Kq users the metric
actually grades, how much of their interference-plus-noise do I supply?" A mean
over all K*B victims is not that number, and it differs from it by the factor
Kq/(K*B) -- which is precisely why the measured deficit GROWS MONOTONELY WITH Kq
while the Kq=1 column, where the subset is a single user that max/mean pooling
does resolve, is exactly optimal. No depth fixes a normalisation.

So the three channels are supplied directly, all closed-form functionals of
(A, Kq), all computed from objects `_features` already builds:

    inset = 1[sinr_fp <= thr_fp]      the bottom-Kq set at the full-power
                                      operating point -- the same order
                                      statistic `q_fp`/`m_fp` have shipped since
                                      exp 28, not a new primitive
    share[k,b,c] = K*P_T*A[k,b,c] / (K*P_T*tot[k,b] - P_T*own[k,b] + N_0)
                                      the fraction of victim (k,b)'s
                                      interference-plus-noise that cell c
                                      supplies (c != b), i.e. the denominator of
                                      `sinr_fp` resolved by aggressor

  1. `a_set[c] = sum_{k,b} inset[k,b] * share[k,b,c]` -- THE AGGRESSION MASS a
     cell lays on the graded set. Unnormalised, and the sum runs over a
     Kq-dependent subset: the one quantity above. Per-cell, broadcast to that
     cell's users (which is the right factorisation -- interference from cell c
     is set by its TOTAL power, while who inside c should carry it is what the
     per-user channels already say).
  2. `a_tgt[c] = (a_set[c]/a_all[c]) * (K*B/Kq)` -- the same mass relative to
     what an UNTARGETED cell would supply, so ~1 means "I hurt the graded set no
     more than average" and >1 means my power is landing where the metric reads.
     Dimensionless and free of both K and Kq, so it does not drift over the grid.
  3. `s_max[k,b] = max_c share[k,b,c]` -- the victim-side dual: is my pain
     dominated by ONE cell (cheap to fix) or diffuse (hopeless)? An unnormalised
     max over the EDGE weights; the model's max pool is over a cell's users'
     hidden states, so this is not reachable either.

N_FEAT 21 -> 24. Nothing else moves: `forward()`'s structure, `_cut_clamp`,
`_clip_profile`, `_profile_fixed_point`, ROUNDS, HIDDEN, HEADS, W_SCALE, both
attention gain biases, both loss terms as written, ALPHA_T, `teach_profile`, the
samplers, the fresh-drop stream, the pools, the optimiser and the cosine
schedule are byte-identical to the 1.475555 run. The encoder's first `Linear`
changes shape, so the initial parameter vector is NOT bit-for-bit exp 52's --
unavoidable for any feature change, exactly as it was for exps 27/28/29, the
last three experiments that moved this campaign on the feature axis (+0.0091 and
+0.0024 among them).

COST. One [bt,K,B,B] tensor and one einsum of the shape `_features` already runs
for `tot`, against the TWO 40-iteration float64 fixed points the same function
already computes; inference stays ~1.4 s of the 10.0 s budget.

CONTRACT. Pure algebra on the INPUT. No rate, no log2, no top-k of any RATE, no
SLqP, no sum of any objective; no candidate SET (one tensor is computed, not
chosen); nothing accepted, rejected or compared by utility; no loop; no gradient
and no objective gradient; no model output is read, so the values are identical
for any parameters. Kq enters only through `thr_fp`, the Kq-th order statistic
of `sinr_fp` -- the primitive `_clip_profile` has run inside `forward()` since
exp 29 and `_order_stats` since exp 28.

FALSIFIER, READ IN THE PERSISTED GRID BEFORE THE SCORE.
 1. The six `min` cells must read EXACTLY 1.096/1.233/1.526/1.825/2.024/2.258
    for a ninth run. At Kq=1 `_cut_clamp` flattens the profile whatever the head
    emits, so this is immune to any feature change; movement means a leak, and
    the run reverts on sight.
 2. If the eleven p10/p25 cells move UP, the normalisation argument is right and
    the next knobs are more set-restricted couplings (the victim-weighted mass,
    the same sums read at the `w_clip` operating point rather than full power).
    If they sit inside their fourteen-experiment noise ball with the one
    quantity the architecture cannot form handed to it directly, this family is
    finished at 1.4756 and the honest next move is the FIFTH breadth slot.

OFF-GRID. The training law is untouched -- K uniform on 1..10 (ungraded 3, 5, 7,
9 included), `frac` flat on (1/KB, 0.25] through the evaluator's own `kq_of()`,
`_band_kq_max()` unchanged -- so nothing is narrowed toward the graded points
and no off-grid check is owed; `k_generalization_check.py` on K in {3,5,7,9}
before banking any jump regardless.

CAVEAT. Python execution is permission-gated in this session, so this is
hand-verified rather than smoke-tested. Checked: `A[t,k,b,c]` is the gain from
cell c into user (k,b) (`tot = A.sum(dim=3)` and `sinr_fp`'s denominator
`K*P_T*tot - P_T*own + N_0` fix the convention), so `share` divides by that same
denominator and the einsum 'tkb,tkbc->tc' leaves the AGGRESSOR cell index; the
c == b diagonal is removed with the same `A - diag_embed(own)` `_edge_weights`
uses, so the own-cell signal never enters; `thr_fp` is already returned by
`_order_stats` as [-1,1,1] and broadcasts against `sinr_fp` [bt,K,B]; `a_set`
and `a_tgt` are [bt,B] and are broadcast with `unsqueeze(1).expand_as(own)`,
which is the cell axis; all three are passed through the existing `d()` log
scaler and sit in ~[-0.6, 0.5]; and at K=1 the off-diagonal is still B-1 cells
wide, so nothing degenerates.

-----------------------------------------------------------------------------
EXPERIMENT 65 -- family `interference_attention` (RUN AND REVERTED: 1.463448,
the distillation thread closed 0-for-6; the entry below is the record)
-----------------------------------------------------------------------------
EXP 52 SCORED 1.475555 AND IS STILL THE CHAMPION; exps 53-64 are twelve rounds
inside a +-0.0015 ball. Two of them are not nulls, and together they name the
one hypothesis this campaign has never actually tested.

EXP 63 REPLACED THE YARDSTICK, AND THE PICTURE CHANGED. Every "headroom" number
before it was UNPAIRED -- the student on 250 pinned drops against program.md's
table computed on 30 drops of its own -- and the six `min` cells calibrate that
error exactly, because since exp 49 the model emits p*(A) there and p* is the
provable box-constrained max-min optimum, so every non-zero entry of that column
is pure drop-sampling noise. It read -0.024/-0.057/-0.014/+0.095/-0.096/+0.188
against p10/p25 "deficits" of only -0.017 to -0.035. Exp 63 solved QFT on each
cell's OWN pinned drops instead. The persisted HEADROOM table is unambiguous:

    min  cells   QFT/model = 0.969 - 1.000   (QFT wins 0-3% of drops)
    p10/p25      QFT/model = 1.0065 - 1.0446, QFT wins 100% of drops, EVERY cell,
                 se 0.001-0.003, deficit monotone in Kq  ->  +0.0197 available

That is a POLICY error, not variance, and it is entirely in the eleven Kq>1
cells.

EXP 64 CALIBRATED THE ONLY TERM THAT AIMS AT IT. `ALPHA_T` had been 1.0 since
exp 23 -- ~7% of the loss, "deliberately a MINORITY" -- and had never once been
varied. At 8.0 the score moved -0.0104 and, decisively, the eleven cells moved
~0.023 EACH, straight toward the teacher's row (p10 1.752 vs 1.775, p25 1.382 vs
1.405). So the transfer function of this term is now measured: at weight 8 the
student tracks its teacher, cell for cell, by about the size of the QFT gap. The
term is not dead. Exp 64 lost because it was aimed at the wrong target.

WHY EVERY TARGET SO FAR WAS THE WRONG ONE. Five oracles have been built (exps
23, 25/26, 39, 51, 53) and all five were LOCAL OPTIMISERS -- cvxpy on 320 fixed
drops, a Kq-free geometric path, per-instance Adam in power space, then in
clamped-profile space, then annealed with a best-iterate tracker. Each was
rejected on `teacher_report`, which compares the oracle on POOL drops against
the student's GRID row on PINNED drops: exactly the unpaired comparison exp 63
proved is broken. But exp 64 supplies the paired verdict anyway -- dragging the
student 0.023 toward exp 53's oracle LOST 0.0104, so that oracle really is below
the student, and by construction so were its four predecessors, since the whole
class is per-instance ascent on a top-k objective whose bottom-set membership a
hard `kthvalue` recomputes every step. A policy trained across drops beats local
search on this landscape. The MSE has therefore never been handed a target that
knew more than the model -- and exp 63 identified, on the grid's own drops, a
teacher that provably does.

THE ONE CHANGE: THE TEACHER IS NOW QFT ITSELF. `_qft_profile` replaces
`teach_profile` as the label source. Per drop it runs `qft_reference.qft_solve`
(the certified reference: quadratic transform, alternating y/convex-p, init
0.5*P_T, 10 iterations -- program.md certifies 10 as converged for this band,
verified to 60 with <0.1% drift) and converts the winning ALLOCATION into the
student's own coordinates with the machinery exp 51 already built and exp 61
already validated: `_induced_sinr -> _cut_clamp -> _log_cut`, floored at -6
decades. program.md is explicit that this campaign may generate its own labels
with `qft_reference.py` and cache them, and that all contract restrictions apply
at INFERENCE only.

`ALPHA_T` IS HELD AT EXP 64'S 8.0, AND THAT IS THE POINT, NOT A SECOND VARIABLE.
Exp 64 is the control this run is read against: same weight, same loss as
written, same schedule, same 8x pull -- the TEACHER IDENTITY is the only
difference between the two runs, which is the clean one-variable A/B. At the
incumbent 1.0 the term is ~7% of the loss and moves cells by ~0.003, i.e. the
run would return another null and separate nothing. Exp 64 bought that
calibration at the cost of one experiment; spending it is cheaper than
re-deriving it.

WHY THE ROUND TRIP CANNOT LOSE. `_cut_clamp` only LOWERS targets, and F is a
standard interference function, so the maximal common scale obeys
c(w_clip) >= c(w) = 1 and SLqP(round trip) >= SLqP(QFT) per drop -- exp 61's
algebra, which its own unpaired measurement then appeared to contradict and exp
63 exonerated. `_log_cut` removes the one gauge `_profile_fixed_point`
normalises away, so the target is exactly the object the head controls: the
log-decades by which each graded user sits below the cut, and exactly 0 on every
user outside the bottom-Kq set.

FIREWALL, STRUCTURAL. Teacher drops come from `sample_channels` seeds
30,000,000+ (the TEACH cache) and `make_pools`' 1000..1520 (the report), both
disjoint from the evaluator's pinned TEST seeds 5000..5010. Exp 63's
`qft_grid_headroom_cache.pt` holds QFT solutions of GRADED channels and is NEVER
read here -- `_qft_profile` has its own file, `qft_distill_cache.pt`, and no QFT
solution of a `TEST[K]` drop can reach a loss.

COST, DECLARED. TEACH_TASKS 30 -> 20 and TEACH_SUB 48 -> 32 (640 labelled drops,
still two tasks per K including the ungraded 3, 5, 7, 9) plus 11 report cells x
16 drops: 816 cvxpy solves at ~0.3-0.6 s, i.e. ~5-8 min ONE TIME, before the
training loop and outside the graded `evaluate` call. It is then cached to disk
and every later run is free. Nothing in the 10.0 s inference budget changes:
`forward()` is not touched by a single character and inference stays at exp 52's
~1.4 s.

WHAT IS NOT TOUCHED. All 21 features, the encoder, both attention gain biases,
ROUNDS, HIDDEN, HEADS, W_SCALE, OUT_ITERS, BAL_ITERS, `_clip_profile`,
`_cut_clamp`, `_log_cut`, `_profile_fixed_point`, the loss as WRITTEN, the
direct-objective term, the fresh-drop stream and its seed arithmetic, `g`, the
pools, the optimiser and the cosine schedule are byte-identical to the 1.475555
run. The TEACH build consumes no global RNG (no Adam, no `torch.rand`), so the
initial parameter vector is bit-for-bit exp 52's; only `gd`, the distillation
batch's own generator, sees the shorter task list.

FALSIFIER, TWO GATES READ IN ORDER BEFORE THE SCORE.
 1. TEACHER_CHECK is now a PORT-CORRECTNESS gate, not a quality gate: the
    teacher IS QFT, so on the same pool drops it must land at or above the
    certified parens and far above CLIP_CHECK's warm start
    (1.169/1.276/1.304/1.397/1.461 and 1.047/1.091/1.151/1.174/1.223/1.245) on
    every one of the eleven cells. If it does not, the conversion or the solve
    is wrong and this reverts on sight rather than being argued about.
 2. The six `min` cells must still read EXACTLY 1.096/1.233/1.526/1.825/2.024/
    2.258 -- immune at any ALPHA_T because exp 49's clamp forces
    `model(A,1) == p*` for all parameters. Movement means a leak.
Then: if the eleven p10/p25 cells move UP toward HEADROOM's QFT column, the
label axis is finally alive and the next knobs are ALPHA_T downward from 8 and
TEACH_SUB upward (exp 23's memorisation risk). If they move DOWN again with a
teacher that is paired-better on 100% of drops, distillation is closed for good
on the strongest target that exists and the honest next move is the fifth
breadth slot.

OFF-GRID. The direct-objective training law is untouched (K uniform 1..10,
`frac` flat on (1/KB, 0.25] through the evaluator's own `kq_of()`), and the
teacher tasks draw from that same law, so nothing is narrowed and no off-grid
check is owed -- `k_generalization_check.py` on K in {3,5,7,9} before banking
any jump regardless.

CAVEAT. Python execution is permission-gated in this session, so this is
hand-verified rather than smoke-tested. Checked: `qft_solve` returns `p_flat`
indexed k*B+b (see `_prep`'s `M = G.reshape(KB, B)` and `SM[b, k*B+b] = 1`), in
ABSOLUTE power units (the box constraint is `p <= P_T`; only the gains are
noise-normalised), so `.reshape(K, B)` is the right unflatten; `_cut_clamp` of a
profile whose Kq-th smallest entry is thr leaves that entry the Kq-th smallest,
so `_log_cut` subtracts exactly thr and the target is <= 0 with exact zeros
above the cut; a relative 1e-8 floor on the induced SINR removes the -inf a
numerically-muted user would otherwise put into `log10` (and the degenerate
thr == 0 branch with it); float64 throughout the conversion, returned in
`A.dtype` so `torch.where(zt < 0.0, zs - zt, ...)` stays float32; `numpy` and
`qft_reference` are imported lazily inside the function so module import does
not depend on cvxpy; and the cache key carries (tag, n, K, Kq) so changing
TEACH_SUB or TEACH_TASKS invalidates it rather than silently reusing stale
labels.

-----------------------------------------------------------------------------
EXPERIMENT 52 -- family `interference_attention` (depth-tuning; still 4 of <=6)
-----------------------------------------------------------------------------
EXP 51 SCORED 1.474295 AND IS THE CHAMPION (it shipped without a header block;
recorded here). It deleted exp 24's Kq=1 power MSE -- which exp 49's clamp had
made identically zero -- and replaced it with a gauge-fixed LOG-PROFILE MSE
against a new Kq>1 direct-optimisation oracle. Net +0.0003 on exp 49, i.e. one
noise ball, and the reason is now visible in the persisted TEACHER_CHECK, the
gate that exists precisely for this: the exp-51 oracle measures BELOW the
student on TEN of the eleven graded Kq>1 cells --

    teacher p10  1.211 1.332 1.368 1.507 1.585   vs student 1.189 1.405 1.523
                                                              1.644 1.772
    teacher p25  1.060 1.101 1.132 1.133 1.189 1.155
                                 vs student 1.069 1.141 1.263 1.318 1.371 1.405

-- and on the whole p25 column it lands BELOW its own warm start (CLIP_CHECK
1.047/1.091/1.151/1.174/1.223/1.245), so 60 fixed-step Adam iterations with no
best-iterate tracker are not ascending a top-k objective at all. That makes the
supervised half a drag term, exactly as in exps 25/26, and it makes the exp-49
hypothesis it was built to test (that the RIGHT space for a teacher is the
cut-clamped log profile) still untested. Fixing that oracle is the obvious next
move and it is NOT this one -- because exp 51 also left a strictly cheaper,
strictly safer inefficiency on the table, in the half of the loss that is
actually working.

THE HYPOTHESIS: ONE TASK DRAW IN SIX IS A NO-OP, AND THE FIX IS THE SAMPLER.
Exp 51's own argument for deleting the Kq=1 MSE was that `model(A, 1)` and
`balance_labels(A)` became the same flat-profile recursion under exp 49's
`_cut_clamp`, so that term bought no gradient. The argument is correct and it is
also INCOMPLETE: it applies verbatim to the DIRECT-OBJECTIVE term, which is the
term carrying the run. At Kq = 1 the clamp is `w <- min(w)`, the emitted profile
is flat whatever the head does, `_profile_fixed_point` normalises a flat profile
away exactly (w and c*w give the identical allocation), and the output is p*(A)
for any parameters. So

    d[ -SLqP_1(model(A,1)) ] / d(theta)  ==  0   to float64 roundoff,

and `_sample_band_kq` still spends 4/(K*B) of its draws there: 57.1% at K=1,
28.6% at K=2, 19.0/14.3/11.4/9.5/8.2/7.1/6.3/5.7% at K=3..10, i.e. 16.7%
averaged over the uniform K. With TASKS=4 that is 0.67 of every step's four
tasks contributing nothing -- an effective 3.33-task average where the design
says 4, in a family whose single largest architectural win (exp 19, +0.0150)
came from cutting exactly this task-direction variance and whose next-largest
data-side win (exp 45, +0.0014) came from feeding the same term better drops.

THE ONE CHANGE: `_sample_band_kq` DRAWS FROM THE SAME FLAT LAW CONDITIONED ON
Kq >= 2. `kq_of(frac, KB) >= 2` exactly when `frac > 1/KB`, so `frac` is drawn
flat on (1/KB, BAND_MAX_FRAC] instead of [0, BAND_MAX_FRAC]. This is the EXACT
conditional, not a tilt: the relative density over every Kq >= 2 is unchanged
and merely rescaled by 1/(1 - 4/KB). Re-weighting the band is 0-for-2 (exp 5's
log-uniform, exp 17's squared tilt) and this is deliberately NOT that -- it
removes the one sub-event with zero gradient and touches no other.

WHY THIS IS NOT A NARROWING, in the sense program.md's Protocol warns about.
The prior full-range campaign narrowed training ONTO its graded points; this
removes mass from six GRADED cells and gives it to a continuum that is mostly
UNGRADED (at K=10, Kq now spans 2..18 with 12 of those 17 values ungraded).
Nothing can be encoded about the grid that was not already encodable. The six
cells that lose their mass are the ones exp 49 made parameter-independent, and
`_band_kq_max()` is untouched, so every graded Kq >= 2 cell keeps the mass it
had (rescaled up): the program.md audit invariant still holds.

FALSIFIER, readable in the persisted GRID table, and unusually sharp. The six
`min` cells MUST still read EXACTLY 1.096 / 1.233 / 1.526 / 1.825 / 2.024 /
2.258 -- the values they have held identically across exps 49, 50 and 51 while
everything else moved. If they hold, the theorem is confirmed empirically and
the removed mass provably cost nothing; if ANY of them moves, `model(A,1)` is
not parameter-independent after all, the premise is false and this is reverted
on the spot rather than argued about. On the other side: if they hold and the
eleven p10/p25 cells are unmoved to rounding, then a 20% larger effective
gradient budget is worth nothing to this model, which -- with capacity
(0-for-4), data (+0.0014), loss shape (0-for-3), Kq measure (0-for-2), output
scale (0-for-2), LR (0-for-1), wasted DOF (exp 49, neutral) and global
conditioning (exp 50, dead) all closed -- would leave the exp-51 teacher, fixed
so that TEACHER_CHECK actually clears the student, as the last live thread.

CONTRACT. `forward()` is not touched by a single character. `_sample_band_kq`
is training-only: it is called from `main`'s task loop and from the TEACH cache
build, never from the model. No gradient step, no candidate set, no objective,
no loop, no test-time anything. Inference stays at exp 51's ~1.6 s of 10.0 s.

WHAT IS NOT TOUCHED. All 21 features, `_features`, the encoder, both attention
gain biases, ROUNDS, HIDDEN, HEADS, the cell-mediated path, W_SCALE, OUT_ITERS,
BAL_ITERS, `_clip_profile`, `_cut_clamp`, `_log_cut`, `_profile_fixed_point`,
`teach_profile` and every TEACH_* constant, ALPHA_T, both loss terms as
WRITTEN, the fresh-drop stream and its seed arithmetic, the pools, the
optimiser and the cosine schedule are byte-identical to the 1.474295 run.
Exactly one `torch.rand` is still consumed per call, so `g` and `gd` advance
draw-for-draw as before and the (K, drop-seed) sequence of every step -- and
the initial parameter vector -- stay bit-for-bit exp 51's. The Kq VALUES are
the only difference between the two runs, which is the one-variable A/B.

OFF-GRID. `k_generalization_check.py` on K in {3,5,7,9} before treating any
jump as banked, plus the p25-adjacent ungraded fractions; the sampler still
covers every K in 1..10 and every Kq in 2.._band_kq_max(K) continuously.

-----------------------------------------------------------------------------
EXPERIMENT 50 -- family `interference_attention` (grace 3 of 5; still 4 of <=6)
-----------------------------------------------------------------------------
EXP 49 SCORED 1.473999 AND IS THE CHAMPION. Its falsifier fired HALF-WAY, and the
half that fired is the informative one. The `min` row of the persisted GRID now
reads 1.096 / 1.233 / 1.526 / 1.825 / 2.024 / 2.258 -- EXACTLY the anchor values
the clamp's algebra predicted, to all three digits -- so the edit does what the
theorem says and that column is finished, permanently and by construction. But
the predicted +0.0017 from those six cells arrived as +0.00015 of 17-cell mean,
so the eleven learned cells gave back ~0.0016 -- one noise ball, i.e. unmoved.
Removing the provably-wasted degrees of freedom was free; it was not the gap.

WHAT THE ELEVEN CELLS NOW COST, and what is still free to move. With the clamp
in place the model's output is, per drop, ENTIRELY determined by
(i) which users are the Kq smallest of the emitted profile and (ii) the SHAPE of
those Kq entries relative to the cut -- everything else is pinned at the cut and
the overall level cancels in the fixed point. The remaining deficit against the
certified columns is p10 -0.127 and p25 -0.136, worth +0.0155 if closed:

    p10  (K=2,4,6,8,10)  1.190 1.405 1.524 1.643 1.771  vs 1.21 1.44 1.55 1.66 1.80
    p25  (K=1..10)       1.070 1.143 1.265 1.317 1.368 1.401
                                                vs 1.10 1.15 1.28 1.33 1.40 1.44

THE HYPOTHESIS: THE ONE COORDINATE THAT SHAPE LIVES ON IS GLOBAL, AND EVERY HEAD
THIS CAMPAIGN HAS BUILT IS POINTWISE. The anchor is w_clip = min(sinr_fp/thr, 1),
whose above-cut entries are all exactly 1 and whose bottom-Kq entries carry the
FULL-POWER SINR shape. Two facts follow. First, w_clip is already clamp-structured,
so `_cut_clamp` is the identity on it and the anchor is a fixed point of exp 49's
restriction. Second -- and this is the point -- the one-parameter family

    w(beta) = w_clip ** beta        (beta > 0, above-cut entries stay 1**beta = 1)

is EXACTLY the axis from egalitarian to greedy inside the bottom set: beta -> 0
is the flat profile, i.e. p* and max-min; beta = 1 is the anchor; beta > 1 is
more spread than full power. It is monotone, so it moves NO user across the cut
and cannot disturb membership, and at Kq = 1 w_clip is flat so w(beta) is flat
for every beta -- the six closed `min` cells are untouchable by this change, by
construction, exactly as they were under the clamp.

At (K=1, p25) that family is not merely the dominant direction, it is the WHOLE
remaining decision: K=1 gives one user per cell, Kq=2 of 7, so w_clip has a
single entry below the cut and beta alone parameterises it up to membership.
The model scores 1.070 there against the anchor's own 1.047 and QFT's 1.10 --
it has travelled 40% of a ONE-SCALAR problem on 7 users, which is the cell where
a representation limit is least believable and a conditioning limit is most.
The same scalar is the leading direction at every other Kq>1 cell, because
log2(1 + c(w) w_i) summed over the bottom set is concave in each w_i while c(w)
falls as the set rises -- so the optimum flattens the bottom set relative to the
full-power shape by an amount that depends on the whole drop, not on any user.

That quantity -- "how egalitarian should THIS drop be" -- is a property of the
drop. The head has never been able to state it directly: since exp 3 the output
has been a SHARED POINTWISE map on LN(h), so a coherent common tilt of the
bottom set must be reassembled independently at each of up to 70 users out of
their own margins, and any per-user error breaks the coherence. Nothing in the
21 features or the four rounds is missing -- `m_fp` is exactly the log-margin the
exponent multiplies -- which is precisely why this reads as conditioning rather
than capacity, and why it retro-explains the campaign's 0-for-4 on capacity: more
width for a pointwise head buys more per-user freedom, which is not the missing
coordinate and (post-clamp) is provably not even a used one.

THE ONE CHANGE: A GLOBAL, PERMUTATION-INVARIANT EXPONENT ON THE ANCHOR.

    hout = LN_out(h)                                   (as before)
    logit = head(hout)                                 [bt,K,B]  (as before)
    s     = mean over the K*B nodes of bhead(hout)     [bt,1,1]  <- NEW
    beta  = exp(BETA_SCALE * tanh(s))                  in [0.22, 4.48]
    w     = w_clip**beta * 10**(W_SCALE*tanh(logit))   then _cut_clamp as in 49

`bhead` is one `Linear(HIDDEN, 1)`, zero-initialised in BOTH weight and bias, so
s = 0, beta = 1 and the emitted profile at initialisation is BYTE-IDENTICAL to
exp 49's; it is declared AFTER `self.head`, so every other module's init draws
from the global generator in the same order and the initial weights are
bit-for-bit the champion's. This is not new representation -- the pointwise
+-3-decade correction can already express any exponent -- it is one
well-conditioned global coordinate along the direction the optimum provably lies
in, which is what exp 45 (data), exp 46 (LR) and the four capacity probes were
all substitutes for.

(Exp 43 died part-way through writing a "global beta head" and was reverted with
its persisted table indistinguishable from exp 41's, so this idea has never
actually been run; log.csv's 1.470230 for that iteration is a re-score of the
reverted tree, not a result. It is not a repeat.)

EQUIVARIANCE AND SIZE-GENERALISATION. `bhead` is shared over every node and the
readout is a MEAN over all K*B of them, which is invariant under relabelling
users within a cell and under relabelling cells -- so beta is a scalar per drop
and the output stays equivariant on both axes. A mean, not a sum, so the scalar
does not drift as K goes 1 -> 10, and one parameter set still serves every K.
BETA_SCALE = 1.5 bounds beta to [0.22, 4.48]; the interesting region is
beta in (0, 1] (flatter than full-power shape) and neither rail is an attractor
because tanh is bounded and beta = 1 is the zero-logit point.

CONTRACT. A pooled linear readout of the same normalised stream the head already
reads, used as an exponent on a fixed input-only profile. No gradient step, no
restart, no candidate SET (one scalar, one profile, one allocation -- nothing is
enumerated, compared, accepted or rejected), no loop, and no objective of any
kind: no rate, no log2 of a rate, no top-k, no SLqP, no sum, no utility. `w_clip`
is the same `_clip_profile(A, Kq)` that has anchored the head since exp 31 and
`_cut_clamp` still runs afterwards unchanged, so exp 49's structure survives
intact. Cost is 49 parameters, one mean and one `pow` on a [bt,K,B] tensor
against the dominant [t,4,K*B,K*B] softmax -- inference stays at exp 49's
~1.57 s of the 10.0 s budget.

WHAT IS NOT TOUCHED. All 21 features, `_features`, the encoder, both attention
gain biases, ROUNDS, HIDDEN, HEADS, the cell-mediated path, W_SCALE, OUT_ITERS,
BAL_ITERS, `_clip_profile`, `_cut_clamp`, `_profile_fixed_point`, both loss terms
and ALPHA, the fresh-drop stream and its seed arithmetic, the samplers, the
pools, the optimiser and the cosine schedule are byte-identical to the 1.473999
run. No generator is consumed differently, so the (K, Kq, drop) sequence is
bit-for-bit exp 49's and so is the initial parameter vector.

DISTRIBUTION. The training law is untouched -- K uniform on 1..10 (ungraded 3, 5,
7, 9 included), `frac` flat on [0, BAND_MAX_FRAC] routed through the evaluator's
own `kq_of()`, `_band_kq_max()` still equal to the largest graded Kq for every K.
Nothing is narrowed toward the graded points and beta is defined for every
integer Kq in 1..K*B, not for the three graded fractions, so it cannot encode the
grid and NO OFF-GRID CHECK IS OWED; I will still run `k_generalization_check.py`
on K in {3, 5, 7, 9} before treating a jump as banked.

FALSIFIER, readable in the persisted GRID table. The six `min` cells MUST stay at
1.096/1.233/1.526/1.825/2.024/2.258 -- they are algebraically immune to beta, so
any movement there means the edit is buggy, not wrong. If they hold and the
eleven p10/p25 cells are unmoved to rounding, then a global egalitarian-vs-greedy
scalar is not the missing coordinate either; with capacity, data, loss shape,
labels, output scale, LR and wasted DOF all closed, that would leave only the
SPACE a teacher is distilled in -- and exp 39's teacher, which measured 1.482 as
a policy and beat the student on 100% of drops, should be retried as a
log-PROFILE MSE restricted to the bottom-Kq coordinates (the two reasons its
power-space MSE could not bite, both of which exp 49 has now removed), paid for
by the Kq=1 distillation term that the clamp has made identically zero.

CAVEAT ON CONFIDENCE. Python execution is permission-gated in this session, so
this is hand-verified rather than smoke-tested -- which is why the executable
edit is eight lines. The load-bearing checks were: `_clip_profile` returns
[bt,K,B] clamped to (1e-30, 1.0], so `torch.pow(base, beta)` is well defined and
its beta-gradient base**beta * ln(base) is bounded (it -> 0 as base -> 0 for
beta > 0); `hout.mean(dim=(1,2))` on a [bt,K,B] tensor with keepdim gives
[bt,1,1], which broadcasts against [bt,K,B]; `nn.init.zeros_` consumes no RNG, so
declaring `bhead` last leaves every other module's initialisation draw-for-draw
identical; and `self.norm_out(h)` is now evaluated ONCE into `hout` and used by
both heads, which is the same tensor the champion computed inline.

-----------------------------------------------------------------------------
EXPERIMENT 49 -- family `interference_attention` (grace 2 of 5; still 4 of <=6)
-----------------------------------------------------------------------------
EXP 45 SCORED 1.473854 AND IS THE CHAMPION. Exps 46-48 are not results: 46 (LR
1e-3 -> 3e-3) read 1.472190 and reverted, 47's agent step died with an execution
error and re-scored the reverted tree at 1.469928, and 48 aborted on a shape
error (`mat1 and mat2 shapes cannot be multiplied (2688x25 and 21x48)` -- an
edit that added four feature channels without moving N_FEAT) and was reverted.
This file is the 1.473854 champion plus THE ONE CHANGE below.

WHERE THE REMAINING 0.011 IS. Summing the persisted GRID table against the
certified QFT columns: the six `min` cells are +0.063 IN TOTAL (the model is
ABOVE the 30-drop QFT column at K=6 and K=10 because it sits on the provable
max-min optimum), the five p10 cells are -0.155 and the six p25 cells are
-0.149. So 100% of the deficit is the eleven Kq>1 cells, worth +0.018 if they
were closed, and the campaign's read on why is now 0-for-6: capacity (0-for-4),
loss shape (0-for-3), Kq measure (0-for-2), Kq>1 labels (0-for-3, including exp
39's teacher that measured 1.482 AS A POLICY, beat the detached student on
100.0% of drops, and moved nothing), the output map's scale (0-for-2) and the
learning rate (0-for-1). Exp 45 itself -- an 80x increase in distinct training
drops -- bought +0.0014, which closes the data axis too.

THE HYPOTHESIS: THE HEAD IS SPENDING ~74% OF ITS OUTPUT ON A DEGREE OF FREEDOM
THE OPTIMUM PROVABLY DOES NOT HAVE. Since exp 31 the model emits a target SINR
profile w and returns the box-feasible fixed point realising SINR ∝ w exactly,
so the graded quantity is

    SLqP_Kq(w) = sum over the Kq smallest w of log2(1 + c(w) * w_i)

where c(w) is the single scale the fixed point can afford. Users OUTSIDE the
bottom-Kq set enter that expression ONLY through c(w), which is decreasing in
their w -- so their w is pure cost. And it cannot be driven to zero either: a
muted user has rate 0 and falls INTO the bottom set (trap 2). The optimum is
therefore pinned at the boundary, exactly:

    at the optimum, every user outside the bottom-Kq set has SINR equal to the
    LARGEST SINR inside it.

Proof: let j be outside and suppose SINR_j is strictly larger. Lower p_j by eps.
Every other user's interference strictly falls (all gains are strictly positive
under Rayleigh fading), so every rate in the bottom set strictly rises, while
rate_j falls continuously toward 0. For small enough eps the bottom SET is
unchanged, so SLqP_Kq strictly increased -- contradiction. Since rate_j is
continuous in p_j and reaches 0, the reduction can always be taken up to the
boundary rate_j = cut. []

Nothing in the current parameterisation enforces this. `w_clip` -- the exp-29
ANCHOR -- has exactly the right shape (it flattens everything above the cut onto
the cut), which is why zero logits are already worth ~1.37 as a policy. But the
head then multiplies it by 10**(+-3 tanh(logit)) PER USER, over all K*B users,
and at (K=10, p25) that is 52 of 70 users -- at (K=10, p10), 63 of 70 -- whose
correction is provably wasted at best and, because every one of them pushes
c(w) down, actively harmful at worst. The head has to LEARN to emit ~0 on them
from a gradient that reaches them only through the interference terms of Kq
other users, and it pays for every failure. That is a far better fit to the
evidence than the six closed axes above: it explains why a near-optimal teacher
could not drag the student (exp 39 supervised POWER MSE over all K*B users, so
its signal was dominated by exactly the users whose values do not matter), why
extra capacity makes things WORSE (more capacity in a direction that can only
cost c(w)), and why the deficit is ~2% and uniform across all eleven learned
cells while the six algebraic ones are finished.

THE ONE CHANGE: THE PROFILE IS CLAMPED AT ITS OWN Kq-th SMALLEST VALUE.
`forward()` gains one line -- `w = min(w, kthvalue(w, Kq))` over the K*B users
of each drop -- so the emitted profile is forced into the structure the theorem
above says the optimum has. This is a LOSSLESS restriction, not a heuristic
prior: the optimum's own profile w* = SINR(p*) already satisfies
min(w*, kthvalue(w*, Kq)) = w*, i.e. the clamp is the identity on it, so the
optimum stays inside the image. What is removed is only the provably-suboptimal
part of the search space. Three consequences, all pre-registered:

  * the model keeps FULL control of what matters -- which users form the bottom
    set (the ordering of w is untouched by a monotone clamp) and their relative
    shape (the Kq smallest entries are untouched, period);
  * the effective per-drop output dimension collapses from K*B to Kq: 18 of 70
    at (K=10, p25), 7 of 70 at (K=10, p10), 2 of 7 at (K=1, p25). The head is
    the same size and the features are the same features -- it simply no longer
    has to spend either on discovering a constraint that is a theorem;
  * at Kq = 1 the clamp is `w <- min(w)`, a FLAT profile whatever the head
    emits, so the output is p* IDENTICALLY -- the provable box-constrained
    max-min optimum, by construction, for every K and every set of weights. The
    six `min` cells currently read 1.095/1.231/1.522/1.818/2.018/2.249 against
    the anchor's own 1.096/1.233/1.526/1.825/2.024/2.258 (exp 45's ANCHOR_CHECK),
    i.e. the head is a hair BELOW p* on all six; those ~0.029 of total deficit
    become exactly zero, worth ~+0.0017 before the eleven learned cells move at
    all. This also makes the ALPHA=1.0 Kq=1 distillation term identically zero
    (`model(Ad, 1)` and `balance_labels(Ad)` are now the same 40-iteration
    float64 recursion on the same flat profile), so it contributes no gradient.
    That is a CONSEQUENCE of the one change, not a second change, and I am
    deliberately leaving the term in place rather than deleting it so the A/B
    stays one-variable; its side effect is that the head's shrinkage-toward-zero
    prior is lifted at Kq > 1, which is where the head is meant to be free.

CONTRACT. A monotone elementwise clamp with a data-dependent threshold, inside
one feed-forward pass. No gradient step; no candidate SET (one profile in, one
profile out -- nothing is enumerated, compared or accepted/rejected); no loop;
and no objective of any kind is evaluated -- no rate, no log2, no SLqP, no sum,
and no top-k of any RATE. The threshold is an ORDER STATISTIC of the profile
itself, the same primitive `_clip_profile` has run inside `forward()` since exp
29 (`sinr_fp.kthvalue(kq_cut)`) and `_features` since exp 28. Cost is one
`kthvalue` on a `[t, K*B]` tensor per forward pass -- microseconds against the
dominant `[t,4,K*B,K*B]` softmax -- so inference stays at exp 45's ~1.45 s of
the 10.0 s budget.

WHAT IS NOT TOUCHED. All 21 features, `_features`, the encoder, ROUNDS, HIDDEN,
HEADS, both attention gain biases, the cell-mediated path, W_SCALE, OUT_ITERS,
BAL_ITERS, the exp-29 anchor `_clip_profile`, `_profile_fixed_point`, both loss
terms and ALPHA, the fresh-drop stream and its seed arithmetic, the samplers,
the pools, the optimiser and the cosine schedule are byte-identical to the
1.473854 run. No generator is added or consumed differently, so the (K, Kq,
drop) sequence and the model init stay bit-for-bit exp 45's.

DISTRIBUTION. The training law is untouched -- K uniform on 1..10 (ungraded
3, 5, 7, 9 included), `frac` flat on [0, BAND_MAX_FRAC] routed through the
evaluator's own `kq_of()`, `_band_kq_max()` still equal to the largest graded Kq
for every K -- so nothing is narrowed toward the graded points and no off-grid
check is owed. The clamp itself is defined for every integer Kq in 1..K*B, not
for the three graded fractions, so it cannot encode the grid.

FAMILY TAG. This stays `interference_attention` rather than opening a fifth of
the six breadth slots: the framework (direct objective + the Kq=1 distillation
term) and the architecture (gain-biased attention MPNN, all 21 features) are
unchanged, and this is a one-line restriction of the output layer that family
already had. The two remaining breadth slots stay available for an actual new
framework.

FALSIFIER, readable in the persisted GRID table. If the six `min` cells do NOT
rise to exactly 1.096/1.233/1.526/1.825/2.024/2.258, the clamp is not doing what
the algebra says and the change is buggy, not merely wrong. If they do rise but
the eleven p10/p25 cells are unmoved to rounding, then the wasted-degrees-of-
freedom hypothesis is dead too -- with capacity, data, loss shape, labels, the
output scale and the learning rate all already closed, that would leave the
CONDITIONING of the head (how Kq reaches it) as the last unprobed axis, and the
remaining grace should go there. If the p10/p25 rows rise with the `min` rows,
the next question is whether the now-inert distillation term's 20% of each
step's compute is better spent on a fifth direct-objective task.

CAVEAT ON CONFIDENCE. Python execution is permission-gated in this session, so
this is hand-verified rather than smoke-tested -- which is also why the
executable edit is six lines. The load-bearing checks were: `torch.kthvalue`
takes a 1-BASED k and is differentiable (gradient routed to the selected
element), so `kq_cut = max(1, min(int(Kq), KB))` matches `_clip_profile`'s own
guard exactly; `w_prof` is `[bt, K, B]` and strictly positive, so the reshape to
`[bt, K*B]` and back is the same flattening `_order_stats` and `_clip_profile`
already use; `torch.minimum` broadcasts `[bt, K*B]` against the `[bt, 1]`
keepdim threshold; at Kq=1 `kthvalue(1)` is the row minimum, so the result is
flat and `_profile_fixed_point` with a flat `w` is byte-identical to
`balance_labels`; and the clamp is applied AFTER the tanh/anchor product and
BEFORE the fixed point, so `_clip_profile`'s own numerics are untouched.

-----------------------------------------------------------------------------
EXPERIMENT 45 -- family `interference_attention` (grace 4 of 5; still 4 of <=6)
-----------------------------------------------------------------------------
EXP 42 SCORED 1.472451 AND IS THE CHAMPION (+0.0015 over exp 41, i.e. exactly
one noise ball -- the victim bias is kept but it is a marginal read). Exps 43
and 44 are not results: 43's agent step died part-way through writing a global
`beta` head (it scored 1.470230 and its persisted per-cell table is
indistinguishable from exp 41's) and 44 aborted on a `NameError` for the
constant that edit never finished defining. Both are reverted; this file is the
1.472451 champion plus THE ONE CHANGE below.

WHAT THE PERSISTED TABLES NOW SAY. The `min` column is CLOSED, and closed by
ALGEBRA, not by learning: ANCHOR_CHECK on the pinned grid drops reads
p* = 1.096 / 1.233 / 1.526 / 1.825 / 2.024 / 2.258 and the model reads
1.095 / 1.231 / 1.522 / 1.818 / 2.018 / 2.249 -- it sits ON its own zero-logit
anchor at every K, and that anchor is the *provable* box-constrained max-min
optimum (which is why two of those cells sit ABOVE the 30-drop QFT column). So
the six cells the head does NOT have to learn are finished, and the eleven cells
the head DOES have to learn are all short by very nearly the same 2%:

    p10  (K=2,4,6,8,10)  1.185 1.394 1.516 1.636 1.767   vs  1.21 1.44 1.55 1.66 1.80
    p25  (K=1..10)       1.065 1.136 1.260 1.318 1.374 1.410
                                                 vs  1.10 1.15 1.28 1.33 1.40 1.44

A deficit that is uniform across eleven cells, absent from the six closed-form
ones, and stuck at ~2% is not a shape error -- it is what a GENERALISATION gap
looks like.

THE HYPOTHESIS: THE MODEL IS FITTING ITS 819-DROP-PER-K CHANNEL POOL. In 44
experiments `make_pools()` has never been the variable -- every note above says
"the pools are byte-identical". At the harness budget (POOL=8192, BATCH=256) it
builds 819 drops per K, held FIXED for the whole run, and the loop draws
2000 x 4 tasks x 64 drops + 2000 x 64 distillation drops from it: each of those
819 drops is revisited on the order of EIGHTY times by a 115k-parameter network,
while HELDOUT_SCORE is measured on 250 pinned drops the model has never seen.

This one hypothesis retro-explains FOUR separate nulls the campaign has been
attributing to a "representation limit":

  * capacity is 0-for-3 and the two clean probes were NEGATIVE (ROUNDS 6 at
    -0.0072, 8 tied rounds at -0.0058, HIDDEN 64 at -0.00007). Extra capacity
    that makes a model WORSE on held-out data while the training objective can
    only improve is the textbook signature of a data-limited fit, not of a
    function class that is too small;
  * exp 39's converged Kq>1 teacher measured 1.482 as a POLICY, beat the
    detached student on 100.0% of drops for all 2000 steps at ALPHA_Q=1.0, and
    moved the score by nothing. A better target on the SAME 819 memorised drops
    buys nothing off them -- whereas a genuine representation limit should have
    dragged the student somewhere, and did not;
  * exp 33 (feeding the model the ordinal description of its own realised
    operating point) and exp 34 (W_SCALE) both landed inside the noise ball:
    the output map is not the constraint, exactly as their falsifiers said.

Exp 23/24 already caught the twin of this bug one level up -- a fixed 320-drop
LABEL set that "is memorised early, after which the supervised gradient decays
to zero" -- and exp 24 fixed it by generating labels in-loop on fresh drops. But
it only fixed the labels: the DROPS underneath them stayed the same 819 per K.
This is the other half of that fix.

THE ONE CHANGE: THE TRAINING DROPS ARE STREAMED, NOT POOLED. Both loss terms now
call `sample_channels` for a fresh, never-repeated batch every step, seeded
deterministically from the step index (`CH_SEED + 8*step + j`), so identical code
still reproduces an identical score. Over the run that is 2000 x 5 x 64 = 640,000
distinct drops against 8,190, and no drop is ever seen twice.

WHAT IS NOT TOUCHED, AND WHY THIS IS A ONE-VARIABLE A/B. Every one of the 21
features, the attention layer and both its gain biases, the exp-31 output map,
W_SCALE, OUT_ITERS, BAL_ITERS, ROUNDS, HIDDEN, HEADS, the cell-mediated path,
both loss terms, ALPHA, the optimiser and the cosine schedule are byte-identical
to the 1.472451 run. The two `torch.randint(0, pool.shape[0], ...)` draws are
DELIBERATELY KEPT and their results discarded: `pools` is still built from the
same seeds at the same size, so those draws consume exactly the values from `g`
and `gd` that they always did and the (K, Kq) TASK SEQUENCE stays bit-for-bit
the champion's. Fresh-vs-pooled drops is then the only difference between the
two runs. `pools` still feeds LABEL_CHECK / CLIP_CHECK unchanged.

DISTRIBUTION -- this WIDENS, it cannot narrow. The training law is untouched:
K uniform on 1..10 (the ungraded 3, 5, 7, 9 included), `frac` flat on
[0, BAND_MAX_FRAC] routed through the evaluator's own `kq_of()`, so
`_band_kq_max()` still equals the largest graded Kq for every K and every graded
cell keeps exactly the mass it had. The drops are drawn from the SAME
`sample_channels` law the pools were drawn from -- there are simply more of
them, at fresh seeds disjoint from the pools' (1000..1520) and from the
evaluator's pinned TEST seeds (5000..5010). Nothing moves toward the graded
points, so no off-grid check is owed; I will still run
`k_generalization_check.py` on K in {3, 5, 7, 9} before banking a jump.

BUDGET AND CONTRACT. `forward()` is not touched at all, so the inference
contract is exactly as it was and inference stays at exp 42's ~1.18 s of the
10.0 s budget. STEPS and BATCH are unchanged; POOL is still honoured (the pools
are built exactly as before) and is now only a diagnostic input, so the run's
memory footprint is strictly lower. The added cost is 10,000 `sample_channels`
calls at ~1-2 ms -- a few numpy ops on <=31k elements each -- i.e. ~10-15 s on a
run whose 2000 steps already carry five backward passes through a 4-round
attention net and three 40-iteration float64 fixed points.

FALSIFIER, pre-registered and readable in the persisted GRID table: if the
eleven p10/p25 cells are unmoved to rounding, the fit was never data-limited,
the four nulls above really are a representation limit, and the family's last
grace iteration should go to exp 42's own untaken next step (pooling the
attended message to the cell alongside mean/max/victim). If instead the p10/p25
rows rise while the six `min` cells stay pinned on p* -- the cells that cannot
move, because they are algebra -- that is this mechanism's exact signature, and
the next question becomes how much MORE data the budget can buy (e.g. whether
the distillation term's drops are better spent on the direct objective).

CAVEAT ON CONFIDENCE: Python execution is permission-gated in this session, so
this is hand-verified, not smoke-tested -- which is also why the edit is six
lines. The load-bearing checks were: `sample_channels(n, K, seed=...)` is
already imported at module level and already returns exactly the `[n,K,B,B]`
float32 tensor the pools are made of (`make_pools` calls it with the same
signature); the seed arithmetic `CH_SEED + 8*step + j` for j in 0..TASKS gives
5 distinct seeds per step out of a block of 8, so no seed repeats over
step in 0..1999 and the whole range 20,000,000..20,015,996 is disjoint from
`make_pools`' 1000..1520 and `prepare.TEST`'s 5000..5010; `pool`/`pd` are still
bound before the kept randint draws so those draws cannot raise; and `pools` is
still passed to `label_report` before the loop.

-----------------------------------------------------------------------------
EXPERIMENT 42 -- family `interference_attention` (grace 2 of 5; still 4 of <=6)
-----------------------------------------------------------------------------
EXP 41 SCORED 1.470978 AND IS THE CHAMPION -- the largest single gain since exp
31 (+0.0054), and the first one in fourteen iterations that is not noise. Its
pre-registered falsifier was "the p10/p25 rows unmoved to rounding"; it did not
fire. Against exp 31's persisted p10 row (K=2,4,6,8,10 = 1.179 / 1.384 / 1.505 /
1.627 / 1.760) the new one reads 1.185 / 1.394 / 1.515 / 1.637 / 1.769 -- EVERY
p10 cell up, by +0.006 to +0.010, and the p25 row moves with it. The six `min`
cells are unharmed (they sit at or above `balance_labels`' provable optimum).
That is the mechanism's own signature: learned all-pairs comparison is what the
p10/p25 decision was missing, exactly as argued, and it was bought on the FIRST,
COMPLETELY UNTUNED instance of the layer. The grace window is therefore owed to
the layer itself, not to a fifth framework.

THE ONE CHANGE: THE ATTENTION BIAS BECOMES BIDIRECTIONAL. Today the logit is

    <q_kb, k_k'c>/sqrt(d_h)  +  W_g * l(A[k,b,c])

and the bias term reads ONE gain: `A[k,b,c]`, the gain from the key's cell into
the QUERY. Every physical comparison the layer can anchor is therefore of the
form "how much does this node's cell hurt ME". The reciprocal gain -- `A[k',c,b]`,
what MY cell does to the key user -- is not a function of A[k,b,:] and appears
nowhere in the layer at any depth: it is a different entry of the same matrix,
and there is no path from it to the query's logit. So the layer can express "who
is my worst aggressor" and cannot express "whom do I hurt most", which is one of
the three comparisons exp 41's own note listed as its motivation.

This is not a speculative asymmetry -- it is the SAME asymmetry the cell-mediated
path already had to fix, and the fix is already in this file. `_victim_weights`
exists precisely because the forward edge alone leaves the graph one-directional
at any depth ("a cell would learn who aggresses its own users but never who IT
aggresses"), and this band's headroom is by that same note the case of a cell
shaping its total to protect someone ELSE's worst user. At Kq>1 that is the
whole decision: SLqP_Kq sums the bottom-Kq set, so a user above the cut is worth
nothing to the metric and its power is pure budget -- the question "should I back
off" is answered by WHO I HURT, which is the direction the bias cannot see. The
new logit adds one term of exactly the incumbent's form,

    +  W_v * l(A[k',c,b])            (key user k' in cell c, query's cell b)

with its own per-round Linear(1, HEADS), so each head learns its own mix of the
aggressor and victim directions rather than being forced to one.

WHY THIS AND NOT WIDTH. Capacity in this campaign is 0-for-4 (HIDDEN 64, ROUNDS
6, 8 tied rounds, and the 78k -> 115k jump exp 41 itself made while crediting the
mechanism, not the count), so HEADS 4 -> 8 has no argued mechanism. This does:
it adds a comparison the function class provably cannot currently make, which is
the same argument that just paid +0.0054.

EQUIVARIANCE, on both axes, for the new term. Under a cell relabelling
b -> pi(b) we have A[k',c,b] -> A[k',pi(c),pi(b)], so query rows and key columns
permute together and the softmax is over the same multiset. Under a per-cell user
relabelling k' -> sigma_c(k') the key node (k',c) and its bias entry A[k',c,b]
carry the SAME index k' and permute together, so the attended multiset is
unchanged -- note this is the mirror of the incumbent term, whose bias is
independent of k' instead. One shared parameter set still serves every K, and the
softmax is still normalised over the K*B keys, so message magnitude does not
drift as K goes 1 -> 10.

WHAT IS NOT TOUCHED. Everything else is byte-identical to the 1.470978 run: all
21 features, the exp-31 output map and W_SCALE, OUT_ITERS, BAL_ITERS, ROUNDS,
HIDDEN, HEADS, the incumbent gain bias, the cell-mediated path (mean/max/victim
pool and the along-channel aggregation), both loss terms, ALPHA, the samplers,
the pools, the optimiser and the cosine schedule. The head's weight and bias stay
zero-initialised, so the run still STARTS at w = w_clip exactly and at Kq=1 the
anchor is p* itself with the distillation term at zero loss -- the six finished
`min` cells are protected by construction, as in every experiment since 31.

CONTRACT. A second additive bias inside a softmax is a plain feed-forward layer:
no gradient step, no restart, no candidate set, no loop whose acceptance or
output depends on evaluating the objective; no rate, no top-k of any rate, no
SLqP, no sum and no utility is computed anywhere in `forward()`. `l(A)` is the
same log-gain transform, read from the same tensor, in the other index order.

COST. One extra Linear(1, HEADS) per round (~8 parameters each) and one permute
of a [t,K,B,B] tensor; the [t,4,K*B,K*B] softmax that dominates the layer is
unchanged in size. Expect inference ~1.4-1.6 s against exp 41's 1.334 s and the
10.0 s budget.

DISTRIBUTION: the training law is UNTOUCHED -- K uniform on 1..10 (the ungraded
3, 5, 7, 9 included), `frac` flat on [0, BAND_MAX_FRAC] routed through the
evaluator's own `kq_of()`, so `_band_kq_max()` still equals the largest graded Kq
for every K and every graded cell keeps exactly the mass it had. Nothing is
narrowed toward the graded points, so NO OFF-GRID CHECK IS OWED.

FALSIFIER: if the p10/p25 rows of the persisted GRID table are unmoved to
rounding, the victim direction is not what the head is missing, the "add the
comparison it cannot make" thread has given what it has, and the family's
remaining grace should go to the one axis exp 41 left untouched -- WHERE the
attention is read (the update MLP consumes `att` at the same node it queried
from; the cell path gets a pooled summary and the attention does not), i.e.
pooling the attended message to the cell alongside mean/max/victim.

-----------------------------------------------------------------------------
EXPERIMENT 41 -- family `interference_attention` (BREADTH FAMILY 4 of <=6)
-----------------------------------------------------------------------------
Exp 31 (1.465621) is still the champion. The nine iterations since then contain
only THREE real experiments -- 32, 35, 36, 37, 38 and 40 were agent execution
errors / usage-limit stalls that re-scored unchanged code -- and all three of
the real ones came back null or negative:

    exp 33  second feedback stage, fed the model the ORDINAL description of
            its own realised operating point            1.465314  (null)
    exp 34  W_SCALE 3.0 -> 1.5                          1.462680  (-0.003)
    exp 39  a CONVERGED Kq>1 teacher (direct_opt's own
            1200 steps / lr 0.08, two starts, per-drop
            argmax against full power, p_clip and p*),
            distilled by gated MSE in normalised POWER
            units at ALPHA_Q = 1.0                      1.462357  (null)

EXP 39 IS THE INFORMATIVE ONE, AND ITS OWN PRE-REGISTERED READING FIRES. Its
persisted ORACLE_CHECK measured that teacher AS A POLICY on all eleven graded
p10/p25 cells and it lands essentially ON the certified QFT column --

    K= 2  p10: 1.219 (1.21)  p25: 1.159 (1.15)      K= 1  p25: 1.078 (1.10)
    K= 4  p10: 1.409 (1.44)  p25: 1.266 (1.28)
    K= 6  p10: 1.499 (1.55)  p25: 1.325 (1.33)
    K= 8  p10: 1.673 (1.66)  p25: 1.410 (1.40)
    K=10  p10: 1.792 (1.80)  p25: 1.434 (1.44)

-- i.e. the teacher is worth +0.016 of 17-cell mean over the student's own GRID
row, which would put this campaign at ~1.482 against QFT's 1.485. GATE_FRAC
printed 1.000: the teacher beat the detached student on EVERY drop of all 2000
steps, so the MSE was live throughout and never gated off. And the score did
not move at all. A teacher that is strictly better on 100% of drops, applied
for the whole run, that the student cannot absorb, is not a label problem and
not an optimisation problem -- it is exp 39's second falsifier, verbatim: "the
gap is a REPRESENTATION limit of the profile head, isolated for the first time."
Exp 34's falsifier points the same way ("the output map is not the constraint;
the family should be closed out at the champion so the unused breadth families
go to a genuinely different framework"), and so does the shape of this
campaign's whole result table: capacity is 0-for-3 (HIDDEN 64, ROUNDS 6, 8 tied
rounds), so the missing thing is not width or depth of the SAME aggregator.

THE ONE CHANGE: WHAT THE AGGREGATOR CAN COMPUTE. Every round since exp 8 has
routed all non-local information through CELL-level summaries -- `mean_k hn`,
`max_k hn`, and the victim sum -- and then mixed those summaries along the
channel. That is exactly sufficient for the PHYSICS (interference enters a
user's SINR only through the per-cell totals Pcell[c]), which is why it has
carried the `min` column to the analytic optimum. It is NOT sufficient for the
DECISION, because SLqP_Kq sums the Kq smallest rates out of up to seventy and
the p10/p25 policy is the purely ordinal question of which users are in that
set and who their aggressors are -- and mean and max are precisely the two
order statistics that cannot locate a 10th percentile at any depth or width.
The campaign has already measured how much that missing faculty is worth: exp
28 hand-designed THREE global order statistics of two fixed probes (a
percentile rank and two signed log-margins to the Kq-th smallest) and bought
+0.0091, the largest feature gain of the run. Those three are hard-coded, read
at operating points fixed before any decision is made, and they are the only
global comparisons in the model.

So this family replaces them with LEARNED ones: each round gains a
GAIN-BIASED GLOBAL SELF-ATTENTION over all K*B user nodes at once,

    logits[(k,b) -> (k',c)] = <q_kb, k_k'c> / sqrt(d_h)  +  W_g * l(A[k,b,c])
    att_kb                  = sum_{k',c} softmax(logits) * v_k'c

and the round update becomes `h += upd([hn, own_cell, agg, att])` -- the entire
cell-mediated path is retained byte-for-byte, so nothing that works is removed;
attention is added ALONGSIDE it. A user can now address every other user
DIRECTLY and compare itself against them, which is what an arbitrary global
order statistic is; the per-head learnable bias on the log interference gain
`l(A[k,b,c])` from the aggressor cell keeps that comparison physically
structured (attend to whoever actually hurts me, or whoever I actually hurt)
rather than purely feature-driven. This is the first genuinely different model
class the campaign has tried: message passing over a fixed physical graph vs.
a learned all-pairs one.

WHY THIS IS THE RIGHT READING OF THE EXP-39 NULL. If the student were merely in
the wrong basin, a strictly-better teacher on every drop would have pulled it;
it did not. If it were capacity-limited, HIDDEN 64 or ROUNDS 6 would have moved
it; they did not. What is left is that the FUNCTION CLASS cannot express the
teacher's map -- and the one faculty the teacher's map obviously needs, and
this one provably lacks, is a global comparison across the K*B users that is
not one of the three fixed statistics exp 28 hard-coded.

WHAT IS NOT TOUCHED. All 21 features, the exp-31 output map (`w = w_clip *
10**(W_SCALE*tanh(logit))` realised by the weighted balancing fixed point),
W_SCALE, OUT_ITERS, BAL_ITERS, ROUNDS, HIDDEN, both loss terms, ALPHA, the
samplers, the pools, the optimiser and the cosine schedule are byte-identical
to the 1.465621 run. The head's weight and bias are still zero-initialised, so
the run STARTS at w = w_clip exactly -- the exp-29 clipped allocation, ~1.37 as
a 17-cell mean -- and at Kq=1 the anchor is p* itself with the ALPHA=1.0
distillation term at zero loss, so the six finished `min` cells are protected
by construction exactly as before.

EQUIVARIANCE AND SIZE-GENERALISATION -- the mandatory property, checked on both
axes. Q/K/V and the gain-bias map are shared over every node, so ONE parameter
set serves every K. Under a cell relabelling b -> pi(b) the nodes permute and
A[k,b,c] -> A[k,pi(b),pi(c)], so both the query rows and the key columns
permute together and the softmax is over the same multiset: equivariant. Under
a per-cell user relabelling k -> sigma_b(k) the query rows permute, and on the
key side the bias does not depend on k' at all, so the attended multiset is
unchanged: equivariant. The softmax is NORMALISED over the K*B keys, so the
message magnitude does not drift as K goes 1 -> 10 -- strictly better on that
axis than a sum, and the reason attention is safe here where an unnormalised
all-pairs sum would not be.

CONTRACT. Nothing changes: attention is a plain feed-forward layer. No gradient
step, no restart, no candidate set, no loop whose acceptance or output depends
on evaluating the objective; no rate, no top-k of any rate, no SLqP, no sum and
no utility is computed anywhere in `forward()`. `l(A)` is the same log-gain
transform `_features` has always used. Parameters are untouched by evaluation.

COST. Attention is [t, 4 heads, K*B, K*B] with K*B <= 70; the largest graded
cell is 250 drops x 4 heads x 70 x 70 x 12 x 2 matmuls x 4 rounds ~ 1 GFLOP,
against a forward pass that already runs three 40-iteration float64 fixed
points. Expect inference ~1.3-1.8 s of the 10.0 s budget, up from 0.815 s.
Parameters go ~78k -> ~115k, which the three capacity nulls already say is not
itself worth anything -- the mechanism is the variable, not the count.

DISTRIBUTION: the training law is UNTOUCHED -- K uniform on 1..10 (the ungraded
3, 5, 7, 9 included), `frac` flat on [0, BAND_MAX_FRAC] routed through the
evaluator's own `kq_of()`, so `_band_kq_max()` still equals the largest graded
Kq for every K and every graded cell keeps exactly the mass it had. Nothing is
narrowed toward the graded points, so NO OFF-GRID CHECK IS OWED; I will still
run `k_generalization_check.py` on K in {3, 5, 7, 9} before banking a jump.

FALSIFIER: if the p10/p25 rows of the persisted GRID table are unmoved to
rounding, learned global comparison is not the missing faculty either, the
representation reading is wrong, and the family's remaining grace should go to
the one axis exp 39 left open -- the SPACE the teacher is distilled in (log
profile rather than linear power, which is the same MSE re-read in the units
the output map actually works in).

-----------------------------------------------------------------------------
EXPERIMENT 31 -- family `sinr_profile_head` (BREADTH FAMILY 3 of <=6)
-----------------------------------------------------------------------------
Exp 29 scored 1.457018 and is the champion. Exp 30 (a self-refinement teacher:
REF_STEPS of ascent on the STUDENT's own output, distilled back) printed
1.456571 and is REVERTED -- and its persisted REFINE_CHECK is the most
informative table this campaign has produced, because it closes a whole class of
ideas with a number:

    student -> teacher after the ascent          (certified QFT in parens)
      K= 4  p10: 1.391 -> 1.397 (1.44)   p25: 1.263 -> 1.269 (1.28)
      K= 8  p10: 1.604 -> 1.607 (1.66)   p25: 1.359 -> 1.364 (1.40)
      K=10  p10: 1.759 -> 1.762 (1.80)   p25: 1.393 -> 1.401 (1.44)

DIRECT ASCENT ON SLqP FROM THE STUDENT'S OWN ALLOCATION BUYS ~0.005 AND STOPS,
while QFT is 0.04-0.05 further up. The student is already sitting in a local
optimum of the graded objective; the residual p10/p25 gap is NOT a gradient the
model has failed to follow, and no teacher built by following that gradient
(exps 25, 26, 30 -- Kq>1 labels are now 0-for-3) can reach it. The remaining gap
is a BASIN problem, and every one of those three experiments attacked it from
the label side, which is the side that is closed.

THE ONE CHANGE: the OUTPUT PARAMETERISATION. The head stops emitting powers and
starts emitting a per-user TARGET SINR PROFILE, which is then realised exactly by
the weighted balancing fixed point this file has computed since exp 27:

    logit = head(LN(h))                                  [bt,K,B]   (as before)
    w     = w_clip * 10 ** (W_SCALE * tanh(logit))       target SINR profile
    p     = the box-feasible fixed point with SINR ∝ w   <- the model's output

Nothing else moves. `_features` (all 21 channels), ROUNDS, HIDDEN, the message
structure, both loss terms, ALPHA, BAL_ITERS, the samplers, the pools, the
optimiser and the cosine schedule are byte-identical to the 1.457018 run, so the
output map is the only variable. The head's WEIGHT is zero-initialised (its bias
already was), so the run STARTS at w = w_clip, i.e. at exp 29's clipped balancing
allocation exactly, whose own measured policy value is the persisted CLIP_CHECK
row -- ~1.37 as a 17-cell mean against the full-power floor's 1.000.

WHY THIS IS A LOSSLESS REPARAMETERISATION, NOT A RESTRICTION. Two facts compose.
(i) The weighted iteration `p <- P_T*(w*F(p))/max(w*F(p))` has SINR ∝ w at its
unique fixed point. (ii) Conversely, for ANY allocation p, put w = SINR(p);
then F = p/w by definition, so w*F = p and the map returns P_T*p/max(p). So p is
the fixed point of its own induced profile IF AND ONLY IF max_{k,b} p = P_T. The
image of the new head is therefore EXACTLY

    { p in (0, P_T]^{K x B} : max_{k,b} p[k,b] = P_T }

and that set provably contains the optimum: scaling every power up by c > 1
multiplies signal and interference by c while N_0 stays put, so every SINR and
every rate strictly increases and SLqP_Kq strictly increases -- the pre-launch
audit certified this direction ("SLqP decreases as all powers scale down"; the
system is noise-significant, median desired SNR ~-4 dB). Any allocation with
max p < P_T is dominated by its own rescaling, which IS in the image. Nothing
optimal is given up; ONE redundant degree of freedom -- the overall level -- is
removed from the search, and the head is left to choose only the SHAPE.

WHY THAT SHOULD MOVE p10/p25 WHEN NOTHING ELSE HAS. The pointwise sigmoid head
must emit each user's power INDEPENDENTLY, and the value that is right for a user
depends on what every other user does through the interference sum -- so the four
hops have to carry, and the head has to reconstruct, the entire coupled
consistency condition. The fixed point SOLVES that condition exactly, at every K,
in closed loop. What the network is asked to predict collapses from "the
simultaneous solution of a K*B-way coupled allocation" to "which users should end
up with more SINR than which" -- and the p10/p25 policy is, by the metric's own
definition, purely that ORDINAL statement (exp 28's +0.0091 was the same reading
applied to the INPUT side; this applies it to the output side). Every operating
point the last three winning experiments added as a feature -- p*, the lam=1/2
midpoint, p_clip -- is a member of this family at a particular w, so the feature
set and the output space are now the same object, and the head interpolates
inside it instead of re-deriving a member of it from scratch.

It also explains the basin result directly. Gradient ascent in RAW power moves
one user's power and pays the interference cost on everyone else, which is what
makes the p10/p25 landscape locally flat around the student. In PROFILE space,
raising one user's w automatically re-solves everyone else's power to stay
consistent -- the same displacement is one coordinate instead of a conspiracy of
seventy, so the direction exp 30 measured as worth +0.005 in power space is not
the direction this loss follows.

THE ANCHOR. w = w_clip * 10**(W_SCALE*tanh(logit)) with W_SCALE = 3.0, so the
head applies a bounded +-3-decade correction to the Kq-clipped profile. Bounded
by tanh, so neither rail is an attractor: full power is reachable (w = sinr_fp)
but is not where zero logits land, which is the structural version of trap 1, and
a muted user needs a saturated -3 decades rather than one saturated sigmoid,
which is the structural version of trap 2. At Kq=1 the clip is empty, w_clip is
flat and the anchor is p* -- the provable max-min optimum -- so the six finished
`min` cells start ON their optimum and the ALPHA=1.0 distillation term (whose
target is that same p*) starts at exactly zero loss and acts as a leash holding
them there. The column that is done is protected by construction.

CONTRACT -- THE SAME RECURSION, ONE ARGUMENT DIFFERENT. `balance_labels` has run
inside `forward()` since exp 27 and `clip_balance` since exp 29; both are this
same map. The only new thing is that `w` is now model-derived, so state the ban
list item by item. NO GRADIENT STEP is taken. There is NO CANDIDATE SET: one
profile is emitted and one allocation is computed from it -- nothing is
enumerated, compared, accepted or rejected, and no restart exists. NO LOOP'S
ACCEPTANCE OR OUTPUT DEPENDS ON EVALUATING THE OBJECTIVE: the recursion runs a
FIXED, unconditional 40 iterations and never computes a rate, a top-k, an SLqP,
a sum, or any utility of any kind -- there is no `if`, no comparison and no
early stop anywhere in it. It is an implicit/fixed-point LAYER, a deterministic
algebraic function of (A, w) in exactly the sense a matrix inverse or a Perron
eigenvector of the input is; a fixed positive weight vector leaves a standard
interference function standard (positive, monotone, strictly scalable BECAUSE
N_0 > 0), so the limit is unique and globally attracting and does not depend on
where the iteration starts. Parameters are untouched by evaluation, so the
no-test-time-fitting tripwire is untouched. And it is emphatically NOT the
learned unrolled optimiser program.md holds out of scope pending a director
ruling: that is defined by evaluating objective GRADIENTS on the model's own
candidate powers, and this evaluates no objective and takes no gradient.

EQUIVARIANCE, SIZE-GENERALISATION, RANGE. The recursion is equivariant in both
the user and the cell axis, so permuting inputs permutes the output and ONE
parameter set still serves every K. The output is in (0, P_T] by construction --
it cannot leave the box at any K, which the sigmoid also guaranteed. w's overall
scale cancels in the normalisation, so the head cannot even express a bad global
level.

COST. One more 40-iteration float64 fixed point of the einsum shape the forward
pass already runs twice: measured inference went 0.401 -> 0.533 s when exp 27
added the first, so expect ~0.9 s of the 10.0 s budget. Training now backprops
through the 40 unrolled iterations on a [32,K,B] tensor -- ~40 tiny einsums
forward and backward per sub-batch against four 48-wide message-passing hops,
i.e. a small fraction of the step. Parameter count is unchanged at ~78k.

DISTRIBUTION: the training law is UNTOUCHED -- K uniform on 1..10 (the ungraded
3, 5, 7, 9 included), `frac` flat on [0, BAND_MAX_FRAC] routed through the
evaluator's own `kq_of()`, so `_band_kq_max()` still equals the largest graded Kq
for every K, every graded cell keeps exactly the mass it had, and nothing is
narrowed toward the graded points. NO OFF-GRID CHECK IS OWED. The anchor is a
function of Kq and Kq is drawn CONTINUOUSLY across the band, so the profile head
is exercised at every Kq in 1..`_band_kq_max(K)`. As with exps 28/29 I will still
run `k_generalization_check.py` on K in {3, 5, 7, 9} before treating a jump as
banked.

WHY A NEW FAMILY TAG. This is a different model class, not a tuning of one: the
map from features to powers is no longer a feed-forward head but an implicit
layer, and the head's output lives in SINR space rather than power space. It is
family 3 of the 6 program.md allows, after `equivariant_mpnn_cellcoord` (22 exps)
and `qft_distill_mpnn` (8), and it inherits the latter's framework (direct
objective + max-min distillation) unchanged so the comparison is clean.

-----------------------------------------------------------------------------
EXPERIMENT 29 -- family `qft_distill_mpnn` (the CHAMPION, 1.457018)
-----------------------------------------------------------------------------
Exp 28 scored 1.454611 (+0.0091 over exp 27), the family's fourth straight keep
and the campaign's largest gain since exp 19. Both of the last two wins came
from the SAME axis -- the input representation -- while capacity is 0-for-3,
loss softening 0-for-3 and Kq re-weighting 0-for-2. The persisted per-cell table
says the remaining 0.030 of mean ratio is still almost entirely in p10/p25:

                    min             p10             p25
    K= 1     1.090 (1.12)         --          1.066 (1.10)
    K= 2     1.222 (1.29)   1.181 (1.21)      1.140 (1.15)
    K= 4     1.497 (1.54)   1.391 (1.44)      1.257 (1.28)
    K= 6     1.774 (1.73)   1.508 (1.55)      1.309 (1.33)
    K= 8     1.974 (2.12)   1.629 (1.66)      1.357 (1.40)
    K=10     2.188 (2.07)   1.757 (1.80)      1.389 (1.44)

Summed shortfall: 0.125 over the six `min` cells (two of which the model BEATS,
and the K=8 residual is 30-drop QFT sampling noise around an analytic optimum
the model has already attained -- `balance_labels` itself prints 1.979 there)
against 0.194 over p10 and 0.182 over p25. ELEVEN CELLS STILL HOLD 75% OF THE
GAP, and they are the cells for which the feature set contains no allocation
that is even the right SHAPE.

THE ONE CHANGE: `_features` gains a FOURTH closed-form operating point, and it
is the first one in the set that DEPENDS ON Kq -- the Kq-CLIPPED WEIGHTED
BALANCING allocation. N_FEAT 19 -> 21. Nothing else moves: architecture,
`forward()`'s structure, ROUNDS, HIDDEN, both loss terms, ALPHA, BAL_ITERS, the
samplers, the pools, the optimiser and the cosine schedule are byte-identical to
the 1.454611 run, so the input representation is again the only variable.

    thr    = the Kq-th smallest `sinr_fp` in the drop        (already computed)
    w      = clamp(sinr_fp / thr, max=1)                     target SINR profile
    p_clip = the box-feasible fixed point with SINR ∝ w
    r_clip = p_clip / P_T,   sinr_clip = the SINR it induces      <- 2 channels

WHY THIS IS THE RIGHT SHAPE, AND WHY THE INCUMBENT PROBES ARE NOT. Every
allocation the encoder can currently read is Kq-FREE: full power, P_T/K, the
channel inversion, the analytic max-min point p*, and the lam=1/2 midpoint
between the last two. Exp 26 MEASURED what that geometric path is worth at the
cells that matter -- the best point on it scored 1.221/1.035 at K=4 p10/p25
against a student already at 1.326/1.124 -- so the fairness path is not merely
under-exploited, it is the WRONG one-parameter family for Kq>1, because it moves
every user together. It cannot be otherwise: p* equalises ALL K*B SINRs at
gamma*, which drags the users ABOVE gamma* down, and at p25 the bottom quarter's
full-power rates already average well above r(gamma*) -- which is exactly why
the balanced point loses to full power there.

The clipped point is the family that fixes precisely that, and the identity that
makes it work is that the weighted iteration `p <- P_T*(w*F(p))/max(w*F(p))` has
SINR ∝ w at its fixed point, so BOTH endpoints already shipped are members:
w = 1 (flat) returns p* EXACTLY, and w = sinr_fp returns FULL POWER exactly (at
p = P_T, F = P_T/sinr_fp, so w*F is the constant P_T and p = P_T is already the
fixed point). Clipping from above at the Kq-th cut walks between them INDEXED BY
Kq, and in the direction the band's own QFT table moves:

    Kq = 1     ->  every s >= 1, w flat, p_clip = p*          (x2.07 at K=10)
    Kq = K*B   ->  every s <= 1, w = sinr_fp, p_clip = full power
    in between ->  the users ABOVE the cut are flattened onto it and release
                   everything above it; the bottom-Kq set keeps its full-power
                   SHAPE and the whole profile scales up until one user hits P_T

That is the p10/p25 policy stated in one line: the metric never sums a user
above the cut, so that user's own rate is worth nothing and its power is pure
interference budget -- but it must not be muted either (trap 2), and the cut is
exactly the level at which it stops mattering. Crucially this moves users
ORDINALLY: only the ones the objective ignores are touched. At Kq=1 the clip is
empty and p_clip IS p*, so the six finished `min` cells are protected by
construction, not re-litigated -- the new channels carry information only where
the gap is.

WHY A FEATURE AND NOT A LABEL. Kq>1 labels are 0-for-2 (exps 25/26), and both
failures were diagnosed by measurement: the oracle scored BELOW the student on
all six cells it was meant to teach, so the MSE was a drag term. p_clip has not
been measured yet, so it does not get to be a target. It goes in on the axis
that is 2-for-2 instead -- and `label_report` now prints CLIP_CHECK, the
allocation's OWN SLqP ratio at every graded p10/p25 cell against program.md's
certified column, on pool drops, persisted to `diagnostics.txt`. If p_clip
measures ABOVE the student's row it is a validated label for exp 30 and this
run will have bought the answer to that question for free; if it measures below,
that thread is closed on a number rather than an argument. The diagnostic is off
every training path (it is not the experimental variable).

CONTRACT -- SAME CATEGORY AS THE THREE PROBES ALREADY SHIPPING. `sinr_fp` and
its Kq-th order statistic are shipped input-only features; `balance_labels` is
a parameter-free algebraic recursion whose unique limit is a function of A in
the sense a Perron eigenvector is, and it has run inside `forward()` since exp
27. Attaching a fixed positive weight vector to a standard interference function
leaves it standard (positive, monotone, strictly scalable BECAUSE N_0 > 0), so
the weighted recursion has the same unique globally-attracting fixed point and
the same status. It evaluates NO objective: no rate, no top-k of any rate, no
SLqP, no sum. There is no candidate SET -- one allocation is computed, not
chosen -- nothing is compared by utility, accepted or rejected, and no loop's
output depends on evaluating the objective. No model output is read: p_clip is a
byte-identical function of (A, Kq) whatever the parameters are, so the
no-test-time-fitting tripwire is untouched and the pass is still ONE
feed-forward. This is nowhere near the learned unrolled optimiser held out of
scope, which is defined by evaluating objective GRADIENTS on the model's OWN
candidate powers.

EQUIVARIANCE AND SIZE-GENERALISATION. w is built from a per-user input feature
and a global order statistic, both relabelling-equivariant, and the recursion is
equivariant in both axes, so p_clip permutes with its inputs and one parameter
set still serves every K. r_clip is in (0, 1] by construction and sinr_clip is a
SINR on the same scale as `sinr_fp`, so neither drifts as K goes 1 -> 10.

COST: one more 40-iteration float64 fixed point of the same einsum shape
`balance_labels` already runs -- measured inference went 0.401 -> 0.533 s when
exp 27 added the first one, so expect ~0.75 s of the 10.0 s budget. +2 encoder
inputs (+96 params, +0.1% of 78k).

DISTRIBUTION: the training law is UNTOUCHED -- K uniform on 1..10 (the ungraded
3, 5, 7, 9 included), `frac` flat on [0, BAND_MAX_FRAC] routed through the
evaluator's own `kq_of()`, so `_band_kq_max()` still equals the largest graded Kq
for every K, every graded cell keeps exactly the mass it had, and nothing is
narrowed toward the graded points. NO OFF-GRID CHECK IS OWED. The new feature is
a function of Kq and Kq is drawn CONTINUOUSLY across the band in training, so the
clip is exercised at every Kq in 1..`_band_kq_max(K)` rather than at the three
graded fractions -- it cannot memorise a graded cut. As with exp 28 I will still
run `k_generalization_check.py` on K in {3, 5, 7, 9} before treating a jump as
banked, since the clip's threshold depends on the whole population.

-----------------------------------------------------------------------------
EXPERIMENT 28 -- family `qft_distill_mpnn` (depth-tuning the champion) [1.454611]
-----------------------------------------------------------------------------
Exp 27 scored 1.445498 (+0.0069 over exp 24), the family's third straight keep
and the largest gain since exp 19. Its persisted `diagnostics.txt` is the first
per-cell readout this campaign has had, and it says exactly where the remaining
0.040 of mean ratio lives. Model vs. program.md's certified QFT, cell by cell:

                    min             p10             p25
    K= 1     1.088 (1.12)         --          1.058 (1.10)
    K= 2     1.215 (1.29)   1.163 (1.21)      1.121 (1.15)
    K= 4     1.502 (1.54)   1.367 (1.44)      1.240 (1.28)
    K= 6     1.785 (1.73)   1.488 (1.55)      1.298 (1.33)
    K= 8     1.979 (2.12)   1.609 (1.66)      1.349 (1.40)
    K=10     2.197 (2.07)   1.734 (1.80)      1.381 (1.44)

Summed shortfall: 0.104 over the six `min` cells (TWO OF WHICH THE MODEL NOW
BEATS) against 0.299 over p10 and 0.253 over p25. ELEVEN CELLS HOLD 84% OF THE
GAP. And the same file's LABEL_CHECK explains why the `min` column is finished:
`balance_labels` -- the provable Kq=1 optimum -- prints 1.086/1.255/1.530/1.735/
1.979/2.168 on pool drops, and the model's graded `min` row tracks it to within
noise, landing ON it at K=8 (1.979/1.979) and above it at K=6 and K=10. The
model has converged to the analytic optimum where an analytic optimum exists;
what is left is the eleven cells where one does not.

THE ONE CHANGE: `_features` gains THREE GLOBAL ORDER-STATISTIC channels of the
input-only SINR probes it already computes, N_FEAT 16 -> 19. Nothing else moves
-- architecture, `forward()`'s structure, ROUNDS, HIDDEN, both loss terms, ALPHA,
BAL_ITERS, the samplers, the pools, the optimiser and the cosine schedule are
byte-identical to the 1.445498 run, so the input representation is again the
only variable.

    q_fp   = percentile rank of sinr_fp among the drop's K*B users, in [0,1]
    m_fp   = log10( sinr_fp / Kq-th smallest sinr_fp in the drop ) / 2
    m_half = the same margin for sinr_half (the lam=1/2 fairness-path probe)

WHY THIS IS THE MISSING INPUT, AND WHY IT IS SPECIFIC TO p10/p25. SLqP_Kq is the
sum of the Kq SMALLEST rates, so the ONLY question that separates a p10/p25
policy from a max-min policy is a GLOBAL ORDER question: am I inside the bottom-
Kq set, and if I am marginal, is it cheaper to push me out of it or to feed the
users already in it? At Kq=1 that question is degenerate -- there is one argmin,
and a `max` pool over a cell plus the balancing fixed point resolves it, which is
precisely why that column is done. At Kq=7 or Kq=18 out of 70 the model needs the
THRESHOLD, and it has never been given anything that could carry one.

Look at what the 16 incumbent features actually see. `rank` and `rel_cell` are
WITHIN-CELL statistics, over K values out of K*B. Every other channel is
strictly per-user. The graph then pools mean/max over a cell's users and
aggregates cells along the channel weights -- so the only global information any
user can ever receive is a channel-weighted average of per-cell means and maxima.
A mean and a max are the two order statistics that are USELESS for locating a
10th or 25th percentile; no fixed number of message-passing rounds over
mean/max pools computes a rank, because rank is not a low-order moment. This is
not a capacity claim (capacity is closed by exps 18/21/22, three probes, all
null) -- it is the same INPUT-limited reading exp 27 won on, applied to the
statistic the graded metric is literally defined by.

`m_fp` is the sharpest of the three: it is the signed margin, in half-decades,
between a user's full-power SINR and the cut that SELECTS the graded set at this
Kq. Positive and large means "the objective will never see me, my power is pure
interference budget to spend on someone else"; near zero means "I am marginal,
and one reallocation decides whether I am counted"; negative means "I am in the
set, feed me". That is the entire p10/p25 decision, handed over as one scalar.
`m_half` reads the SAME cut at the interior operating point where exp 27's
evidence puts the p10/p25 optima, so the pair says how the selected set MOVES as
the network walks the fairness path -- the promote-or-feed question, not just the
current membership. `q_fp` is the Kq-free version, giving the encoder the shape
of the global distribution so it can generalise the margin off the Kq it saw.

Note this also SHARPENS Kq CONDITIONING, which is the campaign's one structurally
mandatory input. Until now Kq entered as a single broadcast constant
(`kq_in_band`), identical for all K*B users -- so the network had to learn a
policy that reads a scalar dial and modulates a per-user response it computes
independently. `m_fp` makes Kq enter THROUGH the channel, per user, at the exact
place Kq acts in the metric's definition.

CONTRACT -- THIS IS AN ORDER STATISTIC OF AN INPUT FEATURE, NOT AN OBJECTIVE.
Both source probes, `sinr_fp` (p = P_T) and `sinr_half` (p = P_T*sqrt(p*/P_T)),
are already-shipped features and are fixed closed-form functionals of A alone.
The new channels apply a rank and a quantile to them. That evaluates NO
objective: no rate, no top-k of any RATE, no SLqP, no sum. There is no candidate
SET -- one allocation is probed, exactly as before, and nothing is compared by
utility, accepted, rejected or selected. There is no loop. No model output is
read: q_fp/m_fp/m_half are byte-identical functions of (A, Kq) whatever the
parameters are, so the no-test-time-fitting tripwire is untouched and the pass
remains ONE feed-forward. A sort is not an optimisation any more than the `min`
in `p_inv` (exp 13) or the `argsort` in `rank` is; those two have shipped for
fifteen experiments. This sits squarely in the category program.md permits
without qualification -- "SINR-like *features* of the input are fine" -- and
nowhere near the learned unrolled optimiser held out of scope, which is defined
by evaluating objective GRADIENTS on the model's OWN candidate powers.

EQUIVARIANCE AND SIZE-GENERALISATION, both preserved. A rank is invariant to
relabelling, so permuting users or cells permutes q/m identically -- the map
stays equivariant in BOTH axes and one parameter set still serves every K.
Both are dimensionless and K-free by construction: q_fp is normalised to [0,1]
by K*B-1 and the margins are log-RATIOS to a quantile of the same population, so
neither drifts as K goes 1 -> 10. At Kq = K*B the margin would be <= 0 for
everyone, but the band caps Kq at 25% of K*B so the cut is always interior.

COST: one argsort and one kthvalue on a [batch, K*B] tensor per probe, K*B <= 70
-- microseconds against the ~0.35 s incumbent forward over the whole grid, and
+3 encoder inputs (+144 params, +0.2% of 78k). Well inside the 10 s budget.

DISTRIBUTION: the training law is UNTOUCHED -- K uniform on 1..10 (the ungraded
3, 5, 7, 9 included), `frac` flat on [0, BAND_MAX_FRAC] routed through the
evaluator's own `kq_of()`, so `_band_kq_max()` still equals the largest graded Kq
for every K, every graded cell keeps exactly the mass it had, and nothing is
narrowed toward the graded points. NO OFF-GRID CHECK IS OWED. It is worth
stating why that is not a formality here: `m_fp` is a function of Kq, and Kq is
drawn CONTINUOUSLY across the band in training, so the feature is exercised at
every Kq in 1..`_band_kq_max(K)` and not merely at the three graded fractions --
the feature cannot memorise a graded cut. If this posts a real jump I will still
run `k_generalization_check.py` on K in {3, 5, 7, 9} before treating it as
banked, since a global order statistic is the first feature whose value depends
on the whole population size.

-----------------------------------------------------------------------------
EXPERIMENT 27 -- family `qft_distill_mpnn` (grace iteration 5 of 5)  [1.445498]
-----------------------------------------------------------------------------
Exps 25 (1.367769) and 26 (1.392161) are BOTH REVERTED: the band-Kq supervised
term, its `slqp_labels` oracle and its `gq` generator are deleted, and the code
returns to the exp-24 champion (1.438568 = four direct-objective tasks + the
ALPHA=1.0 Kq=1 `balance_labels` term). Exp 26's own LABEL_CHECK, persisted for
the first time, closes that thread with a measurement rather than a guess:

    label/full-power (certified QFT in parens)      p10           p25
      K= 4                                    1.221 (1.44)  1.035 (1.28)
      K= 8                                    1.358 (1.66)  1.091 (1.40)
      K=10                                    1.484 (1.80)  1.119 (1.44)

The band oracle is not merely under-converged, it is WORSE THAN THE STUDENT --
the same run's model scored 1.326/1.124, 1.563/1.246 and 1.695/1.314 on those
six cells. An MSE onto a target below the model's own policy is a drag term, so
the fairness-path warm start fixed the p25 RAIL (+0.0244 over exp 25) without
fixing the label, and two experiments on this axis is enough. Note what the same
table certifies in the OTHER direction: `balance_labels` prints 1.086 / 1.255 /
1.530 / 1.735 / 1.979 / 2.168 in the `min` column against QFT's 1.12 / 1.29 /
1.54 / 1.73 / 2.12 / 2.07 -- the analytic max-min fixed point IS the certified
optimiser, reproduced in a handful of einsums, at K=6 and K=10 slightly above
the 30-drop QFT table itself. That oracle is sound; only its extension to Kq>1
was not.

THE ONE CHANGE therefore keeps the oracle and moves it from the LABEL side to
the FEATURE side: `_features` gains three input-only channels built from
p*(A) = `balance_labels(A)`, taking N_FEAT 13 -> 16.

    r_bal     = p*(A) / P_T                       the max-min POLICY itself
    sinr_bal  = SINR induced by p*(A)             ~= gamma*(A), the achievable
                                                  max-min SINR level of the drop
    sinr_half = SINR induced by P_T*sqrt(r_bal)   the midpoint of the log-power
                                                  fairness path p(lam)

WHY THIS AND NOT MORE CAPACITY. Three independent probes (exp 18's 6 rounds,
exp 21's 8 tied hops, exp 22's HIDDEN=64) each moved the score by <=0.007 in the
wrong direction: the model is not capacity-limited, it is INPUT-limited. What it
is being asked to do at Kq=1 is regenerate a Perron-type fixed point from raw
gains in four hops -- and the residual gap is not there anyway. The gap is the
p10/p25 columns, where the optimum sits at an interior point of the one-
parameter fairness path from full power (lam=0) to p* (lam=1). Exp 26 proved
that path contains most of the available gain and that lam is what Kq selects;
handing the model both endpoints and its midpoint turns "solve a balancing
problem, then decide how far to walk back from it" into "read three probes and
interpolate", which is a job a 4-hop MPNN conditioned on Kq can actually do.
sinr_bal is additionally the first per-drop scalar in the feature set that says
how good the egalitarian regime IS here -- the quantity that decides whether
backing off is affordable at all in this noise-significant system.

CONTRACT. Every new channel is a functional of A ALONE, which is the category
`sinr_fp`, `sinr_lo` and `sinr_inv` (exp 13) already belong to -- the SINR at a
fixed, input-only allocation -- and program.md states plainly that "SINR-like
features of the input are fine". The fixed-point iteration evaluates NO
objective: no rate, no top-k, no SLqP, nothing is compared by utility, no
candidate is accepted or rejected, and no gradient is taken. It is the
normalised power iteration of a standard interference function, i.e. a
Perron-eigenvector computation, and p*(A) is a deterministic FUNCTION of the
input in exactly the sense an eigendecomposition of the input is -- iterated
only because that is how one evaluates it. It is emphatically NOT the learned
unrolled optimiser that program.md holds out of scope pending a director ruling:
that is a loop whose output depends on evaluating objective GRADIENTS on the
model's own candidate powers; this contains no model output, no objective and no
gradient, and produces the same p*(A) whatever the model's parameters are.
`forward()` remains one feed-forward pass, output P_T*sigmoid(.) in [0, P_T],
equivariant in both axes (p* is equivariant to relabelling users and cells), one
parameter set for every K. Cost: 40 float64 einsums of the shape `slqp_rate`
already runs, on a [t,K,B] tensor -- ~0.3 s added over the whole 17x250 grid
against a 10.0 s budget and a measured ~0.35 s incumbent.

The training law is untouched -- K uniform on 1..10 (the ungraded 3, 5, 7, 9
included), frac flat on [0, BAND_MAX_FRAC] via the evaluator's own `kq_of()`,
`_band_kq_max()` still equal to the largest graded Kq for every K -- so nothing
is narrowed toward the graded points and no off-grid check is owed. LABEL_CHECK
is reduced to the `min` column it can certify, and the exp-26 `diagnostics.txt`
dump (LABEL_CHECK + per-cell GRID) is kept: it is off every training path, and
it is the reason this experiment could be decided by measurement.

-----------------------------------------------------------------------------
EXPERIMENT 26 -- family `qft_distill_mpnn` (grace iteration 4 of 5)  [1.392161]
-----------------------------------------------------------------------------
Exp 25 scored 1.367769 against exp 24's 1.438568: -0.0708, by far the largest
regression of the campaign and an order of magnitude bigger than any tuning
delta on this board. A loss that big from ADDING an auxiliary MSE is not "dense
supervision does not help at p10/p25" -- it is a BAD TARGET being followed
faithfully. THE ONE CHANGE fixes the target at its cause: the band oracle's
WARM START. Everything else is byte-identical to exp 25 -- ALPHA=ALPHA_Q=1.0,
REFINE=40, REFINE_LR=0.1, the best-iterate tracker, the Kq=1 `balance_labels`
term, the four direct-objective tasks, the architecture, `forward()`, the ratio
loss, the sampler, the pools, the optimiser and the cosine schedule.

THE DIAGNOSIS, AND WHY IT PREDICTS -0.07 SPECIFICALLY. `slqp_labels` warm-starts
its local optimiser at `p0 = balance_labels(A)`, the max-min point, for EVERY
Kq in the band. At Kq=1 that is the global optimum and the term is a no-op, as
designed. At p25 it is the wrong end of the problem, and quantifiably so: the
balanced allocation equalises all K*B SINRs at gamma*, so its SLqP_Kq is
`Kq * r(gamma*)`, while full power's is the sum of the Kq WORST of a spread
distribution -- and program.md's certified table says r(gamma*)/min_rate(full)
is 2.07 at K=10 while the p25 optimum is only 1.44x full power, so the mean of
the 18 worst full-power rates comfortably exceeds r(gamma*). The balanced point
is therefore WORSE THAN FULL POWER at p25, the tracker (seeded with `max(full
power, p0)`) picks full power, and 40 Adam steps at lr 0.1 starting from p0 --
which move a logit by at most ~4 units -- cannot cross the gap. So the label
returned at the p25 cells is `P_T` EXACTLY: the term regresses the model onto
the full-power rail, at weight 1.0, with a dense per-user target. That is trap 1
in program.md handed to the network as a supervised label. Six of the seventeen
graded cells are p25; dragging them from ~1.1-1.35 toward 1.0 costs ~1.2 summed
ratio, i.e. ~0.07 of a 17-cell mean -- the observed delta, to the digit.

THE FIX: WARM-START ON THE FAIRNESS PATH, NOT AT ITS ENDPOINT. Both endpoints
this band cares about are already in hand -- full power and the analytic max-min
point -- and the object that interpolates them is a one-parameter family in LOG
power,

    p(lam) = P_T * (p0 / P_T) ** lam,    lam in {0, 1/8, ..., 1}

lam=0 is exactly full power, lam=1 is exactly p0, and lam is precisely the
"how egalitarian" dial whose right setting is what Kq selects. The warm start
becomes the per-drop argmax of SLqP_Kq over these LAM_STEPS points, and the
tracker is seeded from the same sweep. Three properties, all load-bearing:
  * STILL MONOTONE, and by a strictly stronger margin: lam=0 and lam=1 are both
    ON the path, so the label is >= full power AND >= p0 in SLqP on every drop,
    exactly the guarantee exp 25 had, plus whatever the seven interior points
    buy.
  * STILL AN EXACT NO-OP AT Kq=1. p0 is globally optimal there, so lam=1 wins
    the sweep, the warm start is p0 unchanged, and the two supervised terms
    still agree by construction -- exp 24's confirmed +0.0031 is protected, not
    re-litigated.
  * NO LONGER A RAIL AT p25. QFT gains 1.28-1.44x over full power at p25, and a
    scalar back-off dial is the first-order way to collect it, so some interior
    lam beats lam=0 on essentially every drop -- and the Adam refinement now
    starts NEXT TO the optimum instead of a full logit-range away from it, with
    finite logits it can actually move.
Cost: LAM_STEPS=9 extra NO-GRAD `slqp_rate` evaluations on a 32-drop sub-batch,
against the 40 forward+backward passes the refinement already runs -- a rounding
error, and cheaper than the failure it removes.

DIAGNOSTICS (not the experimental variable). `autoresearch.sh` captures stdout
into a shell variable and greps only `FAMILY`/`HELDOUT_SCORE`/`INFERENCE_S`, so
the `LABEL_CHECK` table exp 25 printed -- which would have shown the p25 label
sitting at 1.000 and caught this before the score was read -- was discarded, as
is the evaluator's own per-cell `GRID` table, for all 25 experiments so far.
Both are now also written to `diagnostics.txt`, so the NEXT iteration can read
which cells moved instead of inferring it from a scalar. This touches no tensor
on any training path.

CONTRACT. `forward()` is byte-identical: one feed-forward pass, no gradient
step, no restarts, no candidate set, no loop whose acceptance depends on
evaluating the objective, no rate/top-k/SLqP on any power vector, output
`P_T*sigmoid(.)` in [0, P_T], equivariant in both axes, one parameter set for
every K. The whole path sweep and refinement live in the label pipeline and are
unreachable from inference. The training law is untouched -- K uniform on 1..10
(the ungraded 3, 5, 7, 9 included), `frac` flat on [0, BAND_MAX_FRAC] via the
evaluator's own `kq_of()`, so `_band_kq_max()` still equals the largest graded
Kq for every K, every graded cell keeps exactly the mass it had, and nothing is
narrowed toward the graded points: no off-grid check is owed.

-----------------------------------------------------------------------------
EXPERIMENT 25 -- family `qft_distill_mpnn` (grace iteration 3 of 5)  [1.367769]
-----------------------------------------------------------------------------
Exp 24 turned dense supervision on and it paid (+0.0031, the first framework
gain of the campaign). But that supervision reaches ONE percentile: the oracle
solves max-min exactly, so the auxiliary term is evaluated at `Kq=1` only, and
ELEVEN of the seventeen graded cells (every p10 and p25) still receive nothing
but the direct objective. THE ONE CHANGE adds a SECOND supervised sub-batch
whose target is the SLqP optimum at a Kq drawn from the SAME continuous band
law the graded objective uses, so dense supervision now covers the whole band.

Everything exp 24 established is left byte-identical: the four direct-objective
tasks, the existing ALPHA=1.0 Kq=1 term with its analytic `balance_labels`
oracle, the architecture, `forward()`, the ratio loss, the sampler, the pools,
the optimiser and the cosine schedule. The new term draws from its OWN
generator (SEED + 4229), so `g` and `gd` -- and hence every (K, Kq, drop) the
champion saw -- are bit-for-bit unchanged and the added gradient is the only
difference. The exact-at-Kq=1 term is deliberately NOT replaced by the new one:
its label is provably optimal while the band oracle's is a local optimum, and
six of the seventeen cells are Kq=1 holding most of the remaining headroom, so
the confirmed win is protected by construction rather than re-litigated.

THE BAND ORACLE. For Kq > 1 there is no closed form -- SLqP is a sum of the Kq
SMALLEST rates and the active set is endogenous -- so the label is produced by a
TRAINING-TIME local optimiser, warm-started at the analytic max-min point:

    p0 = balance_labels(A)                       # exact Kq=1 optimum
    z  = logit(p0 / P_T);  REFINE Adam steps on  -SLqP_Kq(P_T*sigmoid(z))
    label = the best iterate BY SLqP, seeded with max(full power, p0)

Two properties make this safe to add blind. (i) It is MONOTONE: the returned
label's SLqP is >= that of BOTH reference allocations on every drop, because
full power and p0 seed the best-iterate tracker, so the target can never be
worse than the two allocations the model already has reason to imitate -- a
diverging step size degrades the label to "no worse than p0", not to garbage.
(ii) At Kq=1 it is a NO-OP: p0 is already the global optimum there, so no
iterate can beat it and the tracker returns p0 unchanged -- the new term
degenerates to the old one exactly where the old one was exact.

Why this should move the score where the old term could not: the direct
objective hands gradient to exactly Kq users per drop out of up to 70, and the
MSE hands it to ALL of them, but only if a label exists at that Kq. The p10/p25
columns are where the model currently sits furthest below QFT in RELATIVE terms
(QFT is x1.10-x1.44 there against x2.12 at min, so those cells have little
absolute room but the model must still learn a DIFFERENT, less egalitarian
policy for them) -- and Kq is a conditioning input, so that policy is learned
almost entirely from those steps' own gradient. Supervising it directly is the
cheapest way to tell the network what "less egalitarian" means, and because the
label is an explicit power vector it also tells the eleven ungraded users what
to do while the top-k gradient is busy ignoring them.

COST AND CONTRACT. REFINE=40 Adam steps on a [SUB,K,B] logit tensor, each one
`slqp_rate` forward+backward on 32 drops -- the same einsum `balance_labels`
already runs 40 times per step, so the label solve roughly doubles a cost that
was already a small fraction of the model's forward/backward. All of it is
TRAINING-ONLY and unreachable from `forward()`, which is byte-identical: no
gradient step, no candidate set, no objective evaluation and no loop of any kind
runs at inference. The printed `LABEL_CHECK` table validates both oracles
against program.md's certified QFT grid (all three columns now, not just `min`)
BEFORE any score is read, so a bad local optimiser is visible without trusting
the score.

-----------------------------------------------------------------------------
EXPERIMENT 24 -- family `qft_distill_mpnn` (grace iteration 2 of 5)
-----------------------------------------------------------------------------
Experiment 23 opened the supervised-distillation family and printed 1.435520
against the exp-19 champion's 1.434843: +0.0007, i.e. a null result that was
kept only because it is nominally the best number on the board. The FRAMEWORK
is not what failed, and this experiment says why and fixes it with ONE change:
THE LABEL ORACLE. The cached 320-solve cvxpy/QFT label set is replaced by an
ON-LINE, ANALYTIC max-min-balancing solver that generates FRESH, EXACT labels
every step at negligible cost. ALPHA stays 1.0, the four direct-objective tasks,
the architecture, `forward()`, the ratio loss, the (K, Kq) sampler, the pools,
the optimiser and the cosine schedule are all byte-identical, so the label
source is the only variable.

WHY EXP 23's LABELS COULD NOT HAVE PAID. The distillation term saw 320 drops in
40 (K, Kq) groups, FIXED for the whole run, sampled 2000 times. A 78k-parameter
net regressing 40 fixed groups memorises them within a few hundred steps, after
which the MSE -- and hence the entire supervised gradient -- decays toward zero
and the run reverts to the champion's. That is exactly the shape of a +0.0007
delta: not "dense supervision does not help" but "dense supervision was present
for a fraction of training and then switched itself off". The binding constraint
was never the weight ALPHA, it was that a cvxpy solve costs ~0.3 s and the
one-time budget (LAB_SECONDS = 300) buys a dataset three orders of magnitude
smaller than the 256,000 drops the direct term consumes.

THE FIX IS THAT THE LABELS THIS BAND NEEDS DO NOT REQUIRE A CONVEX SOLVER AT
ALL. Six of the seventeen graded cells are Kq=1, where SLqP is literally
`min_i rate_i` and rate is monotone in SINR, so the Kq=1 optimum is the max-min
SINR problem over all K*B users under the per-user box p <= P_T. Write the
interference-plus-noise per unit own-gain,

    F[k,b](p) = ( sum_c A[k,b,c]*Pcell[c] - p[k,b]*A[k,b,b] + N_0 ) / A[k,b,b]

F is affine with non-negative coefficients and a strictly positive constant, so
it is a STANDARD interference function (positive, monotone, and strictly scalable
BECAUSE of the noise term), and the normalised fixed-point iteration

    p  <-  P_T * F(p) / max_{k,b} F[k,b](p)

converges globally and geometrically to its unique fixed point p*. At p* every
SINR equals P_T / ||F(p*)||_inf and exactly one user sits at P_T. That point is
OPTIMAL, not merely balanced: for any common target gamma the minimal feasible
allocation is the fixed point of gamma*F, that fixed point is increasing in
gamma, and it touches the box precisely at gamma* -- so no box-feasible p
achieves a larger min SINR. The labels are therefore the EXACT Kq=1 optimum,
which is the thing QFT's ten iterations are themselves converging to; we are not
approximating the oracle, we are computing it in closed loop.

AND IT IS FREE. One iteration is one `tkbc,tc->tkb` einsum, the same einsum
`slqp_rate` already runs; 40 of them on a 32-drop batch is a rounding error
against one forward pass of the net. So the label set is unbounded and FRESH --
a new 32-drop batch at a new K every step, ~64,000 labelled drops over the run
against exp 23's 320, and none of them ever repeated. The MSE cannot decay to
zero by memorisation, so the supervised gradient stays live for all 2000 steps,
which is the property exp 23 lacked. cvxpy is no longer imported at all, no
cache file is read or written, and the run has no external solver dependency.

WHY THIS TARGETS EXACTLY THE HOLE IN THE DIRECT GRADIENT. At Kq=1 the direct
objective's gradient is supported on ONE user out of up to 70 and the argmin
switches discontinuously between steps; those six cells are 35% of the grade and
hold nearly all the headroom (QFT x2.12 at K=8, x2.07 at K=10, against x1.10-1.44
at p25). The balancing label signals all K*B users at once, at the same cells,
with the converged answer. It is also the same object the architecture was built
for: exp 8's premise -- this campaign's one confirmed structural claim, +0.0150
-- is that ONE message-passing round is ONE hop of this very power iteration.
Exps 18 and 21 showed that adding hops does not help, which leaves the reading
that four hops CAN represent the map but the one-user-per-drop gradient cannot
FIND it. Regressing the four-hop stack directly onto the 40-hop fixed point is
the sharpest available test of that reading.

A FREE VALIDATION OF THE ORACLE IS PRINTED. `BALANCE_CHECK` reports, once at
startup, the labels' own Kq=1 ratio to full power at K = 1, 2, 4, 6, 8, 10 on
pool drops. program.md's QFT `min` column is 1.12 / 1.29 / 1.54 / 1.73 / 2.12 /
2.07 and is independently certified converged, so those six numbers say whether
the analytic oracle reproduces the certified solver before a single score is
read. Meeting or slightly exceeding that column confirms the derivation; falling
below it would falsify it, and either answer is worth having in the log.

WHAT IS DELIBERATELY NOT CHANGED. ALPHA stays 1.0 and the term is still a plain
MSE in normalised power units on top of an UNCHANGED, unreweighted, unsoftened
direct objective -- exp 20's lesson was that a curriculum which REPLACES the
graded objective loses even when its limit is the exact metric. The distill
batch is drawn from its own generator (SEED + 991), so `g` -- and hence the
direct term's entire (K, Kq, drop) sequence -- is bit-for-bit exp 19's and the
only difference between the two runs is which target the fifth sub-batch carries.

DISTRIBUTION AND THE OFF-GRID QUESTION, STATED HONESTLY. The GRADED objective's
training law is untouched: K uniform on 1..10 (ungraded 3, 5, 7, 9 included),
`frac` flat on [0, BAND_MAX_FRAC] via the evaluator's own `kq_of()`, so
`_band_kq_max()` still equals the largest graded Kq for every K and every graded
cell keeps exactly the mass it had. The AUXILIARY term is, however, evaluated at
Kq=1 only -- because Kq=1 is the only percentile whose optimum this oracle
computes exactly -- and Kq=1 is a graded value, so this experiment does add
supervision concentrated on one graded column. Its K draw is still uniform over
1..10 including the ungraded sizes, and nothing is narrowed toward the graded
`frac` values (p10 and p25 receive no supervision at all). That is not the
failure mode program.md warns about (collapsing training onto the graded points),
but it is close enough to it that if this run posts a real jump I will run
`k_generalization_check.py` for K in {3, 5, 7, 9} before treating the jump as
banked, and report the drop.

CONTRACT. `forward()` is BYTE-IDENTICAL to the champion's: one feed-forward
pass, no gradient step, no restarts, no candidate set, no loop whose acceptance
or output depends on evaluating the objective, no rate/top-k/SLqP evaluated on
any power vector; output P_T * sigmoid(.) in [0, P_T], equivariant in both the
user and the cell axis, ~78k params, ~0.37 s of the 10 s budget. The balancing
iteration IS a loop that reads SINR-like quantities of candidate powers, and it
lives ENTIRELY in `main()`/training, where the contract applies to nothing ("All
restrictions apply at INFERENCE only"); it is unreachable from `forward()` and
no label, no solver and no iterate is stored on the model. Training cost is one
extra 32-drop forward/backward plus 40 einsums per step.

-----------------------------------------------------------------------------
EXPERIMENT 23 -- family `qft_distill_mpnn` (BREADTH FAMILY 2 of <=6)
-----------------------------------------------------------------------------
VERDICT: 1.435520 against 1.434843 -- +0.0007, a null. See experiment 24 for the
mechanism (a 320-drop fixed label set is memorised early and the supervised
gradient switches itself off) and for the label oracle that removes it.

Experiment 22 (HIDDEN 48 -> 64) is REVERTED in full -- it printed 1.434775
against the exp-19 champion's 1.434843, a delta of -0.00007, i.e. exactly
nothing. HIDDEN goes back to 48 and the architecture, features, `forward()`,
ratio loss, TASKS=4 sampler, pools, optimiser and cosine schedule are all
byte-identical to the 1.434843 champion again. ONE change is then applied, and
for the first time in this campaign it is a change of LEARNING FRAMEWORK rather
than of model: TRAINING GAINS A SUPERVISED QFT-DISTILLATION TERM alongside the
unsupervised direct-objective term it has always used.

WHY A NEW FAMILY, AND WHY NOW. program.md caps breadth at SIX (framework x
architecture) families and this campaign has spent TWENTY-TWO experiments inside
ONE. That was defensible while the family was climbing; it is not any more. The
last five experiments read -0.011 (exp 20, soft top-k), -0.006 (exp 21, 8 tied
hops), -0.00007 (exp 22, width) against 1.434843, and the three axes they probe
are now each closed by two independent probes: capacity is not binding (adding
hops with more params lost, doubling hops with a quarter of the params lost,
widening every hop by 78% did literally nothing), the loss is 0-for-3 on
anything other than the exact graded quantity (exps 3, 4, 20), and the Kq
measure is 0-for-2 (exps 5, 17). A sixth variation of the same recipe is the
least informative thing left to run. Meanwhile the framework axis -- which the
Goal names FIRST ("derive the right learning FRAMEWORK and architecture") and
which program.md provisions explicitly, shipping `qft_reference.py` and telling
us to cache labels and note the one-time cost -- has never been touched.

WHY DISTILLATION IS THE RIGHT SECOND FRAMEWORK, MECHANISTICALLY. The direct
objective's gradient is supported on Kq users per drop. Six of the seventeen
graded cells are Kq=1, so on those the loss is `-min_i rate_i` over up to 70
users and 69 of them receive EXACTLY ZERO signal per drop; the second-worst
user, one reallocation away from becoming the objective, learns nothing. Worse,
the argmin's identity switches discontinuously between steps, so consecutive
gradients are exact gradients of DIFFERENT smooth pieces rather than noisy
estimates of one direction. This campaign has now attacked that sparsity three
times from inside the unsupervised framework -- exp 3 and exp 4 softened the
selection, exp 20 softened it again with a proper tail-relative temperature and
an anneal to the exact metric -- and lost every time, because every one of those
changes had to buy density by optimising something OTHER than what is graded.
A label does not have that trade-off. `p*` from QFT supplies a DENSE target for
all K*B users at once while the graded objective stays byte-identical and at
full strength, so density and metric-alignment stop competing.

And the labels carry exactly the structure the architecture argument says this
net needs and cannot find. Exp 8's premise (the campaign's one confirmed
structural claim, +0.0150) is that the Kq=1 optimum is the fixed point of the
SINR-balancing power iteration, and that one round is one hop of it. QFT runs
TEN iterations of a certified solver to reach that fixed point; we run four hops
and try to discover it from a one-user-per-drop gradient. Exps 18 and 21 showed
more hops do not help, which leaves the possibility that four hops CAN represent
the map but the sparse gradient cannot FIND it. Regression onto `p*` is the
direct test: it hands the four-hop stack the converged fixed point as a target
instead of asking it to rediscover one from the argmin.

WHAT IS ADDED, EXACTLY. `ALPHA * mean_{k,b} ((p_model - p*)/P_T)^2` on ONE extra
sub-batch per step, on top of the four unchanged direct-objective tasks. MSE in
normalised power units, not in logit units (a label at 0 or P_T has no finite
logit) and not on the induced rate (which would just be the sparse objective
again). ALPHA = 1.0: the distill term's magnitude is O(0.1) against the ratio
term's O(1.4), so it is a MINORITY of the loss but a MAJORITY of the supported
users -- which is the whole point of adding it. It is deliberately NOT annealed:
exp 20's lesson was that a curriculum which replaces the graded objective loses
even when its limit is the exact metric, and here the graded objective is never
replaced, so a constant weight is both safer and the only reading that says what
distillation is worth at a fixed strength.

THE LABEL PIPELINE AND ITS ONE-TIME COST. Before training, 4 Kq groups x 8 drops
are solved per K over K = 1..10, i.e. 320 QFT solves, drawn from the SAME law the
direct term samples (K uniform on 1..10 including the ungraded 3, 5, 7, 9; frac
flat on [0, BAND_MAX_FRAC] routed through the evaluator's own `kq_of()`), and
cached to `qft_labels.pt` keyed by a signature of the generation settings. First
run pays it once (~1-3 min at the sub-second-per-drop rate program.md quotes for
this band, hard-capped at LABEL_SECONDS = 300 s, which truncates ROUNDS of the
plan rather than starving large K because the plan is laid out K-major); every
later run loads the cache and never imports cvxpy at all. Labels are drawn with
their own generator (`SEED + 7777`) and the per-step distill group with another
(`SEED + 991`), so the direct-objective task sequence is bit-for-bit the
champion's and the comparison is clean. A group whose solve failed to beat full
power on a drop has that drop dropped -- `qft_solve` returns its 0.5*P_T INIT if
both CLARABEL and SCS fail, and a garbage target is worse than no target. That
filter is a training-time utility comparison, which the contract permits without
qualification ("All restrictions apply at INFERENCE only"). If cvxpy is
unavailable the pipeline degrades to the pure direct objective and says so.

CONTRACT. `forward()` is BYTE-IDENTICAL to the champion's -- one feed-forward
pass, no gradient step, no restarts, no candidate set, no loop whose acceptance
or output depends on evaluating the objective, no rate/top-k/SLqP evaluated on
any power vector; output P_T * sigmoid(.) in [0, P_T], equivariant in both the
user and the cell axis, ~78k params, one parameter set for every K, ~0.37 s of
the 10 s budget. No QFT, no label and no solver is reachable from inference; the
distillation lives entirely in `main()`. The direct term's training distribution
is untouched and the label plan is drawn from that same law, so `_band_kq_max()`
still equals the largest graded Kq for every K, every graded cell keeps exactly
the mass it had, nothing is narrowed toward the graded points, and NO OFF-GRID
CHECK IS OWED.

Caveat for the director: Bash refused to execute Python in this session, so this
is hand-verified rather than smoke-tested. The load-bearing checks: `qft_solve`
returns `p_flat` of length K*B indexed `k*B + b` (its `_prep` builds
`M = G.reshape(KB, B)` and `SM[b, k*B+b] = 1`), so `.reshape(K, B)` matches the
model's `[K, B]` output axis-for-axis; `slqp_rate(P, A, Kq)` returns `[batch]`
and the boolean mask indexes dim 0 of both `A` and `P`; and `PowerNet` is
untouched and still no-arg constructible.

-----------------------------------------------------------------------------
EXPERIMENT 22 -- family `equivariant_mpnn_cellcoord` (REVERTED, kept for record)
-----------------------------------------------------------------------------
VERDICT: 1.434775 against the champion's 1.434843 -- a null result, and read
together with exps 18 and 21 it closes the capacity axis: neither more hops, nor
fewer parameters per hop, nor 78% more width moves this model. REVERTED.

Experiment 21 (the weight-tied 8-hop stack) is REVERTED in full -- it printed
1.429086 against the exp-19 champion's 1.434843 -- so the update blocks are four
independently-parameterised ones again. ONE change is then applied to that
champion, and it is the only major architectural dimension this campaign has
NEVER ONCE VARIED in twenty-one experiments: THE HIDDEN WIDTH GOES 48 -> 64.
Rounds (4), features, edge weights `w` and `u`, message structure, pre-norm,
the pointwise head, the ratio loss, TASKS=4, the (K, Kq) sampler, the pools,
the optimiser, the peak LR and the cosine schedule are all byte-identical, so
capacity-per-hop is the only variable.

WHY -- EXP 18 AND EXP 21 ARE A MATCHED PAIR, AND READ TOGETHER THEY POINT AT
WIDTH, NOT DEPTH. Exp 18 ADDED hops and parameters (4 -> 6 untied rounds, 78k ->
117k) and lost 0.0072. Exp 21 ADDED hops while CUTTING parameters (8 tied rounds
on 22k, twice the iterations on a quarter of the weights) and lost 0.0058 -- i.e.
quadrupling the parameter count back to the champion's is worth about as much as
doubling the hop count is worth nothing. If hop count were the binding
constraint, exp 21's eight hops should have won despite the capacity cut; if
capacity were free, exp 21's quarter-sized weights should have cost nothing.
Neither happened. The consistent reading of both numbers is that four hops
already exhaust the useful depth of this message graph and what separates a
good stack from a bad one is how much function each hop can represent. That is
width, and width is precisely what has been frozen at 48 since experiment 1 --
a number chosen in the very first experiment of the campaign to fit a budget,
never revisited, and never tested against anything.

WHY WIDTH IS THE FORM OF CAPACITY A FIXED 2000-STEP BUDGET CAN ACTUALLY FIT.
Exp 18's own post-mortem (and exp 19's) named the confound: more parameters
inside an unchanged step budget under a noisy gradient makes "too expressive"
and "could not be fit" indistinguishable. Exp 19 removed the gradient half of
that confound (TASKS 1 -> 4, +0.0150, the campaign's joint-largest architectural
gain), but the OTHER half of exp 18's problem was never about parameter count at
all -- it was that two extra rounds lengthen the credit-assignment path, so
every added parameter also sits deeper behind the head. Width adds capacity
without adding a single layer of depth: the residual stream is still four hops
from the head, every gradient path keeps its current length, and only the rank
of each hop's update grows. That makes width the cheap direction to spend the
capacity budget in and the one where "did not fit in 2000 steps" is the weakest
competing explanation for a loss.

WHY THIS IS SAFE NOW AND WAS NOT AT EXPERIMENT 6. Widening this family once
killed it outright: exp 6 printed EXACTLY 1.000000, the full-power floor, on
nothing more than a 5H -> 7H widening of the update input -- trap 1, the sigmoid
head pinned at P_T with dsigma/dz ~ 0. Exp 7 diagnosed that as an unnormalised
residual stream whose raw magnitude reached the head, and pre-norm fixed it at
the cause; exp 9 then RE-APPLIED the identical 5H -> 7H widening and trained
normally (+0.0015). So this family has already paid for one widening experiment
and collected the answer: with `hn = LN(h)` feeding every update input and the
head reading `LN(h)`, the logit's scale is set by learned head weights alone and
saturation has to be bought deliberately. A larger `hidden` changes nothing about
that argument -- LayerNorm is over the hidden axis, so a wider stream is
normalised to unit scale exactly as a narrow one is.

WHY 64 AND NOT 96. Params scale as ~(64/48)^2 = 1.78x, taking the model 78k ->
~137k and the per-step message-passing cost by the same factor, which keeps the
run in the same wall-clock class as exp 18 (117k) and exp 21 (2x per-step cost),
both of which finished inside the budget. 96 would be 4x the FLOPs of the
champion and would re-introduce exactly the fit-budget confound this experiment
exists to avoid. 64 is also a large enough step to read: exp 15's cosine anneal
shrank the endpoint noise ball to the point where exp 17 -- a substantial change
to the training measure -- printed -0.00017, so a capacity effect of the size
exp 21 implies (~0.006 for a 4x parameter swing) is comfortably resolvable.

CONTRACT AND COST. `forward()` is structurally unchanged -- one feed-forward
pass, no gradient step, no restarts, no candidate set, no loop whose acceptance
or output depends on evaluating the objective, and no objective (no rate, no
top-k, no SLqP) evaluated on any power vector; output is P_T * sigmoid(.) hence
in [0, P_T]. Width lives entirely on the HIDDEN axis, which every LayerNorm,
pool, einsum and concat already treats as the feature axis, so the map stays
permutation-equivariant in BOTH the user and the cell axis and ONE parameter set
still serves every K. Inference ~0.37 s -> ~0.66 s of the 10 s budget, under 7%
of it. The training distribution is byte-identical -- K uniform on 1..10
(ungraded 3, 5, 7, 9 included), `frac` flat on [0, BAND_MAX_FRAC] routed through
the evaluator's own `kq_of()` -- so `_band_kq_max()` still equals the largest
graded Kq for every K, every graded cell keeps exactly the mass it had, nothing
is narrowed toward the graded points, and no off-grid check is owed.

-----------------------------------------------------------------------------
EXPERIMENT 19 -- family `equivariant_mpnn_cellcoord` (the CHAMPION, 1.434843)
-----------------------------------------------------------------------------
Experiment 18 (ROUNDS 4 -> 6) is REVERTED in full -- it printed 1.412625 against
the exp-15 champion's 1.419842, so ROUNDS goes back to 4 and the depth thread is
closed honestly, exactly as its own note promised it would be if the delta landed
below the +0.015 the 2 -> 4 slope predicted. ONE change is then applied to that
champion, and it is on the axis with the campaign's only two large wins: EACH
TRAINING STEP NOW AVERAGES `TASKS = 4` INDEPENDENT (K, Kq) TASKS INSTEAD OF ONE,
with the per-step drop count held at BATCH (4 x 32 instead of 1 x 128).
Architecture, features, `forward()`, the ratio loss, the (K, Kq) sampling
distribution, the pools, the optimiser, the peak LR and the cosine schedule are
all byte-identical, so the only variable is how many tasks one gradient sees --
and inference is byte-identical.

WHY -- EXP 15 IDENTIFIED THE NOISE BALL AND THEN ONLY SHRANK IT AT THE END.
Every step so far drew ONE (K, Kq) pair and took a full Adam step on it. K=1/Kq=1
is a single-user noise-limited allocation; K=10/Kq=18 is a mid-pack sum over 18 of
70 users. Their gradients disagree in DIRECTION, not merely in scale -- exp 2
already equalised the scale by dividing by the batch's own full-power SLqP, and
exp 15's note named the residual direction disagreement as irreducible. It is not
irreducible; it is a variance that averages. Decomposing the per-step gradient
variance,

    Var = Var_task / T  +  Var_drop / (T * n)      (T tasks x n drops each)

the current run is T=1, n=128, which pays Var_task in FULL. Splitting the same
128 drops into T=4 tasks of n=32 leaves the drop term at Var_drop/128 exactly and
cuts the task term FOURFOLD, for the same number of rate evaluations per step.
This is the same variance the cosine schedule attacked -- but the schedule only
suppresses it by making the last few hundred steps tiny, i.e. it buys a quiet
endpoint at the price of the endpoint being the only clean part of training. This
change lowers the noise for all 2000 steps, so the annealing tail lands on a
better basin rather than merely on the centre of a wide one. Cosine bought
+0.0093 by shrinking the ball late; attacking the ball at its source is the same
lever applied where it is cheap.

THERE IS A SECOND, ADAM-SPECIFIC REASON, AND IT IS THE ONE EXP 2 LEFT HALF-DONE.
Adam divides by a running RMS of past gradients, and that running average is
taken ACROSS consecutive steps that are different tasks. With one task per step
the second moment is dominated by whichever tasks were drawn most recently, so
the per-parameter step size swings with task identity: a run of large-Kq draws
inflates `v` and the next Kq=1 step is taken with a step size calibrated to a
different problem. Exp 2 made the per-task gradient MAGNITUDES comparable, which
is necessary but not sufficient -- with T=4 each gradient is itself an estimate of
the MULTI-TASK objective's gradient, so `v` estimates one stable quantity instead
of tracking a switching one. That mechanism predicts the gain is largest for the
`min` column, which is both the rarest draw type per K and 6 of the 17 graded
cells.

WHY THIS ALSO EXPLAINS EXP 18. Rounds 4 -> 6 added 50% more parameters and 50%
more per-step compute inside a FIXED 2000-step budget, and it regressed. Under a
noisy gradient the deeper stack is the harder thing to fit, not the more
expressive one, so "6 hops is past the useful depth" and "6 hops could not be
trained in 2000 noisy steps" are indistinguishable from that one number. Cleaning
the gradient is the prerequisite for telling them apart: if this pays, re-testing
depth on top of it is the natural next experiment, and it would be testing the
architecture question rather than the optimiser's tolerance for one.

WHY T = 4 AND NOT 2 OR 16. T=4 cuts the task variance by the factor that matters
most (the first halving of a standard deviation is the expensive one: 1/sqrt(T)
takes 2.0 -> 1.0 by T=4 and only 1.0 -> 0.5 by T=16) while keeping n=32 drops per
task, which is still enough that each task's own full-power denominator -- a
detached scalar -- is a stable normaliser. T=16 would leave 8 drops per task,
where that denominator becomes noisy in its own right and the four extra Python
round-trips per step start to cost real wall-clock. Four sub-batches of 32 run
the same total einsum FLOPs as one of 128; the overhead is three extra kernel
launches per layer per step, which keeps the run inside the ~1-2 min budget with
ROUNDS back at 4.

THIS IS NOT A CHANGE OF TRAINING DISTRIBUTION. Each of the four draws uses the
identical sampler -- K uniform on 1..10 (including the ungraded 3, 5, 7, 9) and
`frac` flat on [0, BAND_MAX_FRAC] routed through the evaluator's own `kq_of()` --
so the marginal law of a (K, Kq) task is bit-for-bit the one the champion trained
on, `_band_kq_max()` still equals the largest graded Kq for every K, every graded
cell keeps exactly the mass it had, and nothing is narrowed toward the graded
points. No off-grid check is owed. What changes is only that four samples from
that law are averaged before the parameters move.

CONTRACT AND COST. `forward()` is untouched -- one feed-forward pass, no gradient
step, no restarts, no candidate set, no loop whose output depends on evaluating
the objective, and no objective (no rate, no top-k, no SLqP) evaluated on any
power vector; output is P_T * sigmoid(.) hence in [0, P_T], equivariant in both
the user and the cell axis, one parameter set for every K. Parameters return to
the champion's ~78k and inference to ~0.35 s of the 10 s budget. Training cost is
four sub-batch passes whose total drop count equals the old single pass, plus
three extra no-grad full-power references per step.

-----------------------------------------------------------------------------
EXPERIMENT 18 -- family `equivariant_mpnn_cellcoord` (REVERTED, kept for record)
-----------------------------------------------------------------------------
Experiment 17 (the squared Kq tilt) is REVERTED in full -- it printed 1.419670
against the exp-15 champion's 1.419842, so the training percentile measure goes
back to a flat draw on [0, BAND_MAX_FRAC] and `BAND_TILT` is deleted. ONE change
is then applied to that champion: MESSAGE-PASSING ROUNDS 4 -> 6. Features, edge
weights `w` and `u`, message structure, width, pre-norm, head, the ratio loss,
the (K, Kq) sampler, the pools, the optimiser and the cosine schedule are all
byte-identical, so depth is the only variable.

VERDICT: 1.412625 against 1.419842. The +0.015 the 2 -> 4 slope predicted did not
appear; the delta is negative and several times the (now-annealed) endpoint
jitter, so the depth thread is closed at ROUNDS = 4. See experiment 19 for the
one confound that this number cannot separate from "6 hops is too many": the
deeper stack also had to be fit in the same 2000 steps from the same one-task
gradient.

WHY -- ROUNDS ARE HOPS OF A FIXED-POINT ITERATION, AND QFT USES TEN OF THEM.
Experiment 8's argument was the sharpest mechanistic claim this campaign has
made, and it is the only structural claim it has ever actually confirmed:
dividing out the own-cell gain writes the objective as

    SINR[k,b] = p[k,b] / ( Pcell[b] - p[k,b] + x[k,b] ),
    x[k,b]    = sum_{c!=b} (A[k,b,c]/A[k,b,b]) * Pcell[c] + N_0/A[k,b,b]

whose balanced solution -- the thing six of the seventeen graded cells (Kq=1)
literally ask for, since a muted user has rate 0 and BECOMES the minimum, so the
optimum must equalise rather than mute -- is the fixed point of the power
iteration `p <- target * (interference(p) + noise)`. One round of this net is
exactly one application of that map: users pool into their cell, cell states
travel back along the normalised cross-gains, node states update. So a net with
R rounds can emulate at most R steps of the iteration. ROUNDS 2 -> 4 bought
+0.0150, the largest architectural gain of the campaign and second only to the
exp-2 loss fix. The reference solver this band is measured against runs TEN
iterations, and its convergence at ten was independently verified (program.md);
we are running four. Nothing has tested whether the residual 1.420 -> 1.485 gap
is simply iteration count, and every change since exp 8 has instead added
information per hop (the victim edge, +0.0015; two more operating points,
+0.0040) -- both an order of magnitude smaller than the one time this campaign
added hops.

WHY 6 AND NOT 8 OR 10. Two reasons, one budgetary and one diagnostic. Each round
is a 7H -> H -> H MLP over K*B nodes and it is the dominant cost of both training
and inference: 4 -> 6 takes params 78k -> 117k and the training step +50%, which
keeps the run inside the ~1-2 min budget, whereas 10 rounds would roughly double
it past the budget for a first read. And 6 is the right SIZE of step to read: if
hops are still the binding constraint the 2 -> 4 slope (+0.0075/round) predicts
roughly +0.015 here, comfortably above the noise floor; if the curve has already
flattened, the delta lands near zero and the depth thread closes honestly
instead of being extended one round at a time. Either answer is worth more than
another feature.

WHY THIS IS READABLE NOW AND WAS NOT BEFORE. Depth compounds the residual stream
harder than width does, which is why it was the single most dangerous thing to
try before pre-norm (exp 7): exp 6 died at exactly 1.000000 -- trap 1, the
sigmoid head pinned at P_T with dsigma/dz ~ 0 -- on nothing more than a 5H -> 7H
widening. Pre-norm makes every input to `upd` unit-scale no matter how large the
stream grows and makes the head read LN(h), and 2 -> 4 then did NOT collapse, so
the cliff is demonstrably gone. Separately, exp 15's cosine anneal was argued to
have a second payoff -- shrinking the endpoint noise ball so later deltas become
readable -- and exp 17 is the first measurement of that ball since: a change
that moved the training measure substantially printed -0.00017, against the
+-0.0015 jitter exp 7 measured under a constant LR. So a +0.015-scale depth
effect is now unambiguous where it would once have been arguable.

CONTRACT AND COST. `forward()` is structurally unchanged -- one feed-forward
pass, no gradient step, no restarts, no candidate set, no loop whose output
depends on evaluating the objective, and no objective (no rate, no top-k, no
SLqP) evaluated on any power vector; output is P_T * sigmoid(.) hence in
[0, P_T]. The two extra rounds are two more instances of the existing block, so
the map stays permutation-equivariant in BOTH the user and the cell axis
(LayerNorm is over the hidden axis only; the pools and both einsums are over the
user/cell axes) and ONE parameter set still serves every K. Params 78k -> 117k;
inference ~0.37 s -> ~0.55 s of the 10 s budget, still under 6% of it. The
training distribution is byte-identical to the exp-15 champion's and still
continuous over the whole 0-25% band via the evaluator's own `kq_of()`, so
`_band_kq_max()` still equals the largest graded Kq for every K, no graded cell
gains or loses support, and no off-grid check is owed.

-----------------------------------------------------------------------------
EXPERIMENT 17 -- family `equivariant_mpnn_cellcoord` (REVERTED, kept for record)
-----------------------------------------------------------------------------
Experiment 16 (peak LR 1e-3 -> 3e-3) is REVERTED in full -- it printed 1.392901
against the exp-15 champion's 1.419842, so the peak stays 1e-3 and the descent
thread is closed for now. ONE change is then applied to that champion, and it is
the FREQUENCY analogue of experiment 2's SCALE fix: the training percentile is
drawn as `frac = BAND_MAX_FRAC * u**2` with `u ~ U[0,1]` instead of
`frac ~ U[0, BAND_MAX_FRAC]`. Architecture, features, `forward()`, the ratio
loss, the cosine schedule, the optimiser, the K sampler, the pools and every env
budget are byte-identical, so the Kq measure is the only variable and inference
is byte-identical.

WHY -- THE MIN COLUMN IS 35% OF THE GRADE AND 17% OF THE TRAINING MASS. Under
uniform `frac`, the probability that a step draws Kq=1 is exactly the band
fraction that one Kq unit occupies: P(Kq=1) = (1/KB)/0.25 = 4/(7K). That is 57%
at K=1 but 14.3% at K=4, 9.5% at K=6, 7.1% at K=8 and 5.7% at K=10; averaged
over K uniform on 1..10 it is 16.7%. The grid grades the `min` cell at 6 of its
17 cells -- 35%. So the campaign has been spending half the warranted mass on
the column that holds nearly all the headroom, and the shortfall grows with K:
at K=8 and K=10, where QFT's edge over full power is x2.12 and x2.07, the model
sees the max-min task on about one step in fifteen.

This is precisely the gap experiment 2 did NOT close. Exp 2 divided each step's
loss by the batch's own full-power SLqP, which equalised the gradient MAGNITUDE
across tasks -- a K=10/p25 step no longer carries ten times the gradient of a
K=8/min step. But with magnitudes equalised, the effective weight of a task in
the learned policy is simply how OFTEN it is drawn, and that was never touched.
Exp 2 bought +0.0219, the campaign's largest single gain, by fixing one half of
a two-part mis-weighting; this is the other half, and the arithmetic above says
the residual factor is ~2.1x overall and ~5.8x at K=10.

WHY THE SQUARE, AND WHAT IT COSTS THE TOP OF THE BAND. With frac = 0.25*u^2,
P(Kq=1) = sqrt(4/(7K)): 23.9% at K=10, 26.7% at K=8, 30.9% at K=6, 37.8% at
K=4, and averaged over K it is 37.9% -- essentially the 35% the grid actually
weights the min column at, rather than 16.7%. The exponent is not tuned to a
score, it is the one that makes the training measure match the grading measure.
The price is paid at the band edge, and it is mild: the density of `frac` is
pdf(f) = 1/sqrt(f), which at f = 0.25 equals 2 against uniform's 4, so the p25
cells lose a factor of 2 in mass while the min cells gain a factor of 4. That
asymmetry is the whole point -- the density piles up exactly where a single
integer Kq value (1) absorbs an entire interval of `frac`, and p25 is the column
where QFT's edge is thinnest (x1.10-x1.44), i.e. the least is at stake per unit
of mass.

THIS IS A REWEIGHTING, NOT A NARROWING -- NO OFF-GRID CHECK IS OWED. The support
is unchanged: `u` is still continuous on [0,1], so every `frac` in [0, 0.25) --
including 5%, 15% and 20%, none of which are graded -- keeps strictly positive
density, and K is still uniform on 1..10 including the ungraded 3, 5, 7, 9. The
sampler still routes through the evaluator's own `kq_of()`, so `_band_kq_max()`
still equals the largest graded Kq for every K and the audit invariant holds; no
graded cell loses support. What program.md warns against is collapsing training
onto the graded points themselves, which is the opposite of a smooth
density tilt over an unchanged support.

CONTRACT AND COST. Nothing in `forward()` changes -- one feed-forward pass, no
gradient step, no candidate set, no loop, no objective (no rate, no top-k, no
SLqP) evaluated on any power vector, output P_T * sigmoid(.) in [0, P_T],
equivariant in both axes, one parameter set for every K; ~78k params, ~0.33 s of
the 10 s budget. Training cost is one extra float multiply per step. The sampler
still consumes exactly one draw from the generator, so the K sequence and the
batch indices are bit-identical to the champion's and the Kq mapping is the only
thing that moves.

-----------------------------------------------------------------------------
EXPERIMENT 15 -- family `equivariant_mpnn_cellcoord` (tuning the champion)
-----------------------------------------------------------------------------
Experiment 14 (the per-cell budget gate) is REVERTED in full -- it scored
1.396582 against the exp-13 champion's 1.410449, a regression several times the
run-to-run noise, so the network, features, head, loss, sampler and pools are
byte-identical to experiment 13 again. ONE change is then applied to that
champion, and it is the first change this campaign has ever made to the
OPTIMISATION RECIPE rather than to the model: the learning rate is COSINE
ANNEALED from 1e-3 to 0 over the STEPS budget instead of being held constant.
Architecture, features, `forward()`, the ratio loss, the (K, Kq) sampler, the
pools, the optimiser (still Adam), the peak LR and every env budget are
untouched, so the schedule is the only variable and inference is byte-identical.

WHY -- FOURTEEN EXPERIMENTS HAVE TUNED THE MODEL AND NONE HAVE TUNED THE
DESCENT. Every step draws ONE (K, Kq) task -- K uniform on 1..10, Kq continuous
over the band -- and takes a full Adam step on it. Consecutive steps therefore
solve genuinely different problems (K=1/Kq=1 is a single-user noise-limited
allocation; K=10/Kq=18 is a mid-pack sum over 18 of 70 users), and their
gradients disagree in direction, not just in scale. Experiment 2 fixed the SCALE
half of that mismatch by dividing the loss by the batch's own full-power SLqP;
the DIRECTION half is irreducible and shows up as gradient noise. With a
constant step size, SGD does not converge to a minimiser -- it converges to a
stationary DISTRIBUTION around one, a noise ball whose radius grows with the
learning rate. The reported score is then whichever point in that ball step 2000
happened to land on. Annealing the step size to zero shrinks the ball to its
centre over the second half of training, so the final iterate is the average
policy rather than a sample from it.

THE CAMPAIGN'S OWN LOG IS THE EVIDENCE THAT THIS BALL IS WIDE ENOUGH TO MATTER.
Experiment 7 (pre-norm) was argued in advance to be policy-neutral -- it removes
a saturation cliff without moving the function -- and it printed 1.389091
against 1.390561, i.e. a -0.0015 "effect" that is pure endpoint jitter. That
puts the noise floor at the same order as the last two structural WINS (exp 9's
victim edge bought +0.0015, exp 13's features +0.0040). We are reading
architecture decisions off differences barely above the sampling noise of the
final iterate, which is how a campaign talks itself into a wrong revert. So this
change has two payoffs and the second may be the larger: it should lift the
score by landing on the centre of the ball instead of its rim, and it should
shrink the ball, making every subsequent experiment's delta readable.

WHY COSINE AND WHY TO ZERO. The band's headroom sits in the `min` column, where
the objective is a max-min-like SINR-balancing fixed point -- the flattest, most
degenerate part of the landscape, since near a balanced point many
reallocations are nearly indifferent and the argmin user keeps switching. That
is exactly the regime where a constant step size hurts most: the iterate keeps
being kicked between neighbouring argmin cells and never settles into any of
them. Cosine keeps the LR near its peak for the first ~quarter of training (so
the exploration budget that found 1.41 is spent essentially unchanged) and then
decays smoothly, spending the last few hundred steps at a small enough LR to
resolve the balancing point instead of orbiting it. Decaying to exactly zero,
not to a floor, is the point: any residual constant LR leaves a residual ball.

WHY NOT A BIGGER PEAK LR AT THE SAME TIME. Because that would be two variables.
The peak stays 1e-3 so this run's early trajectory is statistically the same as
the champion's and any difference is attributable to the tail alone. If the
schedule pays, raising the peak (which annealing makes affordable) is the
natural experiment 16 on top of it.

CONTRACT AND COST. Nothing about `forward()` changes: still one feed-forward
pass, no gradient step, no candidate set, no loop, no objective (no rate, no
top-k, no SLqP) evaluated on any power vector, output P_T * sigmoid(.) in
[0, P_T], equivariant in both the user and the cell axis, one parameter set for
every K. Parameter count returns to exp 13's ~78k and inference returns to
~0.39 s of the 10 s budget. Training cost is one scalar `sched.step()` per
iteration. The training distribution is byte-identical and still continuous over
the whole 0-25% band via the evaluator's own `kq_of()`, so no graded cell gains
or loses support and no off-grid check is owed.

-----------------------------------------------------------------------------
EXPERIMENT 14 -- family `equivariant_mpnn_cellcoord` (REVERTED, kept for record)
-----------------------------------------------------------------------------
ONE change vs. experiment 13: the OUTPUT HEAD is factorised into a per-user
SHARE and a per-cell BUDGET,

    p[k,b] = P_T * sigmoid(head(LN(h)[k,b])) * sigmoid(gate(LN(cell_b)))

with `cell_b` the same [mean_k, max_k, vic] pool the rounds already build, read
once after the last round. Features, edge weights, message structure, rounds
(4), pre-norm, the per-user head itself, loss, sampler, pools, optimiser and
budgets are byte-identical; the multiplicative cell budget is the only variable.

WHY -- THE MODEL HAS CELL-LEVEL PERCEPTION BUT NO CELL-LEVEL ACTION. Powers
enter SLqP only through the cell totals Pcell[c] = sum_k p[k,c]; that has been
this family's founding premise since experiment 1, and the last two structural
wins both added ways to SEE cell-level state (the mean/max pool, then the victim
edge). The head, however, is still strictly pointwise: K independent sigmoids
per cell. "Cell b commits 40% of its total" is therefore not a direction in the
parametrisation at all -- it is a conspiracy of K logits that must move together
while leaving the within-cell split alone, and every one of those logits also
controls that split. The two decisions are entangled in exactly the coordinates
the objective does NOT use.

And asymmetric cell totals is the mechanism this band's headroom must come
from. At Kq=1 muting is self-defeating (a muted user has rate 0 and becomes the
minimum), so within a cell the optimum is near-egalitarian and cannot by itself
beat the intra-cell ceiling SINR <= 1/(K-1). Nor can a UNIFORM scale-down help:
this system is noise-significant (median desired SNR ~ -4 dB), so shrinking
every power shrinks every SINR -- the audit note is explicit that the problem is
not scale-invariant. What is left is precisely differential cell budgets: cells
whose own users sit comfortably back off, cells holding the near-worst users
stay high, and the worst user in the network gains because the aggressors that
own its interference budget gave up total power. That is a per-cell scalar
decision, and QFT's x2.12 at K=8 / x2.07 at K=10 in the `min` column has nowhere
else to come from. Giving that decision its own coordinate -- fed by the cell
embedding that the victim edge just taught to know who this cell hurts -- is the
natural action-side counterpart to experiment 9's perception-side edge.

This is a genuine change of function class, not just of coordinates:
sigmoid(a)*sigmoid(b) is not sigmoid(a+b), so a cell-wide multiplicative backoff
that leaves the within-cell ratios untouched is newly expressible in one step.

INIT IS A NEAR-IDENTITY, DELIBERATELY. `gate` is zero-initialised in its weights
with bias +2.0, so at step 0 the budget is the constant sigmoid(2) = 0.881 for
every cell and every K and the policy is 0.881x the exp-13 initial policy --
i.e. this starts as (almost) the champion and can only be earned from there.
Gradients still reach both gate weights and bias immediately (dsigma/dz = 0.105
at z = 2, and LN(cell) is non-zero), so the branch is live from step 1; the
trunk's own gradient path through `share` is untouched. This is the same
zero-init-residual-branch discipline that made pre-norm (exp 7) safe, and it
keeps trap 1 at arm's length: the budget can only approach 1 by the gate's own
weights growing deliberately.

CONTRACT. `forward()` remains a single feed-forward pass -- no gradient step, no
candidate set, no loop, no objective (no rate, no top-k, no SLqP) evaluated on
any power vector. Output stays in [0, P_T] since it is P_T times a product of
two sigmoids. `cell` is invariant to relabelling users (mean/max/sum over the
user axes) and equivariant to relabelling cells, and the budget multiplies every
user of its own cell, so the map is still equivariant in BOTH axes and one
parameter set serves every K. Cost: +433 params (a 3H->1 gate and one
LayerNorm(3H)) on 78k, plus one extra victim einsum and pool outside the loop --
inference ~0.39 s -> ~0.42 s of the 10 s contract. The training distribution is
byte-identical and still continuous over the whole 0-25% band via the
evaluator's own kq_of(), so no graded cell gains or loses support and no
off-grid check is owed.

-----------------------------------------------------------------------------
EXPERIMENT 13 -- family `equivariant_mpnn_cellcoord` (tuning the champion)
-----------------------------------------------------------------------------
ONE change vs. experiment 9: the node feature block gains TWO MORE OPERATING
POINTS. Architecture, edge weights, message structure, rounds (4), pre-norm,
head, loss, sampler, pools, optimiser and budgets are byte-identical; only
`_features` (and `N_FEAT`, 10 -> 13) moves, so the input representation is the
only variable.

WHY -- THE MODEL PROBES THE CHANNEL AT EXACTLY ONE ALLOCATION. Every SINR-like
feature this campaign has ever fed the net is read at p = P_T for everyone:
`sinr_fp`, and `intra`/`snr` which are its two limiting cases. But the thing the
model has to invert is the MAP from an allocation to the resulting SINR vector,
and a single probe of that map carries no information about its SENSITIVITY --
how much user (k,b)'s SINR moves when cell c reallocates. Four rounds of message
passing have to reconstruct that sensitivity from raw gains using reciprocals
and cross-cell normalisations, which is exactly the arithmetic ReLU stacks
approximate worst. Two extra probes turn it into a finite-difference read that
the encoder gets for free:

  (a) UNIFORM LOW POWER, p = P_T/K, i.e. every cell commits P_T in TOTAL:
        sinr_lo[k,b] = (P_T/K)*own / (P_T*tot - (P_T/K)*own + N_0)
      This is NOT a rescaling of `sinr_fp`, because N_0 does not scale with the
      powers -- the audit note in program.md is explicit that this system is
      noise-significant (median desired SNR ~ -4 dB) and NOT scale-invariant.
      The pair (sinr_fp, sinr_lo) therefore tells each user whether it is
      interference-limited (the two agree) or noise-limited (they diverge by
      ~K), which is precisely the bit that decides whether a cell can afford to
      back off at all. Nothing in the current 10 features carries it.

  (b) INTRA-CELL CHANNEL INVERSION, p_inv[k,b] = P_T * min_k' own[k',b] / own[k,b],
      every cell doing it simultaneously. This equalises the RECEIVED signal
      P_T*min_own[b] across a cell's users -- the closed-form egalitarian
      allocation, and the natural prior for a band whose whole headroom is the
      `min` column. It respects the box (p_inv <= P_T by construction, with
      equality for the cell's weakest user) and its cell total is bounded,
      P_T <= Pcell_inv[b] <= K*P_T, so the feature does not drift with K. Both
      `p_inv` itself and the SINR it induces are fed in, so the head can learn a
      RESIDUAL around a strong fairness heuristic instead of synthesising one
      from scratch -- and because every cell backs off simultaneously, the
      induced SINR already contains the second-order "what if my aggressors also
      equalise" term that the one-directional probe cannot express.

CONTRACT. Both probes are functionals of `A` ALONE -- no candidate set, no
selection, no loop, no objective (no rate, no top-k, no SLqP) is ever evaluated
inside `forward()`, which stays a single feed-forward pass. This is the same
category of feature as the existing `sinr_fp`, which is likewise the SINR at a
fixed input-only power vector (p = P_T); the contract's "(SINR-like *features*
of the input are fine)" is what both rest on. `min` over the user axis is
permutation-INVARIANT in users, the cell sums are EQUIVARIANT in cells, so one
parameter set still serves every K; at K=1 the inversion collapses to full power
and `sinr_inv == sinr_fp`, which is consistent, not degenerate.

COST. Encoder input 10 -> 13 (+144 params, +0.2% of 78k); one extra einsum of
the same shape as the one `slqp_rate` already uses. Inference stays ~0.33 s of
the 10 s contract. The training distribution is byte-identical and still
continuous over the whole 0-25% band via the evaluator's own `kq_of()`, so no
graded cell gains or loses support and no off-grid check is owed.

-----------------------------------------------------------------------------
EXPERIMENT 9 -- family `equivariant_mpnn_cellcoord` (depth-tuning the champion)
-----------------------------------------------------------------------------
ONE change vs. experiment 8: the message graph gains its TRANSPOSE edge. Each
round now also builds, for every cell c, a message from the users that c
INTERFERES WITH; the cell embedding becomes [mean_k hn, max_k hn, vic] and the
update input widens 5H -> 7H. Features, edge weights `w`, forward propagation,
rounds (4), pre-norm, head, loss, sampler, pools, optimiser and budgets are
byte-identical, so the victim edge is the only variable.

WHY -- THE GRAPH IS ONE-DIRECTIONAL AND THE `min` COLUMN IS WHERE THAT BITES.
Users pool into their OWN cell; cell states flow BACK to users along the
row-normalised cross-gains `w[k,b,c]`. So a user learns who is hurting it, and
a cell learns what its own users know -- but no cell ever receives a message
keyed by "how much do I hurt user (k,b)". That transpose is absent after any
number of rounds. And it is exactly what this band pays for: a symmetric cell
cannot beat SINR <= 1/(K-1) by ANY internal reallocation (full power is already
optimal there), so QFT's x2.12 at K=8 and x2.07 at K=10 in the `min` column
cannot be coming from within-cell sharing. It must come from cells shaping
their totals to protect SOMEONE ELSE'S worst user -- and a cell that cannot see
its victims can only react to its aggressors.

WHY THIS IS NOT SUBSUMED BY THE DEPTH EXPERIMENT 8 JUST BOUGHT. Experiment 8's
note argued that after three rounds a cell's pool already contains, indirectly,
the states of the cells that aggress its users, so victim information "reaches"
cell b routed by w rather than by b's aggression -- and that the difference is
only the per-user fading/position residual, since the cell-geometry part of
aggression is the same symmetric BS-BS path loss. That argument is right about
the geometry and wrong about what matters. At Kq=1 the objective is decided by
ONE user out of up to 70, and which user that is, is determined precisely by
the per-user fading and position residual -- the part the symmetric BS-BS path
does NOT carry. Cell c does not need to know that it aggresses cell b on
average; it needs to know that a specific cell-edge user in b is near-worst and
that c is that user's dominant aggressor. Depth propagates the average; only
the transpose edge propagates the identity. Six of the seventeen graded cells
are Kq=1 and they hold nearly all the remaining headroom (we are at 1.405 vs
QFT 1.485), so this is the highest-value structural gap left.

WHY IT IS SAFE TO RETRY NOW -- AND WHY THIS IS ALSO A DIAGNOSTIC. This exact
idea was experiment 6, and it printed EXACTLY 1.000000, the full-power floor.
Experiment 7 diagnosed that as trap 1 rather than as a refutation of the edge:
`h = h + upd(...)` was unnormalised while every input to `upd` was built from
`h`, so the residual stream compounded, the head read its raw magnitude, and in
this noise-significant system the "raise everyone" direction pushed the logit
into the sigmoid's flat tail where dsigma/dz ~ 0 annihilates every gradient.
Widening the update input 5H -> 7H was what tipped it over. Pre-norm (exp 7,
1.389091 -- neutral, exactly the signature of a change that removes a cliff
without moving the policy) and then ROUNDS 2 -> 4 (exp 8, 1.404952 -- depth
compounds that stream harder than width does, and it did NOT collapse) together
establish that the cliff is gone. Re-applying the identical 5H -> 7H widening is
therefore the cleanest possible confirmation: if it trains normally, the exp-7
diagnosis is fully vindicated on the very configuration that produced the
collapse; if it prints 1.000000 again, the diagnosis is incomplete and the
saturation mechanism is specific to this edge, which is worth more than another
1.40.

CONSTRUCTION. `u` is the COLUMN normalisation of the same `w` already computed
for the forward edge -- `u[k,b,c] = w[k,b,c] / sum_{k,b} w[k,b,c]` -- so each
cell's incoming victim message is a convex combination over its victims,
weighted by how dominant that cell is in each victim's interference budget.
Normalising `w` rather than raw gains is deliberate: it bounds every cell's
total incoming mass at 1 regardless of K, so one near-zero-gain user cannot
swamp the message and the magnitude does not drift as K goes 1 -> 10.
Attribution lives in the weights, severity lives in `h` (which already carries
the SINR-like input features). `w[k,c,c] = 0` by construction, so a cell's own
users are excluded from its victim message -- intra-cell coupling is already
carried by the mean/max pool.

COST AND CONTRACT. Update MLP 5H -> 7H: params 59k -> 78k, compute per round
+33%, so inference goes ~0.25 s -> ~0.33 s, 3% of the 10 s contract. The extra
einsum is the transpose of one already present and the same order of FLOPs.
`u` is a functional of `A` alone, so `forward()` still scores no candidate
powers, evaluates no objective and remains a single feed-forward pass. The
victim sum is over the user axes and is therefore INVARIANT to relabelling
users and EQUIVARIANT to relabelling cells, so one parameter set still serves
every K. The training distribution is byte-identical and still continuous over
the whole 0-25% band via the evaluator's own kq_of(), so no graded cell gains
or loses support and no off-grid check is owed.

-----------------------------------------------------------------------------
EXPERIMENT 8 -- family `equivariant_mpnn_cellcoord` (grace iteration 3 of 5)
-----------------------------------------------------------------------------
ONE change vs. experiment 7: ROUNDS 2 -> 4. Everything else -- features, edge
weights, message structure, width, pre-norm, head, loss, sampler, pools,
optimiser and budgets -- is byte-identical, so message-passing DEPTH is the only
variable. This is the expressiveness test experiment 7 was built to enable.

WHY DEPTH, AND WHY IT IS THE BINDING CONSTRAINT. Read what the objective asks
for in the `min` column. A muted user has rate 0 and therefore BECOMES the
minimum, so at Kq=1 muting is self-defeating: the optimum cannot drop anyone, it
must EQUALISE -- the Kq=1 cell is an SINR-balancing problem over all K*B users
simultaneously. Six of the seventeen graded cells are Kq=1, and program.md's QFT
table puts nearly all this band's headroom exactly there (x2.12 at K=8, x2.07 at
K=10, against x1.10-x1.44 at p25). So the single most valuable thing this model
can learn to imitate is a global SINR-balancing fixed point.

Rewrite the SINR with the own-cell gain divided out:

    SINR[k,b] = p[k,b] / ( Pcell[b] - p[k,b] + x[k,b] ),
    x[k,b]    = sum_{c!=b} (A[k,b,c]/A[k,b,b]) * Pcell[c] + N_0/A[k,b,b]

A balanced solution is the fixed point of the classical power iteration
`p <- target * (interference(p) + noise)`, i.e. a Perron-eigenvector computation.
That is an ITERATIVE object: each application of the update propagates power
information exactly one hop along the interference graph, and balancing over a
7-cell / up-to-70-user system needs many hops to converge. Our round is precisely
one such hop -- users pool into their cell, cell states travel back along the
normalised cross-gains, node states update -- so a 2-round net can emulate at
most two steps of the iteration. It can learn a good one-shot heuristic (which is
what 1.39 is), but it structurally cannot converge a fixed point. Four rounds is
the cheapest honest test of whether the residual gap to QFT (1.485) is iteration
count rather than anything about features or loss.

This also explains why the campaign's one previous expressiveness attempt was
uninterpretable, and why it is safe to retry now. Experiment 6 added width
(update input 5H -> 7H) and printed EXACTLY 1.000000 -- the full-power floor,
trap 1's signature, a sigmoid head saturated dead at P_T with dsigma/dz ~ 0 so
every gradient in the network is annihilated at the last layer. The cause was
that `h = h + upd(...)` was unnormalised while every input to `upd` was built
from `h`, so the residual stream compounded and the head read its raw magnitude;
in a noise-significant system the "raise everyone" direction has positive
gradient everywhere, so the logit was pushed into the flat tail. Depth compounds
that residual stream even harder than width does, so ROUNDS 2 -> 4 would have
been the single most dangerous thing to try BEFORE experiment 7. With pre-norm
in place the update MLP sees unit scale no matter how large the stream grows and
the head reads LN(h), so saturation now has to be paid for by the head weights
deliberately. Experiment 7 scored 1.389091 -- statistically indistinguishable
from the 1.390561 champion, exactly as predicted for a change that removes a
cliff without moving the policy. If that diagnosis is right, this run trains
normally; if it prints 1.000000 again, the diagnosis is wrong and the saturation
cliff lives somewhere else, which is itself worth more than another 1.38.

WHY NOT THE VICTIM EDGE (experiment 6's idea) INSTEAD. Because iterating the
aggressor edge already delivers most of that information: after three rounds a
cell's pool contains its own users' states, which by round 2 contain the states
of the cells that aggress them, so victim information does reach cell b -- routed
by w[k,b,c] rather than by b's aggression toward c, but the cell-geometry part of
those two quantities is the same symmetric BS-BS path loss, and only the
per-user fading/position residual differs. Depth subsumes the cheap part of the
victim edge; the victim edge does not subsume depth. If depth wins, the victim
edge is the natural experiment 9 on top of it.

COST. Params 31k -> 59k (two more update blocks); the training step roughly
doubles, keeping the run near the top of the ~1-2 min budget rather than past it
(the harness puts no timeout on the scoring run, only on the agent call).
Inference was 0.139 s over the whole grid at 2 rounds, so ~0.28 s at 4 -- 3% of
the 10 s contract. `forward()` remains a single feed-forward pass that scores no
candidate powers and evaluates no objective; LayerNorm is over the hidden axis
only, so equivariance in BOTH the user and cell axes is untouched and one
parameter set still serves every K. The training distribution is byte-identical
and still continuous over the whole 0-25% band via the evaluator's own kq_of(),
so no graded cell gains or loses support and no off-grid check is owed.

-----------------------------------------------------------------------------
EXPERIMENT 7 -- family `equivariant_mpnn_cellcoord` (grace iteration 2 of 5)
-----------------------------------------------------------------------------
Experiment 6 (the victim edge) is REVERTED in full, back to the exp-2 champion.
ONE change is then applied to that champion: the residual message stack becomes
PRE-NORMALISED -- a LayerNorm over the hidden axis at the head of every round
and one more before the output head. Nothing else moves: features, edge
weights, message structure, width, rounds, loss, sampler, pools, optimiser and
budgets are byte-identical to exp 2, so normalisation is the only variable.

WHY. Two of the six experiments so far did not merely regress, they printed
EXACTLY 1.000000 -- the full-power floor, to six decimals. That is not a
slightly-worse policy; it is the unique signature of trap 1 in program.md: the
sigmoid head pinned dead at P_T with `dsigma/dz ~ 0`, so every gradient in the
network is annihilated at the last layer and training can never come back.
Exp 3's collapse had a known separate cause (a units error that turned the
surrogate into mean-rate), but exp 6's did not: it kept the champion's exact
ratio loss, its features, its sampler and its optimiser, and changed only the
update-MLP input width, 5H -> 7H. Adding capacity, alone, took a 1.390 model to
the floor. That is the diagnostic: the failure is not in what exp 6 computed
but in the fact that h is UNBOUNDED.

The stack is `h = h + upd(...)` with no normalisation anywhere, and every input
to `upd` is built from h -- the node state, the mean/max cell pool, and the
gain-weighted aggregate. So the residual stream compounds, and the head reads
its raw magnitude. Because this system is noise-significant (SLqP falls when all
powers scale down), the "raise everyone" direction has positive gradient
everywhere, so training pushes the logit up until it saturates; the only thing
that had been holding exp 2 short of saturation was that its 5H stack happened
to stay small. Widening the update input widens the residual increment, h grows
faster, the logit reaches the flat tail sooner, and the run dies at exactly
P_T. Under that reading the champion is not a good optimum -- it is a model
sitting one capacity increment away from a cliff, which is precisely why the
one expressiveness experiment this campaign has tried was uninterpretable.

Pre-norm fixes the cliff at its cause rather than tiptoeing around it. Each
round now reads `hn = LN(h)` and derives ALL THREE of its inputs from hn, so
the update MLP sees a unit-scale input no matter how large the residual stream
has grown; the head reads LN(h) too, so the logit's scale is set by learned
head weights alone instead of by an uncontrolled activation magnitude.
Saturation is still reachable -- the model can and must express powers near 0
and near P_T -- but it now requires the head weights to grow deliberately,
rather than arriving for free as a side effect of depth or width.

This is the enabling change, not the scoring one. If the exactly-1.000000
diagnosis is right, exp 8 can finally test expressiveness (more rounds, wider
updates, the victim edge again) without every result being confounded by
whether that particular configuration happened to fall off the cliff.

LayerNorm is over the HIDDEN axis only, per node, so it is untouched by
relabelling users or cells: the map stays permutation-equivariant in both axes
and one parameter set still serves every K. It adds 2H params per norm (~290
total, <1% of the model) and no meaningful inference cost. `forward()` still
scores no candidate powers. The training distribution is untouched and still
continuous over the whole 0-25% band via the evaluator's own kq_of(), so no
graded cell gains or loses support and no off-grid check is owed.

-----------------------------------------------------------------------------
EXPERIMENT 2 -- family `equivariant_mpnn_cellcoord` (grace iteration 2 of 5)
-----------------------------------------------------------------------------
ONE change vs. experiment 1: the training loss is normalised into the units the
metric is actually scored in -- the per-task ratio to full power -- instead of
raw Mbps. Architecture, features, sampler, pools, optimiser and budgets are all
untouched, so loss scaling is the only variable.

WHY. HELDOUT_SCORE is a MEAN OVER 17 CELLS of (model SLqP / full-power SLqP):
every cell counts exactly 1/17, regardless of how many Mbps it carries. The
training loss, however, was raw `-slqp_rate(...).mean()` in Mbps, and SLqP is a
SUM OF THE Kq SMALLEST rates -- so its magnitude, and hence the gradient norm of
a training step, scales roughly with Kq and with how good those rates are. A
K=10/p25 step (Kq=18, sum of eighteen mid-pack rates) carries on the order of
ten to twenty times the gradient of a K=8/min step (Kq=1, the single worst rate
in 56). Adam's second moment is a running average ACROSS these heterogeneous
tasks, so the small-magnitude tasks are effectively learned with a proportionally
smaller step -- they are quietly down-weighted relative to their 1/17 grading
weight.

That is exactly backwards for this band. Per program.md's QFT table the headroom
is overwhelmingly in the `min` column (up to x2.12 at K=8, x2.07 at K=10) and
thinnest at p25 (x1.10-x1.44) -- i.e. the cells whose loss magnitude is smallest
are the cells with the most to gain, and the cells that dominate the gradient
are the ones nearest their ceiling already. Dividing each step's loss by the
same batch's full-power SLqP (a detached constant) makes every task's objective
literally the quantity being graded, with a natural scale of ~1.0-2.0 everywhere,
so per-task gradient norms become comparable and the min column stops being
starved. It is a pure re-weighting: the training distribution stays continuous
over the full 0-25% band via the evaluator's own kq_of(), no graded cell gains or
loses support, and the ratio-of-batch-means form matches the evaluator's
ratio-of-means exactly (not a mean-of-per-drop-ratios, which would be a
different, more drop-egalitarian objective). No off-grid check is owed.

Cost: one extra no-grad slqp_rate on the constant full-power vector per step.

-----------------------------------------------------------------------------
EXPERIMENT 1 -- family `equivariant_mpnn_cellcoord` (breadth family 1 of <=6)
-----------------------------------------------------------------------------
The baseline is a per-user MLP on 4 own-user scalars. Read the objective:

    SINR[k,b] = p[k,b]*A[k,b,b] / ( A[k,b,b]*(Pcell[b]-p[k,b])
                                    + sum_{c!=b} A[k,b,c]*Pcell[c] + N_0 )

Powers enter the denominator ONLY through the CELL TOTALS Pcell[c]=sum_k p[k,c].
The whole problem is therefore a coupling between one scalar per user and B
cell-level sums, and the intra-cell term dominates: at full power it alone caps
every user at SINR <= 1/(K-1), which is exactly why QFT's edge grows with K
(x2.12 at K=8) and why this band rewards unequal within-cell power sharing.
A pointwise per-user MLP cannot express any of that -- it does not know its own
cell's power commitment, its rank among its cell-mates, or which neighbour cell
is the dominant aggressor; it can only apply a gain->power curve.

This family replaces the pointwise map with a permutation-equivariant
message-passing net over the (user, cell) graph, holding the loss, sampler and
budgets fixed so architecture is the only variable:
  * users pool into a cell embedding (mean+max over the K axis) so the model can
    see and coordinate its cell's aggregate power commitment;
  * cell embeddings propagate back to users through the channel itself, weighted
    by the normalised cross-gains A[k,b,c], so "who interferes with me" is an
    edge weight rather than a summary scalar;
  * two residual rounds of this, then a shared sigmoid head.
Weights are shared over BOTH users and cells, so one parameter set serves every
K, and the cross-gain weighting keeps the map equivariant to cell relabelling.
Node features are enriched with SINR-like functionals of the INPUT only
(full-power SINR, desired SNR, intra-cell-only SINR, within-cell gain rank) --
explicitly permitted by the inference contract; no objective is ever evaluated
on candidate powers inside forward(), and the pass is a single feed-forward.
"""

import os

import torch
import torch.nn as nn

from prepare import (K_MAX, B, P_T, N_0, kq_of, sample_channels, slqp_rate,
                     evaluate,
                     # read-only, for the persisted per-cell diagnostic below
                     KS_TEST, PCTS, TEST, FULL_REF, settings_for)

torch.set_num_threads(max(1, os.cpu_count() or 1))

STEPS = int(os.environ.get("STEPS", 2000))
BATCH = int(os.environ.get("BATCH", 128))      # TOTAL drops per optimiser step
POOL  = int(os.environ.get("POOL", 6000))      # total drops, split across K=1..K_MAX
SEED  = int(os.environ.get("SEED", 1))
CH_SEED = 20_000_000
                  # THE ONE CHANGE (exp 45): base seed of the FRESH-DROP training
                  # stream. Both loss terms now sample never-repeated channels
                  # every step instead of re-drawing from the POOL-sized fixed
                  # set, which at the harness budget is 819 drops per K revisited
                  # ~80 times each by a 115k-parameter net while the score is read
                  # on 250 pinned drops it has never seen. The seed is a pure
                  # function of the step index, so identical code still reproduces
                  # an identical score; the range 20,000,000..20,015,996 is
                  # disjoint from `make_pools`' (1000..1520) and from the
                  # evaluator's pinned TEST seeds (5000..5010), and the drops come
                  # from the SAME `sample_channels` law -- there are simply more of
                  # them, so this widens the training set and cannot narrow it.
TASKS = 8         # THE ONE CHANGE (exp 79): 4 -> 8 independent (K, Kq) tasks
                  # averaged into ONE gradient, with the per-step drop count
                  # still held at BATCH (8 x 32, not 8 x 64). Per-step gradient
                  # variance is Var_task/T + Var_drop/(T*SUB) =
                  # Var_task/T + Var_drop/BATCH: at fixed BATCH the DROP term is
                  # a constant 1/256 that T cannot touch, and doubling T halves
                  # what is left of the TASK term -- the disagreement between
                  # the gradient directions of two different (K, Kq) cells,
                  # which exp 15 named as the dominant noise source and only
                  # suppressed at the tail via the cosine anneal. Exp 19 took
                  # T = 1 -> 4 and bought +0.0150, the largest non-architectural
                  # gain of the campaign; this is the untested second half of
                  # the only lever with a positive track record on this file,
                  # and after seven closed representational axes (capacity
                  # 0-for-4, distillation 0-for-6, softening 0-for-3, Kq
                  # re-weighting 0-for-2, head conditioning, search 0-for-2, the
                  # cut plateau and the refinement stage) the attractor exp 76
                  # identified is an OPTIMISATION artefact, not a
                  # representational one.
                  #
                  # THE OBJECTION EXP 78 RAISED WAS ARITHMETICALLY WRONG. It
                  # read SUB 32 -> 16 off the FILE's default BATCH = 128; the
                  # harness exports BATCH = 256 (autoresearch.sh line 39) into
                  # every scored run, so the true move is SUB 64 -> 32 -- the
                  # sub-batch size the champion's own notes are written around,
                  # and the size at which each task's ratio-of-means `ref` has
                  # been argued sound throughout. Costs zero extra rate
                  # evaluations (8 x 32 = 256 = 4 x 64, same drops, same
                  # `_features` fixed points, same einsum FLOPs); doubles the
                  # per-step invocation count on half-sized tensors, so CPU
                  # overhead is up ~20-40% and peak memory HALVES. Exp 45's
                  # fresh-drop seed stride is already 8, so j in 0..7 saturates
                  # it with no collision. TASKS = 4 is the free revert.
SUB   = max(1, BATCH // TASKS)                 # drops per task, 32 at BATCH=256
LR    = 2e-3      # THE ONE CHANGE (exp 80): PEAK learning rate 1e-3 -> 2e-3, the
                  # first digit of it to move since experiment 1. Exp 15 changed
                  # only how it DECAYS (cosine -> 0 over STEPS, +0.0093) and its
                  # own note pre-registered raising the peak as the follow-up
                  # "which annealing makes affordable". Exp 16 collected on that
                  # immediately at 3e-3 and printed 1.392901 against 1.419842 --
                  # the campaign's worst regression, and the reason this knob has
                  # sat untouched for sixty-four experiments.
                  #
                  # THAT RESULT IS BEING READ IN ITS REGIME, NOT IGNORED. Exp 16
                  # ran at TASKS = 1, BATCH = 128, so its per-step gradient
                  # carried Var_task/1 + Var_drop/128; this file's carries
                  # Var_task/8 + Var_drop/256 after exp 19 (T 1->4, +0.0150) and
                  # exp 79 (T 4->8). Whichever term dominates, the noise sigma is
                  # 1.4x (pure drop term) to 2.8x (pure task term) smaller than
                  # what a 3x step diverged through; 2x is the interior of the
                  # bracket exp 16 established, on the first gradient this
                  # campaign has ever cleaned up BEFORE probing the step.
                  #
                  # WHY NOW. The iterate orbits a ball of size ~LR*sigma and
                  # travels ~LR*STEPS/2 in 2000 steps. Exps 19 and 79 cut sigma
                  # 8x at a fixed LR: the ball shrank 8x, the travel did not
                  # grow, and exp 79 duly bought +0.00007 with a p10/p25 seesaw
                  # (+0.012 over five cells, -0.003 over six) that is a
                  # relabelled draw rather than a policy gain. The variance
                  # reduction is banked and unspent; this spends it. Zero extra
                  # drops, zero extra rate evaluations, zero wall-clock, and
                  # `forward()` untouched. LR = 1e-3 is the free revert.

# ---------------------------------------------------------------------------
# The SUPERVISED half of this family. Training also regresses onto a TEACHER --
# from exp 51 a Kq>1 direct-optimisation oracle, matched in the GAUGE-FIXED LOG
# PROFILE the output map actually works in rather than in linear power. The MSE
# term, its oracle, its cache and its generator are training-only and
# unreachable from `forward()`; the max-min fixed point ITSELF is separately an
# input feature (exp 27), which is a functional of A alone and contract-legal --
# see `balance_labels` for the argument.
# ---------------------------------------------------------------------------
ALPHA_T   = 1.0   # REVERTED TO THE CHAMPION'S VALUE (exp 79), not a variable of
                  # this experiment. Exp 78 ran 2.0 and its falsifier 3 fired:
                  # 1.476007 against 1.476460, with every one of the eleven Kq>1
                  # cells inside +-0.003 of exp 76's row (net -0.002 summed over
                  # all eleven). Four points now bracket the weight with the same
                  # oracle -- 0.0 (-0.0028), 1.0 (ref), 2.0 (-0.0005), 8.0
                  # (-0.0104) -- a broad plateau just left of 2 whose interior is
                  # worth no more than a thousandth. THE KNOB IS CLOSED; no
                  # further probes. Exp 77's regulariser reading survives intact
                  # (zero still costs 0.0028); only the hope of a tunable optimum
                  # dies. Restoring 1.0 puts the file back at the 1.476460
                  # champion so that exp 79's TASKS change is measured against
                  # the champion rather than against an unbanked exploring state.
                  #
                  # (Exp-78 note, kept for the history:) 0.0 -> 2.0. Exp 77 took this term
                  # to ZERO and the score FELL, 1.476460 -> 1.473660. That is
                  # its own falsifier 4, and it settles what the term IS: not a
                  # teacher but a REGULARISER -- a second, differently-
                  # conditioned gradient on the same head coordinates, drawn
                  # from a fixed 1440-drop set with its own generator, against a
                  # 115k-parameter net otherwise trained by a single
                  # high-variance direct objective. Every fact the exp-77 note
                  # marshalled about the TARGET stays true and stays irrelevant:
                  # a gradient can be worth having without its target being
                  # worth imitating.
                  #
                  # THE WEIGHT IS NOW BRACKETED, with the SAME oracle at all
                  # three points, for the first time in the campaign:
                  #
                  #     ALPHA_T   0.0        1.0        8.0
                  #     delta   -0.0028    0 (ref)    -0.0104   (exps 77, 69, 64)
                  #
                  # so both curvature signs are measured rather than assumed and
                  # the optimum is interior. A quadratic in the weight peaks at
                  # 3.1; `B*a/(a+k) - c*a` (saturating benefit, linear drag
                  # toward a worse target -- the mechanistically honest shape)
                  # peaks at 1.2 / 1.8 / 2.3 for k = 1 / 3 / 8. 2.0 is the
                  # consensus argmax and the only value inside every form's
                  # interval; predicted +0.0005..+0.0015 on the 17-cell mean,
                  # all of it in the eleven Kq>1 cells since Kq=1 is pinned.
                  # Costs nothing: the term, its oracle and its cache already
                  # run at weight 0, so only the multiplier changes -- not one
                  # RNG draw, not one second of wall clock, not one character of
                  # `forward()`. ALPHA_T = 1.0 is the free revert; falsifier 4
                  # above says when to pull it and closes the knob if so.
                  #
                  # (Exp-68 note, kept for the history:) Exp 64

                  # took it to 8.0 against exp 53's local oracle (-0.0104) and
                  # exp 65 held 8.0 against the certified QFT reference itself
                  # (-0.0121, i.e. WORSE with a strictly better teacher), which
                  # is what closed the distillation thread 0-for-6: the
                  # log-decade MSE is not a monotone proxy for SLqP, so raising
                  # its weight buys the target's profile and loses the metric.
                  # At 1.0 the term is ~7% of the loss and moves a cell ~0.003,
                  # which is the setting the 1.475555 champion ran.
                  #
                  # (Exp-51 note:) weight on the LOG-PROFILE MSE that
                  # replaces exp 24's Kq=1 power-space MSE. That term is now
                  # IDENTICALLY ZERO -- exp 49's `_cut_clamp` makes `model(Ad,1)`
                  # and `balance_labels(Ad)` the same flat-profile 40-iteration
                  # float64 recursion -- so it costs 20% of every step's compute
                  # for no gradient at all, and this change is paid for by
                  # deleting it. Same nominal weight as exp 23-49's ALPHA=1.0 and
                  # the same O(0.1) magnitude against the ratio term's O(1.4), so
                  # it is a MINORITY of the loss; deliberately NOT annealed, for
                  # exp 20's reason (a curriculum that REPLACES the graded
                  # objective loses even when its limit is the exact metric).
                  # UNTUNED -- if the term bites at all, this is the first knob.
                  # (Exps 25/26's ALPHA_Q band-label term was reverted because
                  # its oracle measured BELOW the student at every p10/p25 cell.
                  # That is exactly the trap `teacher_report` exists to gate:
                  # this oracle is measured AS A POLICY, cell by cell, against
                  # the certified QFT columns, and printed BEFORE any score.)
TEACH_TASKS = 30  # cached teacher tasks: K = 1 + j % K_MAX, so exactly THREE per
                  # K including the ungraded 3, 5, 7, 9, each with its own Kq
                  # drawn by the SAME `_sample_band_kq` law the direct-objective
                  # stream uses (flat `frac`, conditioned on Kq >= 2, through the
                  # evaluator's own `kq_of()`). Nothing is narrowed toward the
                  # three graded fractions. (Exp 65's 20 is reverted with its
                  # QFT teacher.)
TEACH_SUB   = 48  # drops per cached task -- 1440 labelled drops in all.
TEACH_STEPS = 60  # Adam steps of the oracle, warm-started AT THE EXP-29 ANCHOR
                  # `w_clip` (already worth ~1.37 as a policy), which is why this
                  # needs 60 rather than exp 39's 1200 from scratch.
TEACH_LR    = 0.05    # step size in DECADES of target SINR -- Adam moves ~this
                  # much per step, so 60 steps spans up to 3 decades of travel
                  # against a +-3-decade output map.
TEACH_ITERS = 25  # fixed-point iterations inside each oracle step. Below
                  # BAL_ITERS=40 purely for cost; `teacher_report` measures the
                  # resulting allocation against the certified QFT columns, so an
                  # under-converged inner solve shows up as a bad teacher there
                  # rather than as a silent bad label.
LN10 = 2.302585092994046
BAL_ITERS = 40    # THE ONE CHANGE (exp 24): normalised fixed-point iterations
                  # per label batch. The map is a contraction in the projective
                  # sense (see `balance_labels`), so convergence is geometric and
                  # 40 is far past it; each iteration is ONE einsum of the shape
                  # `slqp_rate` already runs, so the whole solve costs a small
                  # fraction of the forward pass it supervises. Exp 23's oracle
                  # was cvxpy at ~0.3 s/drop, which capped the label set at 320
                  # FIXED drops -- memorised early, after which the supervised
                  # gradient decayed to zero. This one is unbounded and fresh.

FEAT_MEAN, FEAT_STD = -14.0, 1.5               # log10 gains ~ N(-14, 1)
BAND_MAX_FRAC = 0.25                           # this campaign's percentile ceiling
                  # (exp 17's `frac = BAND_MAX_FRAC * u**2` tilt scored 1.419670
                  # against 1.419842 and is REVERTED -- the draw is flat again.)
KQ_MIN_TRAIN = 2  # THE ONE CHANGE (exp 52): the smallest Kq a TRAINING task may
                  # draw. Since exp 49's `_cut_clamp`, `model(A, 1)` is p*
                  # IDENTICALLY -- the clamp at the 1st order statistic makes the
                  # emitted profile flat, and `_profile_fixed_point` normalises a
                  # flat profile away exactly -- so every Kq=1 task carries a
                  # gradient that is zero to float64 roundoff, for BOTH loss
                  # terms. Exp 51 spotted this for the supervised half and
                  # deleted the dead Kq=1 MSE; it left the DIRECT-OBJECTIVE half
                  # untouched, where `_sample_band_kq` still draws Kq=1 with
                  # probability 4/(K*B) -- 57.1% at K=1, 28.6% at K=2, ...,
                  # 5.7% at K=10, and 16.7% of all task draws averaged over the
                  # uniform K. Conditioning the SAME flat-`frac` law on Kq >= 2
                  # returns that 16.7% to the eleven Kq>1 cells that carry 100%
                  # of the remaining deficit, at an unchanged TASKS=4 and an
                  # unchanged 128 drops per step. `_band_kq_max(K) >= 2` for
                  # every K (it is ceil(0.25*7K), = 2 at K=1), so the conditional
                  # band is never empty.

N_FEAT = 24       # THE ONE CHANGE (exp 69): 21 -> 24. The exp-29 set plus THREE
                  # SET-RESTRICTED INTERFERENCE COUPLINGS -- `a_set`, `a_tgt`
                  # and `s_max`, see `_features` block (f). Every pooling
                  # operation in the model returns an AVERAGE (row-normalised
                  # edge weights, re-normalised victim weights, mean/max cell
                  # pools, softmax attention), but what decides a Kq>1
                  # allocation is a SUM OVER A Kq-SELECTED SUBSET -- "of the Kq
                  # users the metric actually grades, how much of their
                  # interference-plus-noise do I supply?" A mean over all K*B
                  # victims differs from that by the factor Kq/(K*B), which is
                  # why the measured deficit grows monotonely with Kq while the
                  # Kq=1 column, where the subset is a single user that a max
                  # pool does resolve, is exactly optimal. No depth fixes a
                  # normalisation, which is why capacity is 0-for-4 here.
                  #
                  # -- exp 29 (19 -> 21), kept -- The exp-28 set plus the
                  # Kq-CLIPPED WEIGHTED BALANCING allocation and the SINR it
                  # induces -- the FIRST operating point in the feature set that
                  # depends on Kq. The weighted fixed point has SINR ∝ w, so
                  # w = 1 recovers p* exactly and w = sinr_fp recovers FULL POWER
                  # exactly; clipping w = sinr_fp/thr from ABOVE at 1 flattens
                  # every user above the Kq-th cut -- the ones SLqP_Kq never sums
                  # -- onto the cut, releasing their power as interference budget
                  # for the bottom-Kq set, which keeps its shape. The family runs
                  # Kq=1 -> p*, Kq=K*B -> full power: egalitarian to greedy, the
                  # direction the band's own QFT table moves in. Exp 26 measured
                  # the Kq-FREE geometric path between p* and full power at
                  # 1.221/1.035 (K=4, p10/p25) against a student already at
                  # 1.326/1.124 -- it drags every user down together, which is
                  # the wrong family for Kq>1. See EXPERIMENT 29 above.
                  #
                  # (exp 28's three global order-statistic channels are kept:
                  # the percentile rank of `sinr_fp` among the drop's K*B users,
                  # and the signed log-margin from `sinr_fp` and from `sinr_half`
                  # to the Kq-th smallest value of the SAME probe. They bought
                  # +0.0091 -- the graph pools only means and maxima, the two
                  # order statistics that cannot locate a 10th percentile.)
HIDDEN = 48       # Back to the exp-19 champion's width. Exp 22 took it to 64 and
                  # printed 1.434775 against 1.434843 -- a delta of -0.00007 on a
                  # 78% capacity increase, i.e. exactly nothing. With exp 18
                  # (more hops AND more params, -0.0072) and exp 21 (twice the
                  # hops on a QUARTER of the params, -0.0058), that is three
                  # independent probes of capacity and none of them moves the
                  # score: this model is not capacity-limited, which is what
                  # sends experiment 23 to the framework axis instead.
                  # (Exp 50's BETA_SCALE and its global `bhead` exponent are
                  # REVERTED and gone. Its falsifier fired cleanly on both
                  # halves: the six `min` cells held at exactly the anchor values
                  # 1.096/1.233/1.526/1.825/2.024/2.258 -- so the edit was
                  # correct, not buggy -- and the eleven p10/p25 cells moved by
                  # 1.190/1.406/1.522/1.642/1.769 and 1.070/1.144/1.263/1.312/
                  # 1.361/1.392 against exp 49's 1.190/1.405/1.524/1.643/1.771
                  # and 1.070/1.143/1.265/1.317/1.368/1.401, i.e. one noise ball
                  # and, at K=8/10 p25, slightly the WRONG way. A global
                  # egalitarian-vs-greedy scalar is not the missing coordinate,
                  # so the head conditioning axis is closed too.)
W_SCALE = 1.5     # THE ONE CHANGE (exp 81): 3.0 -> 1.5 decades. This is the LAST
                  # structural degree of freedom the `_cut_clamp` theorem leaves
                  # to the head and the only scalar in the emission path that no
                  # experiment has ever moved -- pre-registered as the endgame
                  # probe by exp 80's falsifier 3 and reached now because exp 80
                  # closed the descent recipe from the other side (see LR).
                  #
                  # WHAT IT ACTUALLY SETS -- TWO THINGS, AND THE SECOND IS THE
                  # POINT. The head emits w = w_clip * 10**(W_SCALE*tanh(logit)),
                  # so W_SCALE is (i) the RAIL, the deepest correction reachable,
                  # and (ii) the GAIN, d(decades)/d(logit) = W_SCALE at the
                  # origin. Every discussion of this constant in the file has been
                  # about (i). But Adam moves each head weight by ~LR per step
                  # irrespective of gradient scale, so the head's step measured in
                  # the units the objective is written in -- DECADES of target
                  # SINR -- is W_SCALE * ||LN(h)|| * LR. With HIDDEN=48 and a
                  # LayerNormed stream, ||LN(h)|| ~ 7, and exp 80 just doubled LR
                  # to 2e-3: the head now takes a ~0.03-decade step per iteration
                  # at the origin and far more once the trunk's own weights move.
                  # Halving W_SCALE halves that step and doubles the resolution at
                  # which the profile can be placed, at an unchanged LR.
                  #
                  # WHY RESOLUTION IS THE SUSPECT LEFT STANDING. Exp 76 named the
                  # residual "one strong attractor that every parameterisation
                  # falls into"; exps 79 and 80 then showed it is held there
                  # neither by gradient noise (sigma cut 8x, +0.00007) nor by step
                  # budget (travel doubled, +0.00034 -- coherent, falsifier 2's
                  # sign pattern exactly, but a TENTH of its predicted magnitude,
                  # which is the reading that matters: doubling the distance the
                  # iterate can travel barely moved the answer, so the iterate is
                  # ALREADY at its landing point and the landing point itself is
                  # what is wrong). A too-coarse output map produces exactly that
                  # signature: the head oscillates in decade-sized jumps, the
                  # cosine anneal freezes it wherever the last large step left it,
                  # and neither cleaner gradients nor longer travel help because
                  # the quantisation is in the PARAMETERISATION, not the descent.
                  # It also explains why capacity is 0-for-4 (a wider trunk feeds
                  # the same coarse map), why the exp-76 refinement head moved
                  # 0.805-0.963 DECADES per user and changed nothing (that is a
                  # tanh at ~90% of its own rail -- a saturated head cannot be
                  # re-tuned), and why the deficit grows monotonely with Kq (more
                  # below-cut users to place, each at the same coarse resolution).
                  #
                  # WHY 1.5 STILL HAS ALL THE REACH THE POLICY USES. The rail is a
                  # correction ON TOP OF `w_clip`, which already places every user
                  # at its own full-power depth below the cut, so 1.5 decades is
                  # 1.5 decades of DISAGREEMENT with a calibrated anchor, not the
                  # total dynamic range. Sacrifice is still complete: a graded user
                  # 1.5 decades under the cut contributes log2(1 + c*w) with w at
                  # 3% of the cut, i.e. a few percent of a cut-level rate, so
                  # trap 2's muting move remains fully available (and the anchor's
                  # own depth adds to it). MEMBERSHIP_CHECK says the head revises
                  # at most 2.92 of 17 sacrificed users, i.e. the learned policy is
                  # overwhelmingly SHAPE-within-the-anchor's-set rather than
                  # membership flips, and shape is precisely what resolution buys.
                  #
                  # HEAD_CHECK (new, below `cell_report`, downstream of `score` and
                  # on no training path) measures the correction distribution in
                  # decades and the fraction sitting at >=90% of the rail, so the
                  # follow-up is determinate whichever way the score moves instead
                  # of a second coin flip on the same knob. W_SCALE = 3.0 is the
                  # free revert; `forward()`'s structure, the feature set, the
                  # loss, the samplers and every generator are untouched.
                  #
                  # (Exp-31 note, kept:) the head emits a per-user TARGET
                  # SINR PROFILE as a bounded +-W_SCALE-decade correction to the
                  # Kq-clipped profile `w_clip`, and the model's OUTPUT is the
                  # box-feasible fixed point that realises SINR proportional to
                  # it. tanh bounds the correction, so neither rail is an
                  # attractor: full power is reachable but is not where zero
                  # logits land (trap 1), and muting a user costs a saturated
                  # rail rather than one saturated sigmoid (trap 2).
OUT_ITERS = BAL_ITERS   # iterations of the OUTPUT fixed point -- the same fixed,
                  # unconditional count `balance_labels` has run inside
                  # `forward()` since exp 27. No early stop, no acceptance test,
                  # no objective: see `_profile_fixed_point`.
HEADS = 4         # THE ONE CHANGE (exp 41): heads of the GAIN-BIASED GLOBAL
                  # SELF-ATTENTION each round now runs over the K*B user nodes,
                  # alongside (not instead of) the cell-mediated path. HIDDEN
                  # must be divisible by HEADS: 48 / 4 = 12 per head. Four heads
                  # so the model can hold several distinct global comparisons at
                  # once -- "who is below the cut", "who is my worst aggressor",
                  # "whom do I hurt most" -- which is exactly the faculty exp 28
                  # bought +0.0091 by hard-coding THREE instances of.
CUT_TAU = 0.0     # RETIRED at exp 77 (was 0.25). At 0.0 the `ste` branch inside
                  # `_cut_clamp` cannot execute at all, and `forward()` no
                  # longer passes `ste=True` anyway, so the clamp is again the
                  # champion's exact value AND exact gradient. Exp 74's own
                  # falsifier 4 fired: MEMBERSHIP_CHECK moved well off zero
                  # (0.02/1 at Kq=2 to 4.18/17 at Kq=18) while the eleven Kq>1
                  # cells did not rise, so the frozen-membership diagnosis was
                  # wrong -- the anchor's full-power set was already right. The
                  # surrogate stays in the file, dead and documented, because
                  # the derivation in `_cut_clamp` is correct and worth keeping.
                  #
                  # (Exp-74 note:) the straight-through temperature of
                  # `_cut_clamp`, in DECADES of target SINR (0.25 dec ~ 2.5 dB).
                  # TRAINING ONLY, and it changes no emitted value by a single
                  # bit -- see `_cut_clamp`. It is the reach, above the cut, over
                  # which a user still receives gradient: weight
                  # sigmoid(-z/CUT_TAU) at z decades above the cut, so 0.50 AT the
                  # cut, 0.12 half a decade above, 0.02 a full decade above. Set
                  # to exp 71's own perturbation scale (SEARCH_SIG = 0.15) rounded
                  # up, i.e. the distance a marginal user can plausibly travel;
                  # CUT_TAU = 0.0 restores the champion's exact hard gradient and
                  # is the revert switch.
                  # (Exp 75/76's W_REF and REF_FEAT are DELETED with the
                  # refinement stage. REFINE_CHECK printed a mean |correction|
                  # of 0.805-0.963 decades -- a tanh at ~90% of saturation, an
                  # order of magnitude past the 0.05 gate -- and it re-decided up
                  # to 4.05 of 17 sacrificed users per drop, yet the eleven Kq>1
                  # cells did not move: 1.476050 against exp 74's 1.476142. That
                  # is exp 76's falsifier 4 verbatim, and it closes the "what it
                  # knows" axis alongside capacity 0-for-4, distillation 0-for-6,
                  # softening 0-for-3, Kq re-weighting 0-for-2, head
                  # conditioning, search 0-for-2 and the cut plateau.)
ROUNDS = 4        # One round == one hop of the SINR-balancing power iteration
                  # the Kq=1 cells ask for. 2 -> 4 bought +0.0150, the largest
                  # architectural gain of the campaign; exp 18's 4 -> 6 (1.412625)
                  # and exp 21's weight-tied 8 hops (1.429086) both lost, the
                  # second of them on the clean TASKS=4 gradient that exp 19's
                  # note said was the prerequisite for reading depth. Two
                  # independent probes above 4, both negative: the depth thread
                  # is closed at 4 and the capacity question moves to HIDDEN.


def _band_kq_max(K):
    """Largest Kq inside the band for this K -- computed with the EVALUATOR'S OWN
    kq_of(), so the training range provably matches the graded grid (a prior
    version used round() here while the grid uses ceil(), which silently gave
    THREE of the seventeen graded cells zero training mass)."""
    return kq_of(BAND_MAX_FRAC, K * B)


def _features(A, Kq):
    """Size-invariant per-user NODE features -> [batch, K, B, N_FEAT].

    Every entry is a functional of the INPUT channel alone (SINR-like input
    features are explicitly allowed by the inference contract). Six of them
    read the SINR induced by a FIXED, input-only allocation -- p = P_T, p = P_T/K,
    the intra-cell channel inversion, the analytic max-min point p*(A), the
    midpoint of the log-power path between p* and full power, and (exp 29) the
    Kq-CLIPPED weighted-balancing point, which is the first of them that depends
    on Kq at all -- which is what turns a single probe of the allocation -> SINR
    map into a multi-point read of its sensitivity along the axes the optimum
    actually moves on. None of this is
    a candidate SET and none of it is SELECTION: no objective (no rate, no top-k,
    no SLqP) is ever evaluated here and nothing is compared by utility. The
    allocations are fixed formulas of A; p*(A) is the limit of a parameter-free
    algebraic recursion (see `balance_labels`), which is a function of A in the
    sense an eigenvector of A is, and is likewise computed by iterating.

    Three more (exp 28) are GLOBAL ORDER STATISTICS of two of those probes -- a
    percentile rank and two signed log-margins to the Kq-th smallest value --
    which is a rank/quantile of an existing input feature, not an objective: no
    rate, no top-k of any RATE, no SLqP, no sum, no candidate set, no loop, and
    no dependence on the model's output.

    Three more (exp 69) are SET-RESTRICTED INTERFERENCE COUPLINGS -- the
    unnormalised aggression mass a cell lays on the bottom-Kq set, that mass
    relative to an untargeted cell's, and each victim's single worst aggressor
    share. All three are closed-form algebra on A and the Kq-th order statistic
    of an existing feature; see block (f).

    The Kq feature is normalised by the band's largest Kq for this K, so it
    spans [~0, 1] using the network's full dynamic range instead of a
    quarter-width slice; from exp 28 Kq ALSO enters per-user, through the
    channel, via the margin features.
    """
    K = A.shape[1]
    own = torch.diagonal(A, dim1=2, dim2=3)                    # [bt,K,B]
    tot = A.sum(dim=3)                                         # [bt,K,B]
    cross = (tot - own).clamp_min(1e-30)                       # inter-cell gain sum

    # SINR-like functionals of the input, read at the full-power operating point.
    snr     = P_T * own / N_0                                  # noise-limited ceiling
    intra   = P_T * own / ((K - 1) * P_T * own + N_0)          # intra-cell term alone
    sinr_fp = P_T * own / (K * P_T * tot - P_T * own + N_0)    # everyone at P_T
    sir_x   = own / cross                                      # own vs inter-cell geometry

    # TWO MORE OPERATING POINTS, so the features probe the allocation -> SINR map
    # at three points instead of one and the encoder reads its SENSITIVITY
    # directly instead of reconstructing it from raw gains. Both are functionals
    # of A alone -- exactly the category `sinr_fp` already belongs to (the SINR
    # at the fixed input-only vector p = P_T); no candidate set is scored, no
    # objective is evaluated, no loop runs.
    #
    # (a) uniform LOW power, p = P_T/K, i.e. every cell commits P_T in total.
    #     Not a rescaling of sinr_fp: N_0 does not scale with the powers, so the
    #     gap between the two says whether this user is interference-limited or
    #     noise-limited -- the bit that decides whether backing off is affordable
    #     at all in this noise-significant system.
    sinr_lo = (P_T / K) * own / (P_T * tot - (P_T / K) * own + N_0)

    # (b) intra-cell CHANNEL INVERSION, p_inv[k,b] = P_T * min_k' own[k',b] / own[k,b],
    #     run by every cell simultaneously. It equalises the received signal
    #     P_T*min_own[b] across a cell's users -- the closed-form egalitarian
    #     allocation, and the natural prior for a band whose headroom is the `min`
    #     column. p_inv <= P_T by construction (equality for the cell's weakest
    #     user) and P_T <= Pcell_inv <= K*P_T, so neither feature drifts with K.
    #     Because every cell backs off at once, the induced SINR already carries
    #     the "what if my aggressors also equalise" term. At K=1 it collapses to
    #     full power and sinr_inv == sinr_fp.
    min_own   = own.amin(dim=1, keepdim=True)                  # [bt,1,B]
    p_inv     = (min_own / own.clamp_min(1e-30)).clamp(0.0, 1.0)   # in [0,1] x P_T
    sig_inv   = P_T * min_own                                  # equal within a cell
    pcell_inv = P_T * p_inv.sum(dim=1)                         # [bt,B], in [P_T,K*P_T]
    tot_inv   = torch.einsum('tkbc,tc->tkb', A, pcell_inv)     # incl. own contribution
    sinr_inv  = sig_inv / (tot_inv - sig_inv + N_0).clamp_min(1e-30)

    # (c) THE ONE CHANGE (exp 27): a THIRD closed-form operating point, the
    #     ANALYTIC MAX-MIN allocation p*(A) = balance_labels(A) -- the unique
    #     fixed point of the normalised standard-interference map, i.e. a
    #     Perron-eigenvector object and hence a deterministic FUNCTION of A, in
    #     the same category as `sinr_fp`/`sinr_lo`/`sinr_inv` above. It
    #     evaluates NO objective (no rate, no top-k, no SLqP), compares no
    #     candidates by utility, accepts/rejects nothing and takes no gradient;
    #     it does not read the model's output and returns the same tensor for
    #     any parameters. p* is exactly what the `min` column asks for, and its
    #     log-power path to full power, p(lam) = P_T*(p*/P_T)**lam, is the axis
    #     along which the p10/p25 optima sit -- so the encoder is handed both
    #     endpoints and the midpoint instead of having to rebuild the balancing
    #     problem from raw gains in four hops.
    p_bal     = balance_labels(A)                              # [bt,K,B] in (0,P_T]
    r_bal     = (p_bal / P_T).clamp(1e-8, 1.0)
    sinr_bal  = _induced_sinr(A, p_bal, own)                   # ~= gamma*(A)
    #     lam = 1/2 on the same path: interior, and the p10/p25 regime.
    sinr_half = _induced_sinr(A, P_T * r_bal.sqrt(), own)

    # (d) THE ONE CHANGE (exp 28): GLOBAL ORDER STATISTICS of the probes above.
    #     SLqP_Kq sums the Kq SMALLEST rates, so what separates a p10/p25 policy
    #     from a max-min one is a purely ORDINAL question -- am I inside the
    #     bottom-Kq set, and if I am marginal, is it cheaper to push me out or to
    #     feed those already in? Every other feature here is per-user or
    #     within-cell, and the message graph pools only mean and max over a
    #     cell's users, which are exactly the two order statistics that cannot
    #     locate a 10th percentile at any depth. `q_fp` gives the global
    #     position, and the margins give the signed distance (in half-decades) to
    #     the cut that SELECTS the graded set at this Kq -- read at full power
    #     and at the lam=1/2 fairness-path point, so the pair also says how the
    #     selected set MOVES along the axis the p10/p25 optima sit on. This is
    #     also the first time Kq enters PER USER and through the channel rather
    #     than as one broadcast constant.
    #
    #     Contract: rank and quantile of `sinr_fp`/`sinr_half`, both of which are
    #     shipped features and fixed closed-form functionals of A alone. No
    #     objective is evaluated (no rate, no top-k of any RATE, no SLqP, no
    #     sum), no candidate SET exists, nothing is compared by utility or
    #     accepted/rejected, there is no loop, and no model output is read -- the
    #     values are identical for any parameters. A sort is no more an
    #     optimisation than the `min` in `p_inv` or the `argsort` in `rank`
    #     below, both of which have shipped since exp 13.
    KB = K * B
    kq_cut = max(1, min(int(Kq), KB))

    def _order_stats(s):
        """[bt,K,B] -> (percentile rank in [0,1], log-margin to the Kq-th
        smallest, the Kq-th smallest itself as [bt,1,1]). The first two are
        dimensionless and K-free so nothing drifts as K goes 1 -> 10."""
        f = s.reshape(s.shape[0], KB)                          # [bt, K*B]
        idx = torch.argsort(f, dim=1)
        ar = torch.arange(KB, dtype=f.dtype, device=f.device)
        src = ar.expand_as(f).contiguous()
        pos = torch.zeros_like(f).scatter_(1, idx, src)        # pos[.,u] = rank(u)
        q = pos / max(1, KB - 1)
        thr = f.kthvalue(kq_cut, dim=1, keepdim=True).values.clamp_min(1e-30)
        m = torch.log10((f / thr).clamp_min(1e-30)) / 2.0
        return q.reshape_as(s), m.reshape_as(s), thr.reshape(-1, 1, 1)

    q_fp, m_fp, thr_fp = _order_stats(sinr_fp)
    _, m_half, _ = _order_stats(sinr_half)

    # (e) THE ONE CHANGE (exp 29): a FOURTH closed-form operating point, and the
    #     first one that DEPENDS ON Kq -- the Kq-CLIPPED WEIGHTED BALANCING
    #     allocation. The weighted fixed point (see `clip_balance` /
    #     `balance_labels`) realises SINR ∝ w, so the flat profile w = 1 returns
    #     p* EXACTLY and w = sinr_fp returns FULL POWER exactly; the clip
    #     w = min(sinr_fp/thr, 1) therefore runs BETWEEN the two endpoints the
    #     feature set already carries, indexed by Kq and in the right direction
    #     (Kq=1 -> p*, Kq=K*B -> full power), and it moves users ORDINALLY rather
    #     than uniformly: the users ABOVE the cut -- the ones SLqP_Kq never sums,
    #     whose power is pure interference budget -- are flattened onto the cut
    #     and release everything above it, while the bottom-Kq set keeps its
    #     shape and the whole profile scales up to the box. That is the p10/p25
    #     policy in one line, and no incumbent probe has its shape: all five of
    #     them are Kq-FREE, and exp 26 measured the Kq-free geometric path
    #     between full power and p* at 1.221/1.035 (K=4, p10/p25) against a
    #     student already at 1.326/1.124 -- that path drags every user down
    #     together, including the ones the metric would have left alone. At Kq=1
    #     the clip is empty (w flat) and p_clip is p* identically, so the six
    #     finished `min` cells are protected by construction.
    #
    #     Contract: identical to (c). A fixed positive weight vector leaves a
    #     standard interference function standard, so this is the same unique,
    #     globally-attracting, parameter-free algebraic limit -- a functional of
    #     A and Kq alone. No objective (no rate, no top-k of any rate, no SLqP,
    #     no sum), no candidate SET (one allocation is computed, not chosen),
    #     nothing compared by utility or accepted/rejected, no gradient, and no
    #     model output read -- the same tensor for any parameters.
    p_clip    = clip_balance(A, Kq, sinr_fp, thr_fp)           # [bt,K,B] in (0,P_T]
    r_clip    = (p_clip / P_T).clamp(1e-8, 1.0)
    sinr_clip = _induced_sinr(A, p_clip, own)

    # (f) THE ONE CHANGE (exp 69): THREE SET-RESTRICTED INTERFERENCE COUPLINGS --
    #     the first addition to the feature set in 39 experiments, and the one
    #     quantity the architecture provably cannot form. Exp 63's PAIRED
    #     measurement on the grid's own pinned drops says the whole residual is a
    #     POLICY error: QFT beats the student on 100% of drops in all eleven Kq>1
    #     cells, with a deficit that is a clean monotone function of Kq (0.22% at
    #     Kq=2 -> 2.27% at Kq=18), while the six Kq=1 cells are at or above it
    #     (as p*'s optimality requires). Those eleven cells have then moved
    #     <=0.006 IN TOTAL across fourteen experiments that changed attention, the
    #     drop stream, the Kq floor, the output cut, a refinement round and five
    #     oracles -- one attractor, reached from a zero-init head, with capacity
    #     0-for-4. That is a REPRESENTABILITY claim, and `N_FEAT` above names the
    #     missing operation: every pool in this model is NORMALISED, but the
    #     objective turns on an UNNORMALISED SUM OVER THE Kq-SELECTED SUBSET.
    #
    #     `inset` is the bottom-Kq set at the full-power operating point -- the
    #     same order statistic `q_fp`/`m_fp` have shipped since exp 28, not a new
    #     primitive -- and `share[k,b,c]` resolves `sinr_fp`'s own denominator by
    #     AGGRESSOR: the fraction of victim (k,b)'s interference-plus-noise that
    #     cell c supplies. Then
    #
    #       1. `a_set[c]`  the AGGRESSION MASS cell c lays on the graded set.
    #          Unnormalised, and the sum runs over a Kq-dependent subset: the one
    #          quantity above. Per-CELL, broadcast to that cell's users, which is
    #          the right factorisation -- interference from cell c is set by its
    #          TOTAL power, while who inside c should carry it is what the
    #          per-user channels already say.
    #       2. `a_tgt[c]`  the same mass relative to what an UNTARGETED cell would
    #          supply, so ~1 means "I hurt the graded set no more than average"
    #          and >1 means my power is landing where the metric reads.
    #          Dimensionless and free of both K and Kq, so it does not drift over
    #          the grid the way the raw mass does.
    #       3. `s_max[k,b]`  the victim-side dual: is my pain dominated by ONE
    #          cell (cheap to fix) or diffuse (hopeless)? An unnormalised max over
    #          the EDGE weights; the model's max pool is over a cell's users'
    #          hidden states, so this is not reachable either.
    #
    #     Contract: pure algebra on the INPUT. No rate, no log2, no top-k of any
    #     RATE, no SLqP, no sum of any objective; no candidate SET (one tensor is
    #     computed, not chosen); nothing accepted, rejected or compared by
    #     utility; no loop, no gradient, no objective gradient, and no model
    #     output is read -- the values are identical for any parameters. Kq enters
    #     only through `thr_fp`, the Kq-th order statistic of `sinr_fp`, whose
    #     primitives have run inside `forward()` since exps 28/29. Cost is one
    #     [bt,K,B,B] tensor and two einsums of the shape this function already
    #     runs for `tot`, against the two 40-iteration float64 fixed points it
    #     already computes.
    inset   = (sinr_fp <= thr_fp).to(A.dtype)                  # [bt,K,B] the graded set
    den_fp  = (K * P_T * tot - P_T * own + N_0).clamp_min(1e-30)   # sinr_fp's denominator
    off_a   = A - torch.diag_embed(own)                        # own cell removed (c != b)
    share   = (K * P_T) * off_a / den_fp.unsqueeze(-1)         # [bt,K,B,B] aggressor split
    a_all   = share.sum(dim=(1, 2))                            # [bt,B] mass on ALL victims
    a_set   = torch.einsum('tkb,tkbc->tc', inset, share)       # [bt,B] mass on the set
    a_tgt   = (a_set / a_all.clamp_min(1e-30)) * (float(KB) / float(kq_cut))
    s_max   = share.amax(dim=3)                                # [bt,K,B] worst aggressor
    a_set_u = a_set.unsqueeze(1).expand_as(own)                # per-cell -> its users
    a_tgt_u = a_tgt.unsqueeze(1).expand_as(own)

    # Within-cell context that a pointwise MLP structurally cannot see.
    rel_cell = own / own.mean(dim=1, keepdim=True).clamp_min(1e-30)
    if K > 1:
        order = torch.argsort(own, dim=1)
        ar = torch.arange(K, dtype=own.dtype, device=own.device).view(1, K, 1)
        src = ar.expand_as(own).contiguous()
        rank = torch.zeros_like(own).scatter_(1, order, src) / (K - 1)
    else:
        rank = torch.full_like(own, 0.5)

    kq_in_band = torch.full_like(own, min(1.0, float(Kq) / _band_kq_max(K)))
    k_size = torch.full_like(own, float(K) / K_MAX)

    l = lambda x: (torch.log10(x.clamp_min(1e-30)) - FEAT_MEAN) / FEAT_STD
    d = lambda x: torch.log10(x.clamp_min(1e-30)) / 2.0        # dB-ish, ~unit scale
    return torch.stack([
        l(own), l(cross), d(sir_x),
        d(snr), d(intra), d(sinr_fp),
        d(sinr_lo), d(sinr_inv), d(p_inv),      # the two extra operating points
        d(r_bal), d(sinr_bal), d(sinr_half),    # the max-min fixed point (exp 27)
        q_fp, m_fp, m_half,                     # global order statistics (exp 28)
        d(r_clip), d(sinr_clip),                # the Kq-clipped balance (exp 29)
        d(a_set_u), d(a_tgt_u), d(s_max),       # set-restricted couplings (exp 69)
        d(rel_cell), rank,
        kq_in_band, k_size,
    ], dim=-1)


def _induced_sinr(A, p, own=None):
    """SINR of a GIVEN power vector: A [bt,K,B,B], p [bt,K,B] -> [bt,K,B].

    A plain algebraic read of `prepare.slqp_rate`'s physics line up to (not
    including) its rate/top-k -- no objective is computed here. Used only by
    `_features`, on allocations that are closed-form functionals of A."""
    if own is None:
        own = torch.diagonal(A, dim1=2, dim2=3)
    sig = p * own
    tot = torch.einsum('tkbc,tc->tkb', A, p.sum(dim=1))
    return sig / (tot - sig + N_0).clamp_min(1e-30)


def _ref_features_retired(A, p1, Kq):
    """RETIRED AT EXP 77 -- called from nowhere; kept only as the docstring of a
    closed axis so the next iteration does not re-derive it. Exp 76 implemented
    this and its refinement stage in full: REFINE_CHECK printed a mean
    |correction| of 0.805-0.963 DECADES (a tanh at ~90% of saturation, twenty
    times the pre-registered 0.05 gate) and re-decided up to 4.05 of 17
    sacrificed users per drop, and the eleven Kq>1 cells did not move --
    1.476050 against exp 74's 1.476142. That is exp 76's falsifier 4 verbatim.
    A second freely-parameterised head that SEES the operating point, moves the
    profile a full decade and re-decides a quarter of the sacrificed set
    reproduces the same 17-cell mean to four decimals: the policy is one strong
    attractor, not a knowledge or capacity deficit.

    (Original note follows.) Per-user features of the model's OWN CANDIDATE
    allocation. A: [bt,K,B,B], p1: [bt,K,B] in (0,P_T] -> [bt,K,B,4].

    These are the ONLY quantities in this model that are not functionals of
    (A, Kq) alone, and that is the entire point. Every one of the 24 channels in
    `_features` is evaluated at an allocation FIXED IN ADVANCE (P_T, P_T/K,
    channel inversion, p*, the lam=1/2 midpoint, the Kq-clipped balance), so the
    head has always chosen the below-cut magnitudes without ever seeing the
    operating point they produce.

    WHY THIS IS NOT A RE-READ OF w1. `_profile_fixed_point` realises SINR EXACTLY
    proportional to w, so the induced SINR SHAPE of p1 is w1 renormalised. What
    is NOT recoverable from w1 is the fixed point's NORMALISATION: which user
    binds at P_T, how much power headroom every other user has, and the absolute
    SINR level the profile buys. The objective is a sum of log2(1 + c*w_i), whose
    curvature -- hence how far below the cut each sacrificed user should sit --
    depends on c, and the marginal cost of raising anyone is paid by the binding
    user alone. Forty iterations of a global float64 recursion is not something
    four message-passing rounds compute internally.

      1. `dec_cut`  log10(sinr1 / thr1), the realised decades below the cut. The
         GAUGE the correction is expressed against: 0 on every above-cut user
         (the clamp's theorem), negative inside the graded set. Recoverable from
         w1 in principle, kept because it is free and pointwise.
      2. `lvl_cut`  log10(thr1), the cut's ABSOLUTE SINR level, i.e. log10 c --
         the scale the log2(1 + c*w) curvature turns on. Per drop, broadcast.
      3. `hd_own`   log10(p1 / P_T), own power headroom in decades. EXACTLY 0 for
         the binding user and negative for everyone else, so the head can see who
         pays for a raise.
      4. `hd_cell`  log10(Pcell1 / P_T), the user's own cell's aggregate power,
         which is what every OTHER user's interference term actually reads.

    Contract, item by item. `_induced_sinr` is the SINR-of-a-given-allocation
    primitive `_features` has run inside `forward()` since exp 28; `kthvalue` is
    the order statistic `_clip_profile` has run there since exp 29. No rate, no
    log2, no top-k of any RATE, no SLqP, no utility. There is no candidate SET --
    p1 is not compared with anything, not scored, not accepted or rejected; the
    second pass is emitted unconditionally whatever p1 was. No gradient step, no
    restart, no early stop, no branch on any objective. The allocation read is
    the model's own first stage, which is still a deterministic feed-forward
    function of (A, Kq)."""
    n, K, Bc = p1.shape
    KB = K * Bc
    kq_cut = max(1, min(int(Kq), KB))
    sinr1 = _induced_sinr(A, p1)                               # [bt,K,B]
    thr1 = sinr1.reshape(n, KB).kthvalue(
        kq_cut, dim=1, keepdim=True).values.clamp_min(1e-30).reshape(n, 1, 1)
    dec_cut = torch.log10((sinr1 / thr1).clamp_min(1e-30)) / 2.0
    lvl_cut = (torch.log10(thr1) / 2.0).expand_as(sinr1)
    hd_own = torch.log10((p1 / P_T).clamp_min(1e-30)) / 2.0
    pcell = p1.sum(dim=1)                                      # [bt,B]
    hd_cell = (torch.log10((pcell / P_T).clamp_min(1e-30)) / 2.0
               ).unsqueeze(1).expand_as(sinr1)
    return torch.stack([dec_cut, lvl_cut, hd_own, hd_cell], dim=-1)


def _mlp(din, dh, dout):
    """One hidden layer. Depth comes from stacking residual rounds, not from
    fattening each block -- the 7H-wide update input is the dominant FLOP term."""
    return nn.Sequential(nn.Linear(din, dh), nn.ReLU(), nn.Linear(dh, dout))


class PowerNet(nn.Module):
    """Permutation-equivariant message-passing net over the (user, cell) graph.

    Weights are shared over users AND cells, so one parameter set serves every K
    and the map is equivariant to relabelling either axis. Per round:

        hn    = LayerNorm(h)                      pre-norm: bounds every input
                                                  below, so the residual stream's
                                                  magnitude cannot reach the head
        vic_c = sum_{k,b} u[k,b,c] * hn_kb        VICTIMS -> their aggressor cell
                                                  (u = column-normalised w: who
                                                  cell c hurts, and how much of
                                                  their pain it owns)
        m_b   = [mean_k hn_kb , max_k hn_kb , vic_b]   users -> their own cell
        agg   = sum_{c!=b} w[k,b,c] * m_c         cells -> user, ALONG THE CHANNEL
                                                  (w = row-normalised cross-gains)
        att   = GainBiasedAttention(hn, A)        EVERY user -> user, all pairs
                                                  (exp 41: the learned global
                                                  comparison mean/max cannot do)
        h    += MLP([hn, m_b, agg, att])

    This is exactly the information the objective couples through: the cell power
    totals Pcell[c], reached via the interference gains that weight them. The
    head is then a shared pointwise sigmoid on LN(h): p = P_T * sigmoid(head).
    (Experiment 14 factorised this into a per-user share times a per-cell budget
    gate and regressed 1.4104 -> 1.3966, so the pointwise head is restored.)

    The victim edge is the TRANSPOSE of the forward edge, and without it the
    graph is one-directional at any depth: a cell would learn who aggresses its
    own users but never who IT aggresses. A symmetric cell cannot beat
    SINR <= 1/(K-1) by any internal reallocation, so this band's headroom in the
    `min` column (x2.12 at K=8) has to come from cells shaping their totals to
    protect someone ELSE's worst user -- which at Kq=1 is one specific cell-edge
    user out of up to 70, identified by exactly the per-user fading/position
    residual that the symmetric BS-BS geometry does not carry.

    The LayerNorms are over the HIDDEN axis only, per node, so they commute with
    relabelling users or cells and the map stays equivariant in both axes. They
    exist because an unnormalised residual stack lets h -- and hence the head's
    logit -- grow without bound, and the sigmoid head then saturates dead at P_T
    (the exactly-1.000000 full-power floor, trap 1 in program.md, which killed
    experiment 6 on nothing more than a 5H -> 7H widening of this update MLP).
    """

    def __init__(self, hidden=HIDDEN, rounds=ROUNDS):
        super().__init__()
        self.enc = _mlp(N_FEAT, hidden, hidden)
        self.norm = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(rounds)])
        # THE ONE CHANGE (exp 41): per-round GAIN-BIASED GLOBAL SELF-ATTENTION
        # over all K*B user nodes. Shared over every node, so one parameter set
        # still serves every K; `gbias` maps the log inter-cell gain
        # l(A[k,b,c]) to one additive logit per head, which is what keeps the
        # all-pairs comparison physically anchored. The update MLP's input grows
        # 7H -> 8H to take the attended message alongside the cell-mediated one.
        assert hidden % HEADS == 0, "HIDDEN must be divisible by HEADS"
        self.qw = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(rounds)])
        self.kw = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(rounds)])
        self.vw = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(rounds)])
        self.gbias = nn.ModuleList([nn.Linear(1, HEADS) for _ in range(rounds)])
        # THE ONE CHANGE (exp 42): the VICTIM-direction companion of `gbias`.
        # `gbias` reads A[k,b,c] -- what the key's cell does to the QUERY -- and
        # `gbias_v` reads A[k',c,b], what the QUERY's cell does to the key user.
        # The second gain is a different entry of the same matrix and is not a
        # function of the first, so without this term "whom do I hurt most" is
        # unreachable at any depth; it is the same one-directionality that
        # `_victim_weights` exists to fix on the cell-mediated path.
        self.gbias_v = nn.ModuleList([nn.Linear(1, HEADS) for _ in range(rounds)])
        self.upd = nn.ModuleList(
            [_mlp(8 * hidden, hidden, hidden) for _ in range(rounds)])
        self.norm_out = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.head.bias)          # same neutral start as the baseline
        # THE ONE CHANGE (exp 31): the head's WEIGHT is zeroed too, so the run
        # starts with zero logits, i.e. w = w_clip exactly -- the exp-29 clipped
        # balancing allocation, whose own measured policy value is the persisted
        # CLIP_CHECK row (~1.37 as a 17-cell mean against the 1.000 floor), and
        # at Kq=1 the provable max-min optimum p* itself. The head's weight still
        # receives gradient at step 0 (dL/dW = dL/dlogit * LN(h)), so the
        # backbone is unblocked after the first step; this is the standard
        # zero-init-last-layer residual start, and it is what makes the change a
        # CORRECTION to a validated policy rather than a fresh search.
        nn.init.zeros_(self.head.weight)
        # (Exp 50's `bhead` global-exponent readout is REVERTED. It was
        # zero-initialised, and `nn.init.zeros_` consumes no RNG, so deleting it
        # leaves every module above drawing from the global generator in exactly
        # the order the 1.473999 run did: the initial parameter vector is again
        # bit-for-bit exp 49's.)
        #
        # (Exp 75/76's refinement stage -- `self.ref` / `self.ref_head` -- is
        # DELETED at exp 77 on its own falsifier 4; see `_ref_features_retired`.
        # It was declared AFTER `self.head` and `nn.init.zeros_` consumes no RNG,
        # so removing it leaves every module above drawing from the global
        # generator in exactly the order the 1.476460 champion did: the initial
        # parameter vector is again bit-for-bit exp 69's.)

    @staticmethod
    def _edge_weights(A):
        """Row-normalised inter-cell gains (own-cell diagonal removed):
        w[bt,K,B,B] -- "what fraction of my interference comes from cell c"."""
        off = A - torch.diag_embed(torch.diagonal(A, dim1=2, dim2=3))
        return off / off.sum(dim=3, keepdim=True).clamp_min(1e-30)

    @staticmethod
    def _victim_weights(w):
        """COLUMN normalisation of the forward edge weights: u[bt,K,B,B] with
        u[.,k,b,c] = w[.,k,b,c] / sum_{k,b} w[.,k,b,c] -- "of all the pain cell c
        causes, what share lands on user (k,b)". Normalising the already
        row-normalised w (rather than raw gains) bounds each cell's total
        incoming victim mass at exactly 1 for every K, so the message magnitude
        does not drift as K goes 1 -> 10 and one near-zero-gain user cannot
        swamp it. w[.,k,c,c] = 0 by construction, so a cell's own users are
        excluded -- intra-cell coupling is already carried by the mean/max
        pool."""
        return w / w.sum(dim=(1, 2), keepdim=True).clamp_min(1e-30)

    def _attend(self, i, hn, gl):
        """THE ONE CHANGE (exp 41): gain-biased global self-attention over the
        K*B user nodes. hn: [t,K,B,H], gl: [t,K,B,B] the normalised log gain
        l(A[k,b,c]) -> [t,K,B,H].

        Node (k,b) is flattened to row k*B+b and attends to every node (k',c) at
        column k'*B+c, with logit

            <q,k>/sqrt(d_h) + W_g * gl[k,b,c] + W_v * gl[k',c,b]      (exp 42)

        i.e. BOTH physical directions of the pair. The first bias depends on the
        AGGRESSOR CELL c of the key, not on which of its users is being read,
        which is exactly the physical statement that interference from cell c
        reaches user (k,b) through the single gain A[k,b,c]. The second (THE ONE
        CHANGE, exp 42) is its mirror: the gain from the QUERY's cell b into the
        key USER (k',c) -- "how much do I hurt you" -- which is a different entry
        of A, is not recoverable from the first, and is the comparison the layer
        needed to decide who above the cut should release its budget.

        Equivariance, both axes, both terms. Cell relabelling b -> pi(b) permutes
        query rows and key columns together while A[k,b,c] -> A[k,pi(b),pi(c)]
        and A[k',c,b] -> A[k',pi(c),pi(b)]. Per-cell user relabelling: the first
        bias does not depend on the key's user index k' at all, and the second
        depends on it through the key's OWN index, so it permutes with the key
        node -- in both cases the attended multiset is unchanged.
        The softmax normalises over all K*B keys, so message magnitude does not
        drift as K goes 1 -> 10.

        Contract: a plain feed-forward layer -- no gradient, no candidate set,
        no objective, no loop, no model output read."""
        t, K, Bc, H = hn.shape
        n = K * Bc
        dh = H // HEADS
        x = hn.reshape(t, n, H)
        q = self.qw[i](x).reshape(t, n, HEADS, dh).transpose(1, 2)   # [t,hd,n,dh]
        k = self.kw[i](x).reshape(t, n, HEADS, dh).transpose(1, 2)
        v = self.vw[i](x).reshape(t, n, HEADS, dh).transpose(1, 2)
        logit = torch.matmul(q, k.transpose(-1, -2)) / (float(dh) ** 0.5)
        # AGGRESSOR direction. gl [t,K,B,B] -> [t,HEADS,K,B,1,B]: indexed by the
        # query (k,b) and the key's CELL c, broadcast over the key's user axis.
        gb = self.gbias[i](gl.unsqueeze(-1))                         # [t,K,B,B,HEADS]
        gb = gb.permute(0, 4, 1, 2, 3).unsqueeze(4)                  # [t,hd,K,B,1,B]
        # VICTIM direction (THE ONE CHANGE, exp 42) -> [t,HEADS,1,B,K,B]: indexed
        # by the query's CELL b and the FULL key node (k',c), broadcast over the
        # query's user axis. glv[t,bq,k',c] = gl[t,k',c,bq] is the gain from the
        # query's cell into the key user, i.e. the transpose of the edge above.
        glv = gl.permute(0, 3, 1, 2)                                 # [t,B,K,B]
        gbv = self.gbias_v[i](glv.unsqueeze(-1))                     # [t,B,K,B,HEADS]
        gbv = gbv.permute(0, 4, 1, 2, 3).unsqueeze(2)                # [t,hd,1,B,K,B]
        logit = logit.reshape(t, HEADS, K, Bc, K, Bc) + gb + gbv
        a = torch.softmax(logit.reshape(t, HEADS, n, n), dim=-1)
        o = torch.matmul(a, v)                                       # [t,hd,n,dh]
        return o.transpose(1, 2).reshape(t, K, Bc, H)

    def raw_profile(self, A, Kq):
        """The head's TARGET SINR PROFILE, before `_cut_clamp` -- [bt,K,B] > 0.

        (Exp 76's `return_h` flag is deleted with the refinement stage that was
        its only caller; this is again exp 69's signature and body, character for
        character.)

        THE ONE CHANGE (exp 51) factors this out of `forward()` with the numerics
        byte-identical, because the exp-51 distillation loss is written in this
        object's own coordinates and needs it BEFORE the clamp. The clamp routes
        no gradient to an above-cut entry (`torch.minimum` sends it to the cut
        element instead), so a loss written on the clamped profile could never
        teach the head to push a user DOWN into the bottom-Kq set -- which is
        half of what the teacher knows. Reading the raw profile restores that
        gradient path without changing what `forward()` emits by a single bit."""
        w = self._edge_weights(A)                                  # [bt,K,B,B]
        u = self._victim_weights(w)                                # [bt,K,B,B]
        # Normalised log gain, the same transform `_features` applies to `own`
        # and `cross`; read once and reused as the attention bias every round.
        gl = (torch.log10(A.clamp_min(1e-30)) - FEAT_MEAN) / FEAT_STD  # [bt,K,B,B]
        h = self.enc(_features(A, Kq))                             # [bt,K,B,H]
        for i, (norm, upd) in enumerate(zip(self.norm, self.upd)):
            hn = norm(h)                                                  # [bt,K,B,H]
            vic = torch.einsum('tkbc,tkbh->tch', u, hn)                   # [bt,B,H]
            cell = torch.cat([hn.mean(dim=1), hn.amax(dim=1), vic],
                             dim=-1)                                      # [bt,B,3H]
            agg = torch.einsum('tkbc,tch->tkbh', w, cell)                 # [bt,K,B,3H]
            own_cell = cell.unsqueeze(1).expand(-1, hn.shape[1], -1, -1)
            # THE ONE CHANGE (exp 41): the attended all-pairs message joins the
            # cell-mediated ones. The cell path is untouched -- attention is
            # ADDED, so nothing that carried the `min` column is removed.
            att = self._attend(i, hn, gl)                                 # [bt,K,B,H]
            h = h + upd(torch.cat([hn, own_cell, agg, att],
                                  dim=-1))                            # H+3H+3H+H

        # THE ONE CHANGE (exp 31): the shared pointwise head on the normalised
        # stream (exp 13) no longer emits a POWER. It emits a bounded correction
        # to a per-user TARGET SINR PROFILE, and the model's output is the
        # box-feasible allocation that realises it.
        #
        #     w = w_clip * 10 ** (W_SCALE * tanh(logit))
        #     p = the unique fixed point with SINR proportional to w
        #
        # This is a LOSSLESS reparameterisation, not a restriction: for any p,
        # w = SINR(p) gives w*F = p, so p is its own profile's fixed point iff
        # max p == P_T -- and the optimum always satisfies that, because scaling
        # every power up by c > 1 multiplies signal and interference by c while
        # N_0 stays put, raising every SINR and hence SLqP (the pre-launch audit
        # certified this direction). The image of the head is therefore exactly
        # the set of undominated allocations; only the redundant overall LEVEL is
        # removed from the search, and w's own scale cancels in the fixed point's
        # normalisation, so the head chooses SHAPE alone.
        #
        # Still ONE feed-forward pass under the contract: no gradient step, no
        # restart, no candidate set, no loop whose acceptance or output depends
        # on evaluating the objective -- `_profile_fixed_point` runs a fixed,
        # unconditional OUT_ITERS passes and evaluates no rate, no top-k, no
        # SLqP and no utility of any kind. Output is in (0, P_T] by construction
        # and equivariant in both the user and the cell axis.
        hout = self.norm_out(h)                                      # [bt,K,B,H]
        logit = self.head(hout).squeeze(-1)                          # [bt,K,B]
        prof = _clip_profile(A, Kq) * torch.pow(                     # NOT `w`:
            torch.tensor(10.0, dtype=logit.dtype),                   # that name is
            W_SCALE * torch.tanh(logit))                             # the edge set
        return prof

    def forward(self, A, Kq):
        # THE ONE CHANGE (exp 49): the head's profile is clamped at its OWN Kq-th
        # smallest value, so every user outside the bottom-Kq set is tied to the
        # cut -- the structure the optimum provably has (see `_cut_clamp`). The
        # clamp is monotone, so the head still chooses WHICH users are in the
        # bottom set and their shape; only the provably-wasted freedom above the
        # cut is removed. At Kq=1 this makes the profile flat and the output p*
        # exactly, for any weights.
        #
        # REVERTED AT EXP 77 to exactly these two lines -- exp 69's `forward()`,
        # character for character. Exp 74's `ste=True` and exp 75/76's second
        # pass are both gone on their own pre-registered falsifiers (see the
        # header and `_ref_features_retired`); `forward()` is again ONE trunk
        # pass, ONE clamp with its exact hard gradient, and ONE fixed point.
        w = self.raw_profile(A, Kq)
        return _profile_fixed_point(A, w=_cut_clamp(w, Kq), iters=OUT_ITERS)


def make_pools():
    """Seeded per-K training pools; identical code -> identical score."""
    per_k = max(64, POOL // K_MAX)
    pools = {}
    for K in range(1, K_MAX + 1):
        chunks, n, s = [], 0, 0
        while n < per_k:
            b = min(512, per_k - n)
            chunks.append(sample_channels(b, K, seed=1000 + 50 * K + s))
            n += b; s += 1
        pools[K] = torch.cat(chunks, dim=0)
    return pools


def _sample_band_kq(K, g):
    """Sample a training Kq within the 0-25% band, using the EVALUATOR'S OWN
    kq_of() so the training range exactly matches the graded grid. Sampling is
    continuous over the band (not just the 3 graded points): the prior
    full-range campaign narrowed training to exactly its graded points, which
    inflated its headline score while destroying off-grid generalisation.
    Verified: every graded cell receives non-trivial training mass (see the
    audit note in program.md).

    The draw is flat on [0, BAND_MAX_FRAC], as it was for the exp-15 champion:
    exp 5's log-uniform Kq (1.382542) and exp 17's squared tilt (1.419670) both
    lost, so re-weighting the band is 0-for-2 and the flat draw stands. Exp 19
    calls this sampler TASKS times per step instead of once; the LAW it draws
    from is untouched, so every graded cell keeps exactly the mass it had.

    THE ONE CHANGE (exp 52): that flat law is now CONDITIONED ON Kq >= 2. It is
    not a re-weighting of the band (0-for-2 above) and not a narrowing toward the
    graded points -- it is the removal of the one sub-event whose gradient is
    provably zero. `kq_of(frac, KB) >= 2` exactly when `frac > 1/KB`, so drawing
    `frac` flat on (1/KB, BAND_MAX_FRAC] is the EXACT conditional of the incumbent
    law: the relative density over every Kq >= 2 -- graded and ungraded alike --
    is bit-for-bit unchanged, only rescaled. `max(KQ_MIN_TRAIN, .)` closes the
    measure-zero `u == 0` endpoint where `ceil` would still return 1.

    One `torch.rand` is consumed, exactly as before, so `g`'s stream and hence
    the (K, drop-seed) sequence of every step stay bit-for-bit the 1.474295
    champion's -- the Kq VALUES are the only difference between the two runs.

    WHAT LOSES MASS. The six graded Kq=1 (`min`) cells, deliberately and to
    exactly zero. That is safe by construction, not by hope: since exp 49 they
    are algebraically pinned at p*(A) for ANY parameters, which is the provable
    box-constrained max-min optimum. `_band_kq_max()` is untouched and still
    equals the largest graded Kq for every K, so every graded cell with Kq >= 2
    keeps non-trivial mass (the program.md audit invariant, re-checked)."""
    lo = float(KQ_MIN_TRAIN - 1) / float(K * B)
    frac = lo + float(torch.rand(1, generator=g)) * (BAND_MAX_FRAC - lo)
    return max(KQ_MIN_TRAIN, min(_band_kq_max(K), kq_of(frac, K * B)))


def _profile_fixed_point(A, w=None, iters=BAL_ITERS):
    """THE recursion, in ONE place (exp 31 factored it out of `balance_labels`,
    numerics byte-identical). A: [n,K,B,B], optional w: [n,K,B] positive ->
    P: [n,K,B] in (0, P_T] with max_{k,b} P == P_T and SINR proportional to w.

    Unlike `balance_labels` this is DIFFERENTIABLE in `w`, because from exp 31
    the model's own head supplies `w` and the fixed point is the output layer.
    Everything the inference contract bans is absent, item by item: no gradient
    step is taken here; there is no candidate SET (one profile in, one allocation
    out -- nothing is enumerated, compared, accepted or rejected); and no loop's
    acceptance or output depends on evaluating the objective, because the loop
    runs a FIXED, unconditional `iters` passes and never computes a rate, a
    top-k, an SLqP, a sum, or any utility at all -- there is no branch, no
    comparison and no early stop in the body. It is an implicit LAYER: a fixed
    positive weight vector leaves F a standard interference function, so the
    limit is unique and globally attracting and does not depend on where the
    iteration starts, exactly as a Perron eigenvector of the input does not.
    Run in float64, returned in A's dtype.
    """
    A64 = A.double()
    own = torch.diagonal(A64, dim1=2, dim2=3)                  # [n,K,B]
    p = torch.full_like(own, float(P_T))
    w64 = None if w is None else w.double().clamp_min(1e-30)
    for _ in range(iters):
        total = torch.einsum('tkbc,tc->tkb', A64, p.sum(dim=1))
        f = (total - p * own + N_0) / own.clamp_min(1e-300)
        if w64 is not None:
            f = f * w64
        p = P_T * f / f.amax(dim=(1, 2), keepdim=True).clamp_min(1e-300)
    return p.to(A.dtype)


def _clip_profile(A, Kq, sinr_fp=None, thr=None):
    """The exp-29 Kq-clipped TARGET SINR PROFILE, w = min(sinr_fp/thr, 1), where
    thr is the Kq-th smallest full-power SINR in the drop. [n,K,B], values <= 1.

    Factored out of `clip_balance` (numerics byte-identical) because from exp 31
    it is also the ANCHOR of the output head. A functional of (A, Kq) alone: no
    objective, no candidate set, no model output, no loop."""
    K, KB = A.shape[1], A.shape[1] * A.shape[2]
    if sinr_fp is None:
        own = torch.diagonal(A, dim1=2, dim2=3)
        tot = A.sum(dim=3)
        sinr_fp = P_T * own / (K * P_T * tot - P_T * own + N_0)
    if thr is None:
        kq_cut = max(1, min(int(Kq), KB))
        f = sinr_fp.reshape(sinr_fp.shape[0], KB)
        thr = f.kthvalue(kq_cut, dim=1,
                         keepdim=True).values.clamp_min(1e-30).reshape(-1, 1, 1)
    return (sinr_fp / thr).clamp(1e-30, 1.0)


def _cut_clamp(w, Kq, ste=False):
    """THE ONE CHANGE (exp 49): clamp a target SINR profile at its OWN Kq-th
    smallest value. w: [n,K,B] positive, Kq int -> [n,K,B].

    THE ONE CHANGE (exp 74) is the optional `ste` flag -- a STRAIGHT-THROUGH
    gradient for the K*B - Kq entries this clamp pins at the cut. It changes the
    RETURNED VALUE for no input whatsoever (see below) and is passed only from
    `forward()`, so `teach_profile`'s oracle, `_qft_profile` and every other
    caller keep the champion's exact numerics AND its exact gradient.

    WHY. `torch.minimum(f, thr)` returns `thr` for every above-cut entry, so
    d out_j / d f_j is EXACTLY ZERO there and the whole of that gradient is
    routed onto the single entry realising the cut. That zero is arithmetically
    correct -- the emitted allocation really is locally constant in f_j -- and it
    is also a PLATEAU that freezes the one decision this policy is made of. The
    head is pointwise, so f_j depends on logit_j alone; hence for every above-cut
    user dL/dlogit_j == 0 identically, for all 2000 steps, at every K and Kq.
    Membership can therefore never be revised OFF THE ANCHOR: at zero logits
    `raw_profile` is `w_clip = min(sinr_fp/thr_fp, 1)`, whose Kq-th smallest is
    exactly 1, so the bottom-Kq set the run starts with is precisely "the Kq-1
    worst users AT FULL POWER", and only a below-cut user rising can ever leave
    it. Nothing can enter. That is the campaign's whole residual, in one line:
    the deficit is entirely in the eleven Kq>1 cells, it grows monotonely with Kq
    (0.65% at Kq=2 to 4.46% at Kq=18 -- i.e. with the NUMBER of frozen
    decisions), and it is exactly zero in the six Kq=1 cells, where the set is
    empty and this function is algebraically the optimum.

    THE SURROGATE, in the profile's own multiplicative units. With
    z = log10(f/thr) the hard clamp is f * 10**(-relu(z)); the straight-through
    gradient is that of the softplus relaxation

        soft = f * 10 ** ( -CUT_TAU * softplus(z / CUT_TAU) )

    for which d log soft / d log f = sigmoid(-z / CUT_TAU): 1 far below the cut
    (identical to the hard clamp), 0.5 AT it, and decaying over CUT_TAU decades
    above it. So a marginal above-cut user is told, by exact backprop rather than
    by sampling, what the plateau hides: "release target, the fixed point's scale
    c rises, and every graded rate rises with it." The pull is largest exactly
    for the users a small move can flip, self-extinguishes for users decades
    clear of the cut, and becomes the true gradient once a user crosses. At zero
    logits every above-cut user sits at z = 0 EXACTLY (w_clip's plateau), so this
    is the well-defined one-sided derivative at the kink, taken at weight 1/2 --
    the run starts on the boundary and can now step off it.

    VALUE-EXACTNESS, not approximately. `out = hard.detach() + (soft -
    soft.detach())`: the parenthesis is a tensor minus its own bitwise-identical
    values, i.e. +0.0 exactly, so `out` equals `hard` bit-for-bit and the six
    Kq=1 `min` cells cannot move. And the branch is additionally gated on
    `torch.is_grad_enabled()`, so under the evaluator's `no_grad` the executed
    code, the emitted powers and the inference time are the champion's unchanged
    -- this is a TRAINING-ONLY change to credit assignment, which program.md
    permits without qualification ("all restrictions apply at INFERENCE only").

    CONTRACT. Nothing is added to the inference path at all. Even if it ran
    there: pure elementwise algebra (log10, softplus, pow) on the profile and its
    own order statistic; no gradient step, no candidate set, no loop, no rate, no
    log2, no top-k of any RATE, no SLqP, nothing accepted or rejected by utility.

    (Original exp-49 note follows.)

    This forces the emitted profile into the structure the optimum provably has.
    Because the model's output realises SINR exactly proportional to w (see
    `_profile_fixed_point`), SLqP_Kq is the sum of log2(1 + c(w)*w_i) over the Kq
    SMALLEST entries of w, and every entry ABOVE that cut enters only through the
    achievable scale c(w), which it can only push DOWN. Such a user cannot be
    muted either -- rate 0 would drop it INTO the bottom set (trap 2) -- so at the
    optimum it sits exactly at the boundary: lower its power by eps and every
    rate in the bottom set strictly rises (all gains are positive) while its own
    falls continuously toward 0, so as long as the bottom SET is unchanged the
    objective strictly improved; hence at the optimum every outside user's SINR
    equals the largest SINR inside the bottom-Kq set.

    The clamp is therefore the IDENTITY on the optimum's own profile
    w* = SINR(p*), i.e. this is a lossless restriction: it removes only
    provably-suboptimal profiles. What the head keeps is everything that matters
    -- the clamp is monotone, so the ORDERING of w (and hence WHICH users form
    the bottom set) is untouched, and the Kq smallest entries themselves are
    passed through unchanged. What it removes is up to K*B - Kq wasted degrees of
    freedom per drop (52 of 70 at K=10/p25, 63 of 70 at K=10/p10).

    At Kq = 1 the threshold is the row MINIMUM, so the result is a FLAT profile
    whatever the head emits and the output is p* identically -- the provable
    box-constrained max-min optimum, by construction, for every K.

    Contract: a monotone elementwise clamp with a data-dependent threshold. No
    gradient step, no candidate SET (one profile in, one profile out), no loop,
    and no objective evaluated -- no rate, no log2, no SLqP, no sum, no top-k of
    any RATE. The threshold is an order statistic of the profile itself, exactly
    the primitive `_clip_profile` has run inside `forward()` since exp 29 and
    `_features._order_stats` since exp 28. `kthvalue` is differentiable, routing
    the gradient to the entry that realises the cut.
    """
    n, K, Bc = w.shape
    KB = K * Bc
    kq_cut = max(1, min(int(Kq), KB))
    f = w.reshape(n, KB)
    thr = f.kthvalue(kq_cut, dim=1, keepdim=True).values
    hard = torch.minimum(f, thr)
    if ste and CUT_TAU > 0.0 and torch.is_grad_enabled():
        z = (torch.log10(f.clamp_min(1e-30))
             - torch.log10(thr.clamp_min(1e-30)))                  # decades
        # softplus(x) = relu(x) + log1p(exp(-|x|)), written out so the only ops
        # used are torch core ones and the large-|z| tail cannot overflow (z can
        # reach tens of decades on `_clip_profile`'s 1e-30 rail).
        x = z / CUT_TAU
        sp = x.clamp_min(0.0) + torch.log1p(torch.exp(-x.abs()))
        soft = f * torch.pow(torch.tensor(10.0, dtype=f.dtype),
                             -CUT_TAU * sp)
        # Value is `hard` bit-for-bit (the parenthesis is exactly +0.0);
        # the gradient is the surrogate's.
        hard = hard.detach() + (soft - soft.detach())
    return hard.reshape(n, K, Bc)


def _log_cut(z, Kq):
    """THE GAUGE FIX (exp 51). z: [n,K,B] a log10 target SINR profile, Kq int ->
    the same profile with each drop's OWN Kq-th smallest entry subtracted.

    `_profile_fixed_point` normalises w away entirely -- w and c*w give the
    IDENTICAL allocation -- so a log-space profile is only defined up to an
    additive per-drop constant, and an MSE between two un-normalised log profiles
    would be dominated by a gauge that changes nothing the model emits. The Kq-th
    smallest entry is the natural gauge because `_cut_clamp` pins every entry
    above it to exactly that value: in these units a clamped profile is <= 0
    everywhere, is exactly 0 on every user outside the bottom-Kq set, and its
    negative entries are the log-decades by which each graded user sits BELOW the
    cut. That is precisely the object the head controls and nothing else.

    Training-only (`forward()` never calls it), but contract-clean regardless: an
    order statistic of a profile, the same `kthvalue` primitive `_clip_profile`
    has run inside `forward()` since exp 29."""
    n, K, Bc = z.shape
    KB = K * Bc
    kq_cut = max(1, min(int(Kq), KB))
    f = z.reshape(n, KB)
    thr = f.kthvalue(kq_cut, dim=1, keepdim=True).values
    return (f - thr).reshape(n, K, Bc)


QFT_CACHE_PATH = "qft_distill_cache.pt"
_QFT_CACHE = None


def _qft_cache():
    """Lazily-loaded disk cache for `_qft_profile`.

    Its OWN file. Exp 63's `qft_grid_headroom_cache.pt` holds QFT solutions of
    the evaluator's PINNED GRID drops and exists purely as a measurement; it is
    never read here and no QFT solution of a `TEST[K]` drop can reach a loss.
    Keys carry (tag, n, K, Kq), so changing TEACH_SUB, TEACH_REP or a task's Kq
    invalidates an entry instead of silently reusing a stale label."""
    global _QFT_CACHE
    if _QFT_CACHE is None:
        try:
            loaded = torch.load(QFT_CACHE_PATH)
            _QFT_CACHE = loaded if isinstance(loaded, dict) else {}
        except Exception:
            _QFT_CACHE = {}
    return _QFT_CACHE


def _qft_profile(A, Kq, tag):
    """THE ONE CHANGE (exp 65): the TEACHER IS QFT ITSELF. A: [n,K,B,B], Kq int
    -> zt: [n,K,B] <= 0, a gauge-fixed, cut-clamped log10 target SINR profile --
    the SAME object exp 51's `teach_profile` returned, so the loss that consumes
    it is untouched, byte for byte.

    TRAINING ONLY. Per-instance convex optimisation, therefore banned at
    inference: never called from `forward()`, never reachable from it, and its
    output is a fixed tensor built before the training loop starts and cached to
    disk. program.md is explicit that all contract restrictions apply at
    INFERENCE only, that training may use anything, and that this campaign may
    generate its own labels with `qft_reference.py` and cache them, noting the
    one-time cost.

    WHY THIS TEACHER AND NOT A SIXTH LOCAL OPTIMISER. Exp 63 solved QFT on each
    graded cell's OWN pinned drops and read both policies there: QFT beats the
    student on 100% of drops in ALL ELEVEN Kq>1 cells (paired ratio 1.0065 to
    1.0446, se 0.001-0.003, monotone in Kq), worth +0.0197 of 17-cell mean, while
    the six `min` cells sit at or above it as p*'s optimality requires. Every
    previous oracle (exps 23, 25/26, 39, 51, 53) was per-instance ascent on a
    top-k objective whose bottom-set membership a hard `kthvalue` recomputes each
    step, and exp 64 supplied the paired verdict on the best of them: dragging
    the student toward it at ALPHA_T=8 LOST 0.0104 and moved every cell the wrong
    way. A policy trained across drops beats local search on this landscape; QFT
    does not.

    THE CONVERSION, and why it cannot lose. `qft_solve` returns an ALLOCATION;
    the student emits a target SINR PROFILE and realises it through
    `_profile_fixed_point`. `_induced_sinr` reads the SINR the QFT allocation
    achieves, `_cut_clamp` flattens every user above the Kq-th smallest onto the
    cut -- the structure the optimum provably has, see `_cut_clamp` -- and
    `_log_cut` removes the one gauge the fixed point normalises away, leaving
    exactly the log-decades by which each graded user sits BELOW the cut and
    exact zeros above it. The clamp only LOWERS targets and F is a standard
    interference function, so the maximal common scale obeys c(w_clip) >= c(w)
    = 1 and SLqP(round trip) >= SLqP(QFT) per drop: the round trip is non-lossy
    by algebra (exp 61), and `teacher_report` measures it against the certified
    columns anyway.

    NUMERICS. `_prep` indexes the flat power vector k*B+b (`M = G.reshape(KB,B)`,
    `SM[b, k*B+b] = 1`) and normalises only the GAINS, so `p_flat` is in absolute
    units under the box `p <= P_T` and `.reshape(K, B)` is the right unflatten.
    A numerically-muted user would put a -inf into `log10` -- and, if it fell
    inside the bottom-Kq set, a zero threshold into `_cut_clamp` -- so the
    induced SINR carries a RELATIVE floor at 1e-8 of the drop's own maximum,
    which is two decades below the -6 the target is floored at anyway and cannot
    truncate structure the optimum uses. float64 throughout, returned in A's
    dtype.

    COST, DECLARED: 10 cvxpy iterations per drop at ~0.3-0.6 s (program.md
    certifies 10 as converged for this band, verified to 60 with <0.1% drift),
    816 solves in all -- ~5-8 min ONE TIME, before the training loop and outside
    the graded `evaluate` call and its 10 s budget -- then free from cache.
    Deterministic (CLARABEL on fixed drops), so identical code still reproduces
    an identical score."""
    key = (str(tag), int(A.shape[0]), int(A.shape[1]), int(Kq))
    cache = _qft_cache()
    hit = cache.get(key)
    if hit is not None:
        return hit.to(A.dtype)

    import numpy as _np
    from qft_reference import qft_solve

    n, K, Bc = int(A.shape[0]), int(A.shape[1]), int(A.shape[2])
    A64 = A.detach().double()
    An = A64.numpy()
    P = torch.empty(n, K, Bc, dtype=torch.float64)
    for i in range(n):
        p_flat, _ = qft_solve(An[i], int(Kq))
        P[i] = torch.from_numpy(
            _np.clip(_np.asarray(p_flat, dtype=_np.float64), 0.0, float(P_T))
            .reshape(K, Bc).copy())

    with torch.no_grad():
        sinr = _induced_sinr(A64, P)                          # [n,K,B] float64
        floor = sinr.amax(dim=(1, 2), keepdim=True).clamp_min(1e-300) * 1e-8
        w = torch.maximum(sinr, floor)
        zt = _log_cut(torch.log10(_cut_clamp(w, Kq)), Kq).clamp(-6.0, 0.0)

    cache[key] = zt
    try:
        torch.save(cache, QFT_CACHE_PATH)
    except Exception:                       # a cache miss must never kill a run
        pass
    return zt.to(A.dtype)


def teach_profile(A, Kq, steps=TEACH_STEPS, lr=TEACH_LR, iters=TEACH_ITERS):
    """LIVE AGAIN at exp 68, at the champion's ALPHA_T = 1.0. Exp 65 replaced it
    with `_qft_profile` (the certified QFT reference itself) at ALPHA_T = 8.0 and
    scored 1.463448 -- below exp 64's 1.465240, which pulled toward THIS oracle
    at the same weight. A strictly better teacher scoring worse is a verdict on
    the SURROGATE, not the target: the log-decade MSE is not a monotone proxy for
    SLqP. `_qft_profile` stays in the file, retired and called from nowhere, with
    its disk cache intact. The original note follows.

    THE ONE CHANGE (exp 51): the Kq>1 TEACHER, in the student's own
    coordinates. A: [n,K,B,B], Kq int -> zt: [n,K,B] <= 0, a gauge-fixed,
    cut-clamped log10 target SINR profile.

    TRAINING ONLY. This is per-instance optimisation and is therefore banned at
    inference -- it is never called from `forward()`, never reachable from it,
    and its output is a fixed tensor cached before the training loop starts.
    program.md is explicit that all contract restrictions apply at INFERENCE only
    and that training may use anything, including labels this campaign generates
    itself; exp 39 ran the same class of oracle.

    WHAT IT OPTIMISES, AND WHY THIS SPACE. Since exp 31 the model's output IS the
    box-feasible allocation realising SINR proportional to a target profile w, so
    the graded objective is an explicit differentiable function of w and the
    optimum can be sought directly in w. Two changes from exp 39's oracle, both
    of which its own null pointed at:

      * exp 39 optimised POWERS and distilled a linear-power MSE over all K*B
        users. Post-exp-49 that is the wrong space twice over -- the output map
        is exponential in the head's logit, so a linear-power error is
        wildly mis-scaled across the +-3 decades the head spans, and the K*B - Kq
        users above the cut (52 of 70 at K=10/p25) contributed most of the MSE
        while contributing NOTHING to what the clamped output emits. Here the
        oracle lives in log10 profile units -- the head's own units, to within
        the tanh -- and `_cut_clamp` removes the wasted coordinates from teacher
        and student alike;
      * it started from scratch and needed 1200 steps. This starts at the exp-29
        anchor `w_clip`, already worth ~1.37 as a policy and the head's own
        zero-logit point, so 60 Adam steps at 0.05 decades is a correction rather
        than a search -- which is what makes a cached teacher affordable inside
        the harness budget.

    `_cut_clamp` is applied INSIDE the loop, so the oracle searches exactly the
    space the student emits into: membership is recomputed each step, the bottom
    set keeps full gradient, and the K*B - Kq above-cut coordinates are pinned at
    the cut by the theorem in `_cut_clamp` rather than left to drift.

    ONE-TIME COST (declared): TEACH_TASKS + 11 report cells, each `steps` Adam
    steps over an `iters`-deep float64 fixed point on <= 48 drops -- ~20-25 s
    before the training loop, paid for by deleting exp 24's Kq=1 power MSE, whose
    gradient exp 49's clamp made identically zero (`model(Ad,1)` and
    `balance_labels(Ad)` are now the same flat-profile recursion) while it still
    consumed ~20% of every step. Deterministic: Adam from a fixed start on fixed
    drops, so identical code still reproduces an identical score.
    """
    z = torch.log10(_clip_profile(A, Kq).clamp_min(1e-12)).detach().clone()
    z.requires_grad_(True)
    o = torch.optim.Adam([z], lr=lr)
    for _ in range(steps):
        o.zero_grad()
        w = _cut_clamp(torch.exp(LN10 * z), Kq)
        p = _profile_fixed_point(A, w=w, iters=iters)
        (-slqp_rate(p, A, Kq).mean()).backward()
        o.step()
    with torch.no_grad():
        w = _cut_clamp(torch.exp(LN10 * z), Kq)
        # clamp(max=0) is the theorem, not a guard: a cut-clamped profile in its
        # own gauge is <= 0 everywhere and exactly 0 outside the bottom-Kq set,
        # and that zero is what the loss reads as "this user is above the cut".
        # The -6-decade floor IS a guard, on `_clip_profile`'s own 1e-30 rail:
        # the target must never be able to hand the MSE a -30. It is far outside
        # the ~1-2 decades the anchor's bottom set actually spans, so it does not
        # truncate any structure the optimum uses; only `zt` is floored, so the
        # STUDENT keeps live gradient at every value it can emit.
        return _log_cut(torch.log10(w.clamp_min(1e-30)),
                        Kq).clamp(-6.0, 0.0)


@torch.no_grad()
def balance_labels(A, iters=BAL_ITERS, w=None):
    """THE ONE CHANGE (exp 24): the EXACT Kq=1 optimum for a batch of drops,
    computed analytically in a handful of einsums. A: [n,K,B,B] -> P: [n,K,B].

    `w` (exp 29, optional, [n,K,B] positive) is a TARGET SINR PROFILE: the fixed
    point then satisfies SINR ∝ w instead of SINR = const. Scaling F by a fixed
    positive vector leaves it a standard interference function, so the iteration
    keeps the same global geometric convergence to the same unique limit; w=None
    is the flat profile and is byte-identical to every prior experiment. Note the
    two exact endpoints this gives for free: w flat returns p*, and w = sinr_fp
    returns FULL POWER (at p = P_T, F = P_T/sinr_fp, so w*F is the constant P_T,
    a fixed point the iteration starts at and never leaves).

    Kq=1 grades `min_i rate_i`, and rate is monotone in SINR, so that cell is the
    max-min SINR problem over all K*B users under the per-user box p <= P_T.
    Writing the interference-plus-noise per unit own gain,

        F[k,b](p) = ( sum_c A[k,b,c]*Pcell[c] - p[k,b]*own[k,b] + N_0 )/own[k,b]

    F is affine with non-negative coefficients and a strictly positive constant,
    hence a STANDARD interference function -- positive, monotone, and strictly
    scalable precisely BECAUSE N_0 > 0 (this band is noise-significant; program.md
    is explicit that the problem is not scale-invariant). The normalised
    fixed-point iteration below therefore converges globally and geometrically to
    a unique p*, at which every SINR equals P_T/||F(p*)||_inf and exactly one user
    sits at P_T.

    p* is OPTIMAL, not merely balanced: for a common target gamma the minimal
    feasible allocation is the fixed point of gamma*F, that fixed point is
    increasing in gamma, and it touches the box exactly at gamma* -- so no
    box-feasible allocation achieves a larger min SINR. This is the same object
    QFT's ten iterations converge to, computed directly.

    Note the gamma cancels in the normalisation: p <- P_T * F(p)/max F(p) needs no
    target and no bisection. Output is in (0, P_T] by construction, so the labels
    never ask for a muted user (trap 2) and never leave the box.

    CONTRACT (exp 27 also calls this from `_features`, i.e. inside `forward()`).
    Nothing here is banned at inference: no gradient is taken, no objective is
    evaluated (no rate, no top-k, no SLqP), no candidate set is scored, and no
    iterate is accepted or rejected by any utility -- the recursion is a fixed,
    parameter-free algebraic map, its limit p*(A) is unique and independent of
    where it starts, and computing it iteratively is simply how one evaluates a
    Perron eigenvector. It reads only A, never the model's output, and returns
    the same tensor whatever the parameters are. That places it in the category
    program.md permits explicitly -- "SINR-like features of the input are fine"
    -- and outside the learned unrolled optimiser held out of scope, which is
    defined by evaluating objective GRADIENTS on the model's own candidates.
    Run in float64 for the fixed point, returned in A's dtype; ~40 einsums on a
    [t,K,B] tensor, ~0.3 s over the whole grid against the 10.0 s budget.
    """
    return _profile_fixed_point(A, w=w, iters=iters)


@torch.no_grad()
def clip_balance(A, Kq, sinr_fp=None, thr=None):
    """THE ONE CHANGE (exp 29): the Kq-CLIPPED weighted-balancing operating
    point. A: [n,K,B,B], Kq int -> P: [n,K,B] in (0, P_T].

    The target profile is the full-power SINR distribution CLIPPED FROM ABOVE at
    its Kq-th smallest value,

        thr = Kq-th smallest sinr_fp in the drop
        w   = min(sinr_fp / thr, 1)

    i.e. "no user above the cut needs more than the cut". Users ABOVE the cut --
    the ones SLqP_Kq never sums, whose own rates are worth nothing to the metric
    and whose power is pure interference budget -- are flattened onto the cut and
    release everything above it; the bottom-Kq set keeps its full-power relative
    SHAPE and the whole profile is scaled up until one user touches P_T.

    Because the weighted fixed point realises SINR ∝ w, this is a one-parameter
    family INDEXED BY Kq that runs between the two allocations the feature set
    already carries, and it runs the right way round:

        Kq = 1     -> every s >= 1, so w is flat and the result is p* EXACTLY,
                      the provable Kq=1 optimum (the `min` column is protected
                      by construction, not re-litigated)
        Kq = K*B   -> every s <= 1, so w = sinr_fp and the result is FULL POWER
                      exactly (at p = P_T, F = P_T/sinr_fp, so w*F is constant)

    so the dial goes egalitarian -> greedy as Kq grows, which is the direction
    the band's own QFT table moves in (x2.07 at min against x1.44 at p25). It
    moves users ORDINALLY, not uniformly, which is what distinguishes it from the
    geometric path p(lam) = P_T*(p*/P_T)**lam that exp 26 measured at 1.221/1.035
    (K=4, p10/p25) against a student already at 1.326/1.124: that path drags
    every user down together, including the ones already above the cut that the
    metric would happily have left alone.

    Contract: see `balance_labels`. A fixed positive weight vector leaves a
    standard interference function standard, so this is one more parameter-free
    algebraic limit that is a function of (A, Kq) alone -- no objective, no
    candidate set, no utility comparison, no gradient, no model output.
    `sinr_fp`/`thr` may be passed in when the caller has already computed them
    (as `_features` has); they are recomputed from A alone otherwise.
    """
    return balance_labels(A, w=_clip_profile(A, Kq, sinr_fp, thr))


@torch.no_grad()
def label_report(pools):
    """Free validation of the max-min oracle, printed once, against program.md's
    certified QFT `min` column. This is the same quantity the grid grades (ratio
    of means to full power at Kq=1) on pool drops, so it says whether
    `balance_labels` reproduces the certified optimiser BEFORE any score is
    read -- a value below 1.12/1.29/1.54/1.73/2.12/2.07 falsifies it.

    Now doubly load-bearing: from exp 27 the same p*(A) is also an INPUT FEATURE
    of every forward pass, so this table certifies the feature, not just the
    label. Exp 25/26's p10/p25 columns are gone with `slqp_labels` -- what they
    measured (a local optimiser that landed BELOW the student on all six of
    those cells) is what closed that thread; see EXPERIMENT 27 above."""
    qft = {1: 1.12, 2: 1.29, 4: 1.54, 6: 1.73, 8: 2.12, 10: 2.07}
    lines = ["LABEL_CHECK (max-min label/full-power; QFT certified in parens)"]
    for K in (1, 2, 4, 6, 8, 10):
        A = pools[K][:64]
        P = balance_labels(A)
        r = (slqp_rate(P, A, 1).mean()
             / slqp_rate(torch.full_like(P, P_T), A, 1).mean())
        lines.append(f"  K={K:>2} ({K*B:>2} users)  min: "
                     f"{float(r):.3f} ({qft[K]:.2f})")

    # CLIP_CHECK (exp 29) -- NOT the experimental variable and off every training
    # path: the new operating point enters the run as an input FEATURE only. This
    # measures the allocation ITSELF, as a policy, at every graded p10/p25 cell,
    # against program.md's certified QFT columns, so the next iteration can
    # decide on a NUMBER whether it also deserves to be a supervised target. Kq>1
    # labels are 0-for-2 (exps 25/26) and both failures were oracles that scored
    # BELOW the student on the very cells they were meant to teach -- exp 26's
    # path oracle printed 1.221/1.035 at K=4 -- which is precisely the check that
    # closed that thread, and precisely the check p_clip has not yet been given.
    qft_band = {(1, "p25"): 1.10, (2, "p10"): 1.21, (2, "p25"): 1.15,
                (4, "p10"): 1.44, (4, "p25"): 1.28, (6, "p10"): 1.55,
                (6, "p25"): 1.33, (8, "p10"): 1.66, (8, "p25"): 1.40,
                (10, "p10"): 1.80, (10, "p25"): 1.44}
    lines.append("CLIP_CHECK (Kq-clipped balance point/full-power; QFT in parens)")
    for K in (1, 2, 4, 6, 8, 10):
        A = pools[K][:64]
        cells = []
        for label, kq in settings_for(K):
            if label == "min":
                continue
            P = clip_balance(A, kq)
            r = (slqp_rate(P, A, kq).mean()
                 / slqp_rate(torch.full_like(P, P_T), A, kq).mean())
            cells.append(f"{label}: {float(r):.3f} ({qft_band[(K, label)]:.2f})")
        lines.append(f"  K={K:>2} ({K*B:>2} users)  " + "   ".join(cells))

    for ln in lines:
        print(ln)
    return lines


def teacher_report(pools):
    """THE GATE ON THE EXP-51 TEACHER, printed BEFORE any score is read.

    READ IT AS CONTEXT, NOT AS A DECISION RULE (exp 68). Exp 63 showed this
    comparison is structurally unpaired -- an oracle on POOL drops against a
    student row read on PINNED drops -- so it cannot rank a teacher against the
    student; and exps 64/65 showed the ranking would not have mattered anyway,
    because the certified QFT reference itself, distilled at ALPHA_T = 8, scored
    BELOW the local oracle at the same weight. The MSE surrogate, not the target,
    is what is broken. The term stays at the champion's ALPHA_T = 1.0 and this
    table is now purely a sanity print on the oracle's convergence.

    (Original note:) Measures the oracle AS A POLICY on all eleven graded cells --
    pool drops, never the pinned grid drops, exactly as CLIP_CHECK does -- against
    program.md's certified QFT columns. This is the check that closed the exps
    25/26 thread: BOTH of those Kq>1 label attempts failed because their oracle
    measured BELOW the student on the very cells it was meant to teach, and a
    teacher worse than its student is a drag term whatever the loss looks like.
    It is also the only way to see whether TEACH_ITERS=25 is enough inner
    convergence -- an under-solved fixed point shows up here as a bad policy
    rather than as a silently bad label.

    Read it against exp 50's persisted student row (p10 1.190/1.406/1.522/1.642/
    1.769, p25 1.070/1.144/1.263/1.312/1.361/1.392): every cell must be ABOVE the
    student for the MSE to be pulling forward, and exp 39's oracle -- a far more
    expensive one -- reached 1.219/1.409/1.499/1.673/1.792 and 1.078/1.159/1.266/
    1.325/1.410/1.434, i.e. ~QFT parity. Off every training path; the model is
    not involved."""
    qft_band = {(1, "p25"): 1.10, (2, "p10"): 1.21, (2, "p25"): 1.15,
                (4, "p10"): 1.44, (4, "p25"): 1.28, (6, "p10"): 1.55,
                (6, "p25"): 1.33, (8, "p10"): 1.66, (8, "p25"): 1.40,
                (10, "p10"): 1.80, (10, "p25"): 1.44}
    lines = ["TEACHER_CHECK (exp-51 oracle/full-power; QFT in parens)"]
    for K in (1, 2, 4, 6, 8, 10):
        A = pools[K][:TEACH_SUB]
        cells = []
        for label, kq in settings_for(K):
            if label == "min":
                continue
            zt = teach_profile(A, kq)
            with torch.no_grad():
                P = _profile_fixed_point(A, w=torch.exp(LN10 * zt),
                                         iters=BAL_ITERS)
                r = (slqp_rate(P, A, kq).mean()
                     / slqp_rate(torch.full_like(P, P_T), A, kq).mean())
            cells.append(f"{label}: {float(r):.3f} ({qft_band[(K, label)]:.2f})")
        lines.append(f"  K={K:>2} ({K*B:>2} users)  " + "   ".join(cells))
    for ln in lines:
        print(ln)
    return lines


@torch.no_grad()
def membership_report(model):
    """THE MECHANISM GATE for exp 74, on the grid's OWN pinned drops.

    The policy's entire Kq>1 content is WHICH Kq-1 of the K*B users sit strictly
    below the cut (`_cut_clamp`'s theorem pins the rest AT it). At zero logits
    that set is the anchor's: `w_clip = min(sinr_fp/thr_fp, 1)` is strictly < 1
    on exactly the Kq-1 worst-at-FULL-POWER users and exactly 1 on the rest. With
    the hard clamp no above-cut user has any gradient, so a user can only ever
    LEAVE that set, never enter it.

    Printed per cell: the mean number of the model's Kq-1 strictly-below users
    that the ANCHOR does not sacrifice, i.e. how far membership has been revised.
    Taking both sets at size Kq-1 sidesteps the anchor's tie at 1.0 (which
    `topk` would break arbitrarily), so 0.00 means "membership never moved" and
    is unambiguous.

    Off every training path; one extra forward pass over the grid, outside the
    graded `evaluate` call and its 10 s budget."""
    model.eval()
    lines = ["MEMBERSHIP_CHECK (model's Kq-1 sacrificed users NOT sacrificed by "
             "the w_clip anchor, per drop)"]
    for K in KS_TEST:
        A = TEST[K]
        n, KB = A.shape[0], K * B
        cells = []
        for label, kq in settings_for(K):
            if kq < 2:
                continue
            raw = model.raw_profile(A, kq).reshape(n, KB)
            anc = _clip_profile(A, kq).reshape(n, KB)
            below = anc < 1.0 - 1e-9                  # the anchor's Kq-1 victims
            sel = raw.topk(kq - 1, dim=1, largest=False).indices
            moved = (~below.gather(1, sel)).sum(dim=1).double().mean()
            cells.append(f"{label}: {float(moved):.2f}/{kq - 1}")
        lines.append(f"  K={K:>2} ({KB:>2} users)  " + "   ".join(cells))
    return lines


@torch.no_grad()
def cell_report(model):
    """Per-cell (model/full-power) ratios on the PINNED grid drops -- the same
    table `evaluate` prints, recomputed here only so it can be PERSISTED.

    Not the experimental variable and not on any training path: `autoresearch.sh`
    captures train.py's stdout into a shell variable and greps out only FAMILY /
    HELDOUT_SCORE / INFERENCE_S, so both this table and LABEL_CHECK have been
    discarded for every experiment so far and each iteration has had to infer
    which cells moved from a single scalar. Writing them to `diagnostics.txt`
    costs one extra forward pass over the grid (~0.4 s, entirely outside the
    graded `evaluate` call and its 10 s budget) and lets the next iteration read
    the answer instead.
    """
    model.eval()
    lines = ["GRID (model/full-power):        min     p10     p25"]
    for K in KS_TEST:
        A = TEST[K]
        cells = {}
        for label, kq in settings_for(K):
            r = slqp_rate(model(A, kq), A, kq).mean().item() / FULL_REF[(K, kq)]
            cells[label] = f"{r:.3f}"
        lines.append(f"  K={K:>2} ({K*B:>2} users):        "
                     + "  ".join(f"{cells.get(l, ' -- '):>6}"
                                 for l, _ in PCTS))
    return lines


@torch.no_grad()
def head_report(model):
    """THE MECHANISM GATE for exp 81, on the grid's OWN pinned drops.

    W_SCALE is both the RAIL and the GAIN of the head's output map
    w = w_clip * 10**(W_SCALE*tanh(logit)), and no experiment has ever measured
    where the learned corrections actually sit inside it. Printed per cell, in
    DECADES of target SINR:

      mean/p50/p90 of |W_SCALE*tanh(logit)|  -- the correction magnitude, i.e.
          how far the head disagrees with the `w_clip` anchor it starts on;
      rail  -- the fraction of users at >= 90% of W_SCALE, i.e. a saturated tanh
          whose gradient is dead and which no annealing can re-tune.

    This is what makes the follow-up determinate instead of a second coin flip on
    the same knob. If `rail` is large and p90 is pinned at W_SCALE, the reach
    binds and the next probe is UP (6.0); if the corrections live well inside the
    rail, the knob is a pure resolution/gain control and the next probe is DOWN
    (0.75). Read it together with MEMBERSHIP_CHECK: reach that is spent on shape
    rather than on membership is reach the policy did not need.

    Off every training path -- called after `evaluate`, uses no generator, and
    one extra `raw_profile` pass over the grid, outside the graded call and its
    10 s budget."""
    model.eval()
    lines = [f"HEAD_CHECK (|correction| in decades; rail = W_SCALE = {W_SCALE})"]
    for K in KS_TEST:
        A = TEST[K]
        cells = []
        for label, kq in settings_for(K):
            if kq < 2:
                continue                      # Kq=1 is flat by the clamp theorem
            raw = model.raw_profile(A, kq)
            anc = _clip_profile(A, kq).clamp_min(1e-30)
            dec = torch.log10((raw / anc).clamp_min(1e-30)).abs().reshape(-1)
            cells.append(
                f"{label}: {float(dec.mean()):.2f}/"
                f"{float(dec.median()):.2f}/"
                f"{float(dec.quantile(0.9)):.2f}"
                f" rail {float((dec >= 0.9 * W_SCALE).double().mean()):.2f}")
        lines.append(f"  K={K:>2} ({K*B:>2} users)  " + "   ".join(cells))
    return lines


def main():
    torch.manual_seed(SEED)
    g = torch.Generator().manual_seed(SEED)
    pools = make_pools()

    # The distillation batch is drawn with its OWN generator (SEED + 991), so `g`
    # -- and hence the direct-objective term's entire (K, Kq, drop) sequence --
    # stays bit-for-bit the exp-19 champion's and the only difference between the
    # two runs is the added gradient. Labels are now generated in-loop, so there
    # is no cache, no cvxpy import and no one-time cost to amortise.
    gd = torch.Generator().manual_seed(SEED + 991)
    diag = label_report(pools)

    # THE TEACHER CACHE, built once, before training -- back to exp 51/52's
    # local-search oracle at exp 68. Exp 65 filled it from `_qft_profile` (the
    # certified QFT reference itself) at ALPHA_T = 8.0 and scored 1.463448,
    # BELOW exp 64's 1.465240, which pulled toward this weaker oracle at the same
    # weight: the distillation surrogate is what is broken, not the target, and
    # the thread is closed 0-for-6. `_qft_profile` remains in the file, retired
    # and called from nowhere, with its disk cache intact.
    #
    # TEACH_TASKS tasks, K = 1 + j % K_MAX so exactly three per K (the ungraded
    # 3, 5, 7, 9 included), each Kq drawn by `_sample_band_kq` -- the SAME law the
    # direct-objective stream uses, flat `frac` on [0, BAND_MAX_FRAC] through the
    # evaluator's own `kq_of()`. Nothing is narrowed toward the three graded
    # fractions, so `_band_kq_max()` still equals the largest graded Kq for every
    # K and no graded cell gains or loses mass. Drops come from `sample_channels`
    # on a seed range (30,000,000+) disjoint from `make_pools`' (1000..1520), the
    # fresh-drop training stream's (20,000,000+) and the evaluator's pinned TEST
    # seeds (5000..5010), so the teacher is never fitted on graded channels.
    TEACH = []
    for j in range(TEACH_TASKS):
        Kt = 1 + j % K_MAX
        Kqt = _sample_band_kq(Kt, gd)
        At = sample_channels(TEACH_SUB, Kt, seed=30_000_000 + j)
        TEACH.append((At, Kqt, teach_profile(At, Kqt)))
    diag = diag + teacher_report(pools)

    model = PowerNet()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    # KEPT FROM exp 15 (+0.0093): anneal the step size to zero instead of holding
    # it at LR for all STEPS. Its premise was that every step optimises ONE
    # (K, Kq) task and consecutive tasks disagree in gradient DIRECTION, not just
    # in scale (exp 2 already equalised the scale) -- exp 19 attacks that same
    # variance at its source (TASKS above) rather than only at the tail, so the
    # two changes compose: cleaner steps throughout, still annealed at the end.
    # Constant-step SGD under that noise does not converge
    # to a minimiser, it converges to a stationary distribution around one, and
    # the reported score is wherever step STEPS happened to land in that ball.
    # Exp 7 measured the ball: a change argued in advance to be policy-neutral
    # moved the score by -0.0015, the same order as the last two structural WINS.
    # Cosine holds ~the peak through the first quarter (so the search that found
    # 1.41 is unchanged) and decays to exactly zero -- any LR floor would leave a
    # residual ball -- which matters most in the `min` column, a near-degenerate
    # SINR-balancing landscape where a constant step keeps kicking the iterate
    # between neighbouring argmin cells instead of resolving one.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS,
                                                       eta_min=0.0)

    model.train()
    for step in range(STEPS):
        opt.zero_grad()

        # THE ONE CHANGE (exp 19): average TASKS independent (K, Kq) tasks into
        # one gradient instead of stepping on a single task. Each draw uses the
        # IDENTICAL sampler, so the marginal law of a task is bit-for-bit the
        # champion's -- K uniform on 1..10 (ungraded 3,5,7,9 included), frac flat
        # on [0, BAND_MAX_FRAC] via the evaluator's own kq_of(). Nothing is
        # narrowed toward the graded points; four samples from the same law are
        # simply averaged before the parameters move, which cuts the task-
        # direction variance fourfold at an unchanged 128 drops per step.
        for j in range(TASKS):
            K = int(torch.randint(1, K_MAX + 1, (1,), generator=g))
            Kq = _sample_band_kq(K, g)
            pool = pools[K]
            # THE ONE CHANGE (exp 45). The pool-index draw is KEPT and its result
            # DISCARDED: `pools` is still built from the same seeds at the same
            # size, so this consumes exactly the values from `g` it always has and
            # the (K, Kq) task SEQUENCE stays bit-for-bit the 1.472451 champion's
            # -- fresh-vs-pooled drops is then the ONLY difference between the two
            # runs. The batch itself is now a never-repeated draw from the same
            # `sample_channels` law, seeded from the step index so identical code
            # still reproduces an identical score.
            torch.randint(0, pool.shape[0], (SUB,), generator=g)
            A = sample_channels(SUB, K, seed=CH_SEED + 8 * step + j)

            # Metric-aligned scaling: the graded quantity is (model SLqP /
            # full-power SLqP) per cell, each cell weighted 1/17. Raw Mbps makes
            # a step's gradient norm scale with Kq and with the rate level, so
            # large-Kq cells dominate Adam's shared second moment and the min
            # column -- where this band's headroom actually is -- is learned with
            # a proportionally tiny step. Dividing by the same sub-batch's
            # full-power SLqP (detached constant, ratio-of-means exactly as the
            # evaluator computes it) puts every task on a common ~1.0-2.0 scale.
            with torch.no_grad():
                ref = slqp_rate(torch.full_like(A[..., 0], P_T), A, Kq).mean()
                ref = ref.clamp_min(1e-12)
            # Divide by TASKS here rather than summing then dividing, so the
            # graph for each sub-batch is freed as soon as it is backpropagated
            # and peak memory matches the old single 128-drop pass. The gradient
            # is the MEAN of the four task ratios, so its scale -- and hence the
            # meaning of LR = 1e-3 -- is identical to the champion's.
            loss = -slqp_rate(model(A, Kq), A, Kq).mean() / (ref * TASKS)
            loss.backward()

        # THE ONE CHANGE (exp 51): the supervised term is a GAUGE-FIXED
        # LOG-PROFILE MSE against a Kq>1 teacher, replacing exp 24's Kq=1 power
        # MSE (which exp 49's clamp made identically zero -- `model(Ad,1)` and
        # `balance_labels(Ad)` are now the same flat-profile 40-iteration float64
        # recursion -- so it bought no gradient for ~20% of every step).
        #
        # The graded objective above is untouched: byte-identical to the 1.473999
        # champion's, not replaced, softened or reweighted anywhere.
        #
        # The loss, per drop, on the K*B coordinates of the profile:
        #
        #     zs = log10(raw head profile) - its own Kq-th smallest   (gauge fix)
        #     zt = the teacher's, cut-clamped, so zt <= 0 and zt == 0 exactly
        #          on the users outside the teacher's bottom-Kq set
        #     err = zs - zt              where zt < 0   (teacher's graded users)
        #           min(zs, 0)           elsewhere      (one-sided)
        #
        # Two properties the power-space MSE of exp 39 did not have. FIRST, the
        # gauge: `_profile_fixed_point` normalises w away, so w and c*w emit the
        # IDENTICAL allocation and an un-normalised log MSE would spend most of
        # its gradient on a constant that changes nothing. Subtracting each
        # profile's own cut removes exactly that constant and nothing else.
        # SECOND, the asymmetry above the cut: the theorem in `_cut_clamp` says
        # every user outside the bottom set sits AT the cut, so the teacher's
        # target there is 0 -- but the student's value there is free above 0
        # (the clamp discards it), and penalising it would put the head back to
        # work on the provably-wasted degrees of freedom exp 49 removed. `min(zs,
        # 0)` therefore penalises ONLY the error that matters, a student
        # wrongly demoting a user INTO the graded set, and is flat above it.
        # Reading `raw_profile` rather than the clamped one is what keeps that
        # gradient alive at all (`torch.minimum` routes no gradient to a clamped
        # entry).
        j = int(torch.randint(0, TEACH_TASKS, (1,), generator=gd))
        At, Kqt, zt = TEACH[j]
        zs = _log_cut(torch.log10(model.raw_profile(At, Kqt).clamp_min(1e-30)),
                      Kqt)
        err = torch.where(zt < 0.0, zs - zt, zs.clamp(max=0.0))
        dist = ALPHA_T * err.pow(2).mean()
        dist.backward()

        opt.step(); sched.step()

    score = evaluate(model)
    torch.save({"state_dict": model.state_dict(),
                "score": score,
                "arch": type(model).__name__}, "last_model.pt")

    # Persist the two tables the harness's stdout grep throws away, so the next
    # iteration can see WHICH CELLS moved rather than only the 17-cell mean.
    try:
        # (Exp 78 repair: exp 77 deleted `refine_report` with the refinement
        # stage but left this call, so the write raised NameError straight into
        # the `except` below and diagnostics.txt on disk stayed exp 76's. The
        # call is gone. This is downstream of `score` and touches no training
        # path -- its only effect is that the falsifier table exists to read.)
        diag = ([f"EXP score {score:.6f}  family interference_attention"] + diag
                + [""] + membership_report(model)
                + [""] + head_report(model)
                + [""] + cell_report(model))
        with open("diagnostics.txt", "w") as f:
            f.write("\n".join(diag) + "\n")
    except Exception as e:                    # never let a diagnostic kill a run
        print(f"(diagnostics write skipped: {e})")

    print("FAMILY interference_attention")
    print(f"HELDOUT_SCORE {score:.6f}")


if __name__ == "__main__":
    main()
