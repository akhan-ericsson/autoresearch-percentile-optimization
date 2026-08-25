"""
prepare.py  --  IMMUTABLE evaluator, LOW-PERCENTILE BAND campaign (band 1 of 4).

Ported faithfully from the authors' MATLAB reference (see verify_model.py for
the geometry certification). Generalizes across network size, but DELIBERATELY
NARROWS the percentile axis to the 0-25% band (min/p10/p25 of K*B):

  * K, the number of users per cell, is a per-instance parameter (1..K_MAX);
  * K_q, the percentile count, ranges over {min, p10, p25} of K*B ONLY -- this
    campaign does not attempt p50 or sum-rate.

Why a narrower scope: a prior campaign spanning the full 0-100% percentile
range found a qualitative regime shift around 50%->100% (egalitarian/
scheduling-like optimal policies at low percentiles vs. concentrated/greedy
policies near sum-rate) that a single model represented poorly across the
transition. The percentile range has been split into four bands (0-25,
25-50, 50-75, 75-100); this is band 1, matching the scope the original paper
(arXiv:2403.16344 Part I) actually focused on -- cell-edge / max-min fairness.

ONE model must serve every (K, percentile-in-band) setting:

    powers = model(A, Kq)          # A: [batch, K, B, B]  ->  powers: [batch, K, B]

The held-out benchmark is a fixed GRID of (K, percentile) cells within the band;
each cell is scored as the ratio of the model's mean SLqP to the FULL-POWER mean
SLqP on the same pinned drops, and HELDOUT_SCORE is the mean ratio over the grid.
1.000 = the trivial full-power floor everywhere. All physics (geometry, path
loss, fading, noise, powers) is byte-identical to the certified base campaign.

The pure-ML inference contract carries over (see evaluate below). QFT's
convergence was independently re-verified for this band's cells before this
file was finalized (this band never touches the sum-rate degenerate case that
caused problems in the prior full-range campaign) -- see the note near
INF_BUDGET_S and program.md for the verification.
"""

import math
import time

import numpy as np
import torch

# ----------------------------------------------------------------------------
# Parameters (physics identical to the certified v2 evaluator)
# ----------------------------------------------------------------------------
K_MAX = 10                     # users per cell may be 1..K_MAX
B     = 7                      # cells

P_T  = 10 ** (43 / 10) * 1e-3           # 43 dBm -> 19.953 W (per-user max power)
W_HZ = 20e6                             # bandwidth
N0_PSD_dBm_Hz = -150.0                  # as USED in the reference code
N_0  = 10 ** (N0_PSD_dBm_Hz / 10) * 1e-3 * W_HZ   # 2.0e-11 W
d0    = 0.392                            # reference distance [m]
ALPHA = 3.76                             # path-loss exponent
R     = 1000.0                           # BS spacing = 2R = 2000 m

# ----------------------------------------------------------------------------
# Geometry: identical to v2 (certified against the MATLAB in verify_model.py)
# ----------------------------------------------------------------------------
_th = 1j * np.pi / 3
BS_POS = R * 2 * np.array(
    [0, 1.0, np.exp(_th), np.exp(2*_th), np.exp(3*_th), np.exp(4*_th), np.exp(5*_th)]
)
_t1 = 2 * R * (2.0 + np.exp(_th))
def _rot(z, k): return z * np.exp(1j * np.pi / 3 * k)
_t2 = _rot(_t1, 1)
SHIFTS = np.array([0, _t1, -_t1, _t2, -_t2, _t1 - _t2, _t2 - _t1])
assert np.allclose(np.abs(SHIFTS[1:]), 2 * R * np.sqrt(7)), \
    "wrap-around shifts must sit at the 7-cell tessellation distance 2R*sqrt(7)"

_d_center = np.abs(BS_POS[:, None] - (BS_POS[None, :] + SHIFTS[:, None, None]))
_pick = np.argmin(_d_center, axis=0)
CELL_IMAGE = BS_POS[None, :] + SHIFTS[_pick]                                    # [B,Bp]


def _drop_users_hex(n, rng):
    rad = R * 2 / np.sqrt(3)
    tri = rng.integers(0, 6, n)
    a0 = np.deg2rad(30 + 60 * tri); a1 = np.deg2rad(30 + 60 * (tri + 1))
    v0 = rad * np.exp(1j * a0); v1 = rad * np.exp(1j * a1)
    u1 = rng.random(n); u2 = rng.random(n)
    fl = u1 + u2 > 1; u1[fl] = 1 - u1[fl]; u2[fl] = 1 - u2[fl]
    return u1 * v0 + u2 * v1


def sample_channels(batch, K, seed=None):
    """
    A[batch, K, B, B]:  A[., k, b, bp] = |h|^2 power gain from BS bp to the k-th
    user of cell b, for K users per cell (1 <= K <= K_MAX). Fresh uniform drops
    and fresh Rayleigh fading per batch element; MATLAB cell-level wraparound.
    """
    assert 1 <= K <= K_MAX, f"K={K} outside 1..{K_MAX}"
    rng = np.random.default_rng(seed)
    offs = _drop_users_hex(batch * B * K, rng).reshape(batch, B, K)
    upos = np.transpose(BS_POS[None, :, None] + offs, (0, 2, 1))               # [batch,K,B]
    dist = np.abs(upos[:, :, :, None] - CELL_IMAGE[None, None, :, :])          # [batch,K,B,Bp]
    pathloss = (dist / d0) ** (-ALPHA)
    fading = rng.exponential(1.0, size=(batch, K, B, B))
    return torch.from_numpy((pathloss * fading).astype(np.float32))


# ----------------------------------------------------------------------------
# Metric: SLqP for arbitrary K_q, in Mbps
# ----------------------------------------------------------------------------
def slqp_rate(power, A, Kq):
    """
    power : [batch, K, B] in [0, P_T];  A : [batch, K, B, B];  Kq : int in [1, K*B].
    returns [batch] sum of the smallest Kq per-user rates (Mbps). Differentiable.
    """
    KB = A.shape[1] * A.shape[2]
    assert 1 <= Kq <= KB, f"Kq={Kq} outside 1..{KB}"
    Adiag = torch.diagonal(A, dim1=2, dim2=3)
    signal = power * Adiag
    Pcell = power.sum(dim=1)
    total = torch.einsum('tkbc,tc->tkb', A, Pcell)
    sinr = signal / (total - signal + N_0)
    rate = torch.log2(1.0 + sinr).reshape(A.shape[0], KB)
    worst = torch.topk(rate, Kq, largest=False).values
    return worst.sum(dim=1) * (W_HZ / 1e6)


# ----------------------------------------------------------------------------
# The pinned held-out GRID
# ----------------------------------------------------------------------------
KS_TEST = (1, 2, 4, 6, 8, 10)
# LOW-PERCENTILE BAND (this campaign): 0-25% of K*B, matching the original
# paper's actual focus (cell-edge / max-min fairness through the first
# quartile). The full 0-100% range was split into four bands after discovering
# a qualitative regime shift around the 50%->100% transition (egalitarian/
# scheduling-like policies at low percentiles vs. concentrated/greedy policies
# near sum-rate) that a single model could not represent well across the whole
# range. This is band 1 of 4 (0-25%, 25-50%, 50-75%, 75-100%). p50 and sum are
# INTENTIONALLY EXCLUDED from this campaign's grid.
PCTS = (("min", 0.0), ("p10", 0.10), ("p25", 0.25))
N_TEST = 250                                   # drops per K (shared across its Kq cells)


def kq_of(frac, KB):
    return 1 if frac == 0.0 else max(1, math.ceil(frac * KB))


def settings_for(K):
    """Deduplicated [(label, Kq)] for one K (small K collapses some percentiles)."""
    KB, out, seen = K * B, [], set()
    for label, frac in PCTS:
        kq = kq_of(frac, KB)
        if kq not in seen:
            seen.add(kq); out.append((label, kq))
    return out


TEST = {K: sample_channels(N_TEST, K, seed=5000 + K) for K in KS_TEST}
GRID = [(K, label, kq) for K in KS_TEST for (label, kq) in settings_for(K)]

# Full-power reference per grid cell (the 1.000 floor of every ratio).
FULL_REF = {}
for K in KS_TEST:
    _pf = torch.full((N_TEST, K, B), P_T)
    for label, kq in settings_for(K):
        FULL_REF[(K, kq)] = slqp_rate(_pf, TEST[K], kq).mean().item()
del _pf


# ----------------------------------------------------------------------------
# Evaluation + the pure-ML inference contract
# ----------------------------------------------------------------------------
INF_BUDGET_S = 10.0     # total forward time over the WHOLE grid (this band has
                        # ~17 cells x 250 drops, vs. the full-range campaign's 29).
                        # QFT needs many minutes on a grid this size at 10 iters/
                        # solve, so the budget still forces a large, deliberate
                        # speedup and structurally rules out per-instance
                        # optimization at test time. No measured model has come
                        # close to this ceiling in practice.


def evaluate(model, verbose=True):
    """
    Mean over the pinned grid of (model SLqP / full-power SLqP), each cell on
    its own pinned drops. 1.000 = full-power floor. Enforces the contract:
    total forward time <= INF_BUDGET_S; parameters unchanged by evaluation.
    """
    model.eval()
    before = [p.detach().clone() for p in model.parameters()]
    ratios, t_inf = {}, 0.0
    for K in KS_TEST:
        A = TEST[K]
        for label, kq in settings_for(K):
            t0 = time.perf_counter()
            with torch.no_grad():
                p = model(A, kq)
            t_inf += time.perf_counter() - t0
            if p.shape != (A.shape[0], K, B):
                raise SystemExit(f"model output shape {tuple(p.shape)} != "
                                 f"{(A.shape[0], K, B)} at K={K}")
            with torch.no_grad():
                val = slqp_rate(p, A, kq).mean().item()
            ratios[(K, label, kq)] = val / FULL_REF[(K, kq)]
    print(f"INFERENCE_S {t_inf:.3f}")
    if t_inf > INF_BUDGET_S:
        raise SystemExit(
            f"inference contract violated: {t_inf:.1f}s > {INF_BUDGET_S}s over "
            f"the grid (per-instance optimization at test time is banned)")
    after = list(model.parameters())
    if len(before) != len(after) or any(
            not torch.equal(b, a.detach()) for b, a in zip(before, after)):
        raise SystemExit("inference contract violated: model parameters changed "
                         "during evaluation (test-time fitting is banned)")
    if verbose:
        labels = [lbl for lbl, _ in PCTS]
        print("GRID (model/full-power):      " + "  ".join(f"{l:>5}" for l in labels))
        for K in KS_TEST:
            cells = {lbl: f"{ratios[(K, lbl, kq)]:.3f}"
                     for lbl, kq in settings_for(K)}
            row = "  ".join(f"{cells.get(lbl, '  -- '):>5}" for lbl in labels)
            print(f"  K={K:>2} ({K*B:>2} users):          " + row)
    return float(np.mean(list(ratios.values())))


if __name__ == "__main__":
    print(f"grid: {len(GRID)} cells  |  K in {KS_TEST}  |  {N_TEST} drops/K  "
          f"|  budget {INF_BUDGET_S}s")
    print("full-power SLqP (Mbps) per cell -- the denominators:")
    for K in KS_TEST:
        cells = "  ".join(f"{lbl}:{FULL_REF[(K, kq)]:8.3f}"
                          for lbl, kq in settings_for(K))
        print(f"  K={K:>2}: {cells}")
