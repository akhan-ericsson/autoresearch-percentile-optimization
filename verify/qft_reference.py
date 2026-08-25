"""
qft_reference.py  --  Python/CVXPY port of the authors' QFT algorithm,
generalized for campaign v3: works for any K (users/cell) and any Kq in [1, K*B].
Gold-reference optimizer + optional label generator for supervised learning.

    p, traj = qft_solve(A, Kq)     # A: numpy [K, B, B] float64, one drop

Algorithm identical to the MATLAB (quadratic transform, alternating y / convex-p,
init 0.5*P_T, 10 iterations, per-user box constraint); noise-normalized for
solver conditioning only.

Usage:
    python3 qft_reference.py                  # small grid demo
    K=8 KQ=14 DROPS=10 python3 qft_reference.py
"""

import os
import time
import warnings

import numpy as np
import cvxpy as cp
import torch

from prepare import B, P_T, N_0, W_HZ, sample_channels


def _prep(A_kbb):
    K = A_kbb.shape[0]
    KB = K * B
    G = np.asarray(A_kbb, dtype=np.float64) / N_0
    M = G.reshape(KB, B)
    d = np.einsum('kbb->kb', G).reshape(KB)
    SM = np.zeros((B, KB))
    for k in range(K):
        for b in range(B):
            SM[b, k * B + b] = 1.0
    return M, d, SM, KB


def true_slqp_mbps(p_flat, M, d, SM, Kq):
    sig = p_flat * d
    tot = M @ (SM @ p_flat)
    rate = np.log2(1.0 + sig / (tot - sig + 1.0))
    return W_HZ * np.sort(rate)[:Kq].sum() / 1e6


def qft_solve(A_kbb, Kq, iters=None, verbose=False):
    """Run QFT on one channel realization A[K,B,B]. Returns (p_flat, trajectory).

    iters default: 10 for Kq < K*B (matches the paper/MATLAB; confirmed converged
    at 10 by direct trajectory inspection -- see qft_true_convergence.py). For
    Kq == K*B (pure sum-rate), the objective degenerates to cp.sum instead of
    cp.sum_smallest (see below), which empirically converges far slower: running
    to 150 iterations and checking the trajectory had flattened showed 10
    iterations understates the true value by up to ~2.7x at large K. Default is
    100 for that case unless the caller overrides it.
    """
    M, d, SM, KB = _prep(A_kbb)
    assert 1 <= Kq <= KB
    if iters is None:
        iters = 100 if Kq == KB else 10
    p_val = np.full(KB, 0.5 * P_T)

    p = cp.Variable(KB, nonneg=True)
    th = cp.Variable(KB)
    c1 = cp.Parameter(KB, nonneg=True)
    y2 = cp.Parameter(KB, nonneg=True)
    sig_e = cp.multiply(d, p)
    intf_e = M @ (SM @ p) - sig_e
    lhs = 1.0 + cp.multiply(c1, cp.sqrt(p)) - cp.multiply(y2, intf_e + 1.0)
    # Kq == KB is the sum-rate case: sum_smallest over ALL entries is just the
    # sum, and CVXPY's sum_smallest canonicalization crashes on the zero-size
    # complement set in exactly that case -- so special-case it.
    obj = cp.sum(th) if Kq == KB else cp.sum_smallest(th, Kq)
    prob = cp.Problem(cp.Maximize(obj), [cp.exp(th) <= lhs, p <= P_T])

    traj = [true_slqp_mbps(p_val, M, d, SM, Kq)]
    for _ in range(iters):
        sig = p_val * d
        intf = M @ (SM @ p_val) - sig
        y = np.sqrt(sig) / (intf + 1.0)
        c1.value = 2.0 * y * np.sqrt(d)
        y2.value = y ** 2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            solved = False
            try:
                prob.solve(solver=cp.CLARABEL)
                solved = prob.status == "optimal" and p.value is not None
            except cp.error.SolverError:
                solved = False
            if not solved:
                try:
                    prob.solve(solver=cp.SCS)
                    solved = (prob.status in ("optimal", "optimal_inaccurate")
                              and p.value is not None)
                except cp.error.SolverError:
                    solved = False
        if not solved:
            break
        p_val = np.clip(p.value, 0.0, P_T)
        traj.append(true_slqp_mbps(p_val, M, d, SM, Kq))
        if verbose:
            print(f"    iter: SLqP = {traj[-1]:.3f} Mbps")
    return p_val, traj


def direct_opt(A_kbb, Kq, steps=1200, lr=0.08):
    """Per-instance Adam on the exact objective -- an independent cross-check."""
    M, d, SM, KB = _prep(A_kbb)
    Mt, dt, St = torch.tensor(M), torch.tensor(d), torch.tensor(SM)
    z = torch.zeros(KB, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([z], lr)
    for _ in range(steps):
        pw = P_T * torch.sigmoid(z)
        sig = pw * dt
        tot = Mt @ (St @ pw)
        rate = torch.log2(1.0 + sig / (tot - sig + 1.0))
        loss = -torch.topk(rate, Kq, largest=False).values.sum()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return true_slqp_mbps((P_T * torch.sigmoid(z)).numpy(), M, d, SM, Kq)


if __name__ == "__main__":
    K = int(os.environ.get("K", 4))
    Kq = int(os.environ.get("KQ", 7))
    drops = int(os.environ.get("DROPS", 5))
    A_all = sample_channels(drops, K, seed=777)
    q, f, t = [], [], []
    for i in range(drops):
        A = A_all[i].double().numpy()
        M, d, SM, KB = _prep(A)
        f.append(true_slqp_mbps(np.full(KB, P_T), M, d, SM, Kq))
        t0 = time.time()
        _, traj = qft_solve(A, Kq)
        t.append(time.time() - t0)
        q.append(traj[-1])
    print(f"K={K} Kq={Kq} over {drops} drops: full={np.mean(f):.3f}  "
          f"QFT={np.mean(q):.3f} Mbps  ({np.mean(t):.2f} s/drop)")
