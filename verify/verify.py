#!/usr/bin/env python3
"""
verify.py -- reproduce the paper's held-out claim end to end.

USAGE
    python verify.py                        # train from scratch, then score
    python verify.py --ckpt last_model.pt   # score shipped weights instead
    python verify.py --qft                  # + sampled QFT reference check
    python qft_grid.py                      # full QFT bar, all 17 cells (slow)

WHAT IT PRINTS
    the per-cell (model / full-power) ratio table, HELDOUT_SCORE and
    INFERENCE_S, straight from the immutable evaluator.

        HELDOUT_SCORE  ->  paper reports 1.4775   (experiment 81)
        QFT_SCORE      ->  paper reports 1.4850

WHY THIS IS CHECKABLE
    The held-out set is NOT shipped as data. `prepare.py` regenerates it from
    pinned seeds (TEST seeds 5000..5010), so there is no curated file to
    distrust. `prepare.py` is the immutable evaluator; its SHA-256 is printed
    below and recorded in the top-level README. If the hash does not match, the
    judge has been altered and the numbers mean nothing.

REPRODUCIBILITY
    Bit-exact scores depend on the torch version and thread count: reduction
    order in einsum/matmul differs and compounds over 2000 Adam steps
    differentiated through a 40-pass fixed point. See VERIFICATION.md -- a
    single-core run reproduced 1.476605 against the campaign's 1.477473 on a
    12-core machine, a difference of 0.06%.
"""
import argparse
import hashlib
import os
import sys
import time


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sampled_qft(n_drops):
    """A fast, sampled version of the reference bar.

    The full bar over all 17 cells x 250 pinned drops takes roughly 40 minutes
    single-core (`python qft_grid.py`); the K=10 min cell alone is ~11 minutes.
    This samples the first `n_drops` of each cell instead.
    """
    import numpy as np
    import torch
    from prepare import TEST, KS_TEST, settings_for, slqp_rate, P_T, B
    from qft_reference import qft_solve

    print(f"\nsampled QFT reference ({n_drops} of 250 drops per cell)")
    ratios, t0 = [], time.time()
    for K in KS_TEST:
        sub = TEST[K][:n_drops]
        A_all = sub.double().numpy()
        # The full-power denominator MUST be computed on the SAME subsample as
        # the numerator; using prepare.FULL_REF (all 250 drops) against a
        # short QFT sample mixes two different drop sets and inflates the ratio.
        pf = torch.full((n_drops, K, B), P_T)
        for label, kq in settings_for(K):
            full = float(slqp_rate(pf, sub, kq).mean())
            vals = [qft_solve(A_all[i], kq)[1][-1] for i in range(n_drops)]
            r = float(np.mean(vals)) / full
            ratios.append(r)
            print(f"  K={K:>2} {label:>4} Kq={kq:>2}   QFT/full = {r:.4f}",
                  flush=True)
    print(f"\nQFT_SCORE (sampled) = {np.mean(ratios):.6f}   "
          f"({time.time() - t0:.0f} s)")
    print("[paper reports 1.4850; a full-grid recomputation gave 1.485645]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None,
                    help="score this checkpoint instead of training")
    ap.add_argument("--qft", action="store_true",
                    help="also run a sampled QFT reference check")
    ap.add_argument("--qft-drops", type=int, default=20,
                    help="drops per cell for the sampled QFT check")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)
    sys.path.insert(0, here)

    for need in ("prepare.py", "train.py"):
        if not os.path.exists(os.path.join(here, need)):
            sys.exit(f"ERROR: {need} not found next to verify.py. "
                     f"See MANIFEST.md for the files this repository needs.")

    import torch                                            # noqa: E402
    print("evaluator : prepare.py")
    print(f"SHA-256   : {sha256('prepare.py')}")
    print(f"torch     : {torch.__version__}   threads: {torch.get_num_threads()}")
    print()

    import train as T                                       # noqa: E402

    if args.ckpt is None:
        # train.py's own entry point: builds the teacher cache, trains 2000
        # Adam steps, evaluates on the pinned grid, prints HELDOUT_SCORE and
        # saves last_model.pt.
        print("training from scratch -- this is the paper's inner loop.")
        print("(~1-2 min on a multicore CPU; ~17 min single-core)\n")
        t0 = time.time()
        T.main()
        print(f"\ntotal wall clock: {time.time() - t0:.0f} s")
    else:
        from prepare import evaluate                        # noqa: E402
        blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        model = T.PowerNet()
        model.load_state_dict(blob["state_dict"])
        print(f"loaded checkpoint : {args.ckpt}")
        if "score" in blob:
            print(f"recorded score    : {blob['score']:.6f}")
        print()
        score = evaluate(model)
        print(f"HELDOUT_SCORE {score:.6f}")

    print("\n[paper reports HELDOUT_SCORE = 1.4775 for experiment 81]")

    if args.qft:
        sampled_qft(args.qft_drops)


if __name__ == "__main__":
    main()
