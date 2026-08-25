import numpy as np, time, json
from prepare import TEST, FULL_REF, settings_for, KS_TEST, P_T
from qft_reference import qft_solve, _prep, true_slqp_mbps
out = {}
for K in KS_TEST:
    A_all = TEST[K].double().numpy()
    for label, kq in settings_for(K):
        t0 = time.time(); qs = []
        for i in range(A_all.shape[0]):
            M, d, SM, KB = _prep(A_all[i])
            _, traj = qft_solve(A_all[i], kq)
            qs.append(traj[-1])
        r = float(np.mean(qs)) / FULL_REF[(K, kq)]
        out[f"{K}|{label}|{kq}"] = r
        print(f"K={K:>2} {label:>4} Kq={kq:>2}  QFT/full = {r:.4f}   ({time.time()-t0:.0f}s)", flush=True)
        json.dump(out, open("qft_grid_partial.json", "w"))
print(f"\nQFT_SCORE (mean over {len(out)} cells) = {np.mean(list(out.values())):.6f}")
