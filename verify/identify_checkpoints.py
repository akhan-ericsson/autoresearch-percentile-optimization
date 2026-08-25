#!/usr/bin/env python3
"""
identify_checkpoints.py -- work out which experiment a checkpoint came from.

Every checkpoint `train.py` writes carries the score it earned:

    torch.save({"state_dict": ..., "score": score, "arch": ...}, "last_model.pt")

so the file identifies itself. Point this at a file or a directory.

    python identify_checkpoints.py families/
    python identify_checkpoints.py last_model.pt
"""
import glob
import os
import sys

import torch

# Scores banked by the campaign, from the changelog in train.py.
KNOWN = {
    1.477473: "experiment 81  <-- THE CHAMPION (paper reports 1.4775)",
    1.478752: "experiment 85  (later than the paper's scope)",
    1.478613: "experiment 84  (later than the paper's scope)",
    1.477648: "experiment 82  (two-shot; later than the paper's scope)",
    1.477360: "experiment 82's W_scale=0.75 probe (reverted)",
    1.475555: "experiment 52",
    1.473999: "experiment 49",
    1.463448: "experiment 65 (QFT-as-teacher, reverted)",
    1.476605: "a single-core RECONSTRUCTION -- not the champion",
}


def report(path):
    try:
        try:
            b = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            # torch < 1.13 has no weights_only argument
            b = torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"{os.path.basename(path):<44} unreadable ({e})")
        return None
    score = b.get("score")
    arch = b.get("arch", "?")
    if score is None:
        print(f"{os.path.basename(path):<44} no score recorded (arch {arch})")
        return None
    label = KNOWN.get(round(float(score), 6), "not a banked score in the log")
    print(f"{os.path.basename(path):<44} {score:.6f}  {arch:<10} {label}")
    return float(score)


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "last_model.pt"
    paths = (sorted(p for p in glob.glob(os.path.join(target, "*"))
                     if os.path.isfile(p))
             if os.path.isdir(target) else [target])
    if not paths:
        sys.exit(f"no files found under {target}")

    print(f"{'file':<44} {'score':>9}  {'arch':<10} identification")
    print("-" * 100)
    scores = [report(p) for p in paths]

    if any(s is not None and round(s, 6) == 1.477473 for s in scores):
        print("\nFound experiment 81. Copy it to verify/last_model.pt and run:")
        print("    python verify.py --ckpt last_model.pt")
    else:
        print("\nNo checkpoint scoring 1.477473 here, so the champion's weights")
        print("are not in this set. Regenerate them instead -- check out the")
        print("experiment-81 commit and run `python train.py` once; it retrains")
        print("in one to two minutes and rewrites last_model.pt. On the campaign")
        print("machine that should reproduce 1.477473 exactly.")


if __name__ == "__main__":
    main()
