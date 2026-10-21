"""diagnose_public: where a submission goes wrong on the public 5-minute window.

Run from the repo root (needs the cache possession.py writes):

    python team/diagnose_public.py evaluation/results/possession.csv

Prints one line per ground-truth segment (21 of them): what the truth says, what the submission
says, and what the tracking looked like over that stretch -- how often the ball was seen, whether
it was out of the pitch, how fast the ball and the players moved, and whether a Smart Tagging set
piece overlaps it. Then one line per true change of possession with the predicted switch time.

Output is aggregate numbers only, so it is safe to paste into a chat. No positions are printed.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES  # noqa: E402
import load_smart_tagging  # noqa: E402
from possession import PARAMS, ball_speed, features, load_arrays  # noqa: E402

GT_PATH = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
FPS = GAMES[GAME]["fps"]


def segments(labels):
    """[(start, end_exclusive, label)] over the graded (non-X) frames."""
    out, start = [], None
    for f in range(len(labels) + 1):
        lab = labels[f] if f < len(labels) else None
        if start is not None and lab != labels[start]:
            if labels[start] != "X":
                out.append((start, f, labels[start]))
            start = None
        if start is None and lab is not None:
            start = f
    return out


def mix(labels):
    counts = pd.Series(list(labels)).value_counts()
    return " ".join("{}{:.0f}".format(k, 100.0 * v / len(labels)) for k, v in counts.items())


def main():
    pred = pd.read_csv(sys.argv[1])["label"].astype(str).str.strip().str.upper().values
    gt = pd.read_csv(GT_PATH)["label"].astype(str).str.strip().str.upper().values

    teams, px, py, ps, ball = load_arrays()
    p = dict(PARAMS, fps=FPS)
    near_team, near_dist, bspeed, pspeed = features(teams, px, py, ps, ball, p)
    length, width = GAMES[GAME]["pitch"]
    seen = ~np.isnan(ball[:, 0])
    out = seen & ((ball[:, 0] < 0) | (ball[:, 0] > length) | (ball[:, 1] < 0) | (ball[:, 1] > width))

    tags = [e for e in load_smart_tagging.load()
            if (e["code"] or "").strip().lower() == "set pieces"]

    print("GT segments (frames are video frames; time is seconds from the first graded frame)")
    print("{:>3} {:>6} {:>6} {:>5} {:>2}  {:<16} {:>5} {:>5} {:>6} {:>6} {:>5}  {}".format(
        "#", "start", "end", "sec", "gt", "predicted mix", "seen%", "out%", "ballv", "plyv",
        "ctrl%", "set piece"))
    first = segments(gt)[0][0]
    for i, (s, e, lab) in enumerate(segments(gt)):
        sl = slice(s, e)
        ctrl = (near_dist[sl] <= p["control_dist"]).mean() * 100
        overlap = ["{} {}".format(t["tags"].get("Type", "?"), t["tags"].get("Team", "?"))
                   for t in tags if t["start_frame"] < e and t["end_frame"] > s]
        print("{:>3} {:>6} {:>6} {:>5.1f} {:>2}  {:<16} {:>5.0f} {:>5.0f} {:>6.1f} {:>6.1f} {:>5.0f}  {}".format(
            i, s, e, (e - s) / FPS, lab, mix(pred[sl]), seen[sl].mean() * 100,
            out[sl].mean() * 100, np.nanmedian(bspeed[sl]) if seen[sl].any() else float("nan"),
            np.nanmedian(pspeed[sl]), ctrl, "; ".join(overlap)))

    print("\nGT changes of label, and the nearest predicted change of the same kind (+ = late)")
    for f in range(first + 1, len(gt)):
        if gt[f] != gt[f - 1] and "X" not in (gt[f], gt[f - 1]):
            window = range(max(f - 75, 1), min(f + 76, len(pred)))
            hits = [g - f for g in window if pred[g] == gt[f] and pred[g - 1] == gt[f - 1]]
            nearest = min(hits, key=abs) if hits else None
            print("  {:>6}  {}->{}  {}".format(
                f, gt[f - 1], gt[f], "{:+d} frames".format(nearest) if hits else "not found"))


if __name__ == "__main__":
    main()
