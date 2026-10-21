"""sanity_full_match: checks on a whole-match submission that do not use the public ground truth.

Run from the repo root, with one or more submissions side by side:

    python team/sanity_full_match.py evaluation/results/possession.csv evaluation/results/baseline_smooth50.csv

The public 5 minutes cannot rank versions, so these are the checks left: does the output look like
a football match? Each line is one statistic per half, for each file:

  ball in play     share of graded frames labelled H or A
  possessions      count, and how many are suspiciously short (< 2 s) or long (> 60 s)
  possession s     median / 90th percentile / longest duration
  stoppages        count of dead-ball spells, and their median / longest duration
  home share       H / (H + A)
  restarts after D share of tagged set pieces with a dead ball in the 10 s before the restart:
                   a free kick, corner, throw-in or goal kick always follows a stoppage, so this
                   should be close to 100%. Low means stoppages are being missed.
  D after restart  share of set pieces still labelled dead 5 s after the tagged window ends:
                   should be close to 0%. High means restarts are being missed.

Reference points, for orientation only (general knowledge, not measured on this match): the ball
is typically in play for roughly 55-65% of a top-flight match, and stoppages over a minute
(injuries, VAR, goal celebrations) happen a handful of times. Use them to spot an outlier, not as
targets. The public window itself has 17 possessions in 5 minutes and 20% dead ball.

Output is aggregate numbers only, so it is safe to paste into a chat.
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
from games import GAME, GAMES, half_video_blocks  # noqa: E402
import load_smart_tagging  # noqa: E402

FPS = GAMES[GAME]["fps"]


def runs(mask):
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return list(zip(edges[::2], edges[1::2]))


def forward_fill_x(labels):
    """What the scorer does to X before matching possessions: repeat the last answer."""
    out = labels.copy()
    for i in range(1, len(out)):
        if out[i] == "X":
            out[i] = out[i - 1]
    for i in range(len(out) - 2, -1, -1):  # back-fill the head of the half
        if out[i] == "X":
            out[i] = out[i + 1]
    return out


def half_stats(labels, set_pieces, first, last):
    lab = forward_fill_x(labels[first:last + 1])
    n = len(lab)
    alive = np.isin(lab, ["H", "A"])
    poss = [(s, e) for team in "HA" for s, e in runs(lab == team)]
    pdur = np.array([(e - s) / FPS for s, e in poss]) if poss else np.zeros(1)
    ddur = np.array([(e - s) / FPS for s, e in runs(lab == "D")] or [0.0])
    h, a = (lab == "H").sum(), (lab == "A").sum()

    before, after, counted = 0, 0, 0
    for start, end in set_pieces:
        if not (first + 10 * FPS <= start and end + 5 * FPS <= last):
            continue
        counted += 1
        before += (lab[max(start - first - 10 * FPS, 0):end - first] == "D").any()
        after += lab[end - first + 5 * FPS] == "D"
    return {
        "ball in play %": "{:.1f}".format(100.0 * alive.sum() / n),
        "possessions": "{} (<2s {}, >60s {})".format(
            len(poss), int((pdur < 2).sum()), int((pdur > 60).sum())),
        "possession s": "{:.1f} / {:.1f} / {:.1f}".format(
            np.median(pdur), np.percentile(pdur, 90), pdur.max()),
        "stoppages": "{} (median {:.1f} s, longest {:.0f} s)".format(
            len(runs(lab == "D")), np.median(ddur), ddur.max()),
        "home share %": "{:.1f}".format(100.0 * h / max(h + a, 1)),
        "restarts after D %": "{:.0f} of {}".format(100.0 * before / max(counted, 1), counted),
        "D after restart %": "{:.0f}".format(100.0 * after / max(counted, 1)),
    }


def main():
    paths = sys.argv[1:]
    if not paths:
        sys.exit(__doc__)
    set_pieces = [(e["start_frame"], e["end_frame"]) for e in load_smart_tagging.load()
                  if (e["code"] or "").strip().lower() == "set pieces"]

    table = {}
    for path in paths:
        labels = pd.read_csv(path)["label"].astype(str).str.strip().str.upper().values
        name = os.path.splitext(os.path.basename(path))[0]
        for half, first, last in half_video_blocks(GAME):
            for key, value in half_stats(labels, set_pieces, first, last).items():
                table.setdefault((key, half), {})[name] = value

    names = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    width = max(36, *[len(n) + 2 for n in names])
    print("{:<22}{:<6}".format("statistic", "half") + "".join(n.ljust(width) for n in names))
    for (key, half), values in table.items():
        print("{:<22}{:<6}".format(key, half) + "".join(values[n].ljust(width) for n in names))


if __name__ == "__main__":
    main()
