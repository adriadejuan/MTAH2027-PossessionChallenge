"""possession: label every frame of the tactical video H / A / D, from the ATD and Smart Tagging.

Run from the repository root (MTAH_DATA_DIR pointing at the data bundle):

    python team/possession.py --name <team_name>
    python evaluation/validate_submission.py evaluation/results/<team_name>.csv

numpy only, deterministic: the same data and parameters always write the same file. The first run
parses the raw ATD (a minute or two) and caches the arrays as <MTAH_DATA_DIR>/cache/atd_arrays.npz,
inside the data folder on purpose, because derived frame-level tracking must never be published.

How it works
------------
1. Dead ball (D). A frame is dead when any of these holds (first rule listed wins the label):
   - out of play: the ball is seen beyond a line (OUT_MARGIN) for OUT_MIN_FRAMES; it stays dead
     until a restart -- a kick (ball faster than KICK_SPEED inside the pitch), the ball seen
     moving faster than RESUME_SPEED for RESUME_FRAMES, or players running again for RERUN_FRAMES;
   - ball missing: unseen for GAP_DEAD_FRAMES while the players' rolling speed is below
     GAP_SLOW_RATIO x their open-play speed on GAP_SLOW_SHARE of the gap;
   - ball still: seen below STILL_BALL_SPEED for STILL_DEAD_FRAMES with slow players;
   - slow players: rolling speed below SLOW_DEAD_RATIO x open-play speed for SLOW_DEAD_FRAMES;
   - the published pre-kickoff stretch of the second half (110808..113741).
   "Open-play speed" is the half's median rolling player speed over frames where the ball is seen
   moving faster than RESUME_SPEED, so the thresholds follow the players as they tire.
2. Smart Tagging set pieces. Each tagged restart (first kick inside the tagged window, else its
   midpoint) ends a stoppage: the quiet stretch before it (ball unseen, out, or slow with slow
   players; up to SETPIECE_BACK frames) is dead, and for POST_RESTART frames after it only an
   out-of-pitch ball can be dead. A Team tag, when present, says who has the ball after it.
3. Control. A frame belongs to a team when its nearest tracked player is within CONTROL_DIST of
   a visible ball.
4. Possession with hysteresis. Inside each live stretch the team in possession changes only when
   the other team has SWITCH_FRAMES control frames within the last SWITCH_WINDOW frames and at
   least twice the holder's. The change is dated to the challenger's first frame within
   TOUCH_DIST of the ball. Loose balls and deflections therefore stay with the last team.
5. Clean-up. Possessions shorter than MIN_POSS_FRAMES between same-team spells are absorbed,
   short dead stretches inside one team's spell are dropped, and short bursts of play between
   two stoppages become dead.
6. Every label is shifted SHIFT_FRAMES later, a calibration of where the reference draws a
   change (the public window's changes landed a consistent 10-18 frames after ours).

Every threshold is in PARAMS and can be overridden on the command line (--switch-frames 20 ...).
After writing the file it prints which rule produced the dead-ball frames, per half.
TECHNICAL.md explains why each rule exists and what was tried and dropped.
"""

import argparse
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))  # the organisers' baseline and tagging loader
from games import DATA_DIR, GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from baseline import team_names, team_of_each_track, write_submission  # noqa: E402
import load_smart_tagging  # noqa: E402

# Params
PARAMS = {
    "control_dist": 2.0,       # m, nearest player this close to the ball "has" it
    "touch_dist": 1.0,         # m, a switch is back-dated to the challenger's first frame this close
    "setpiece_back": 500,      # frames (20 s) a set-piece dead spell may extend backwards
    "resume_speed": 3.0,       # m/s, a ball faster than this inside the pitch is back in play
    "slow_dead_ratio": 0.5,    # x the half's open-play median player speed: below this ...
    "slow_dead_frames": 100,   # ... for 4 s -> stoppage
    "slow_window": 75,         # frames (3 s) the player speed is averaged over
    "shift_frames": 12,        # every label moved this much later (see docstring)
    "switch_frames": 15,       # opponent control frames needed to change possession (0.6 s)
    "switch_window": 50,       # ... within this many recent frames (2 s), and twice the holder's
    "post_restart": 250,       # frames (10 s) after a set-piece restart where only "out" is dead
    "rerun_frames": 250,       # frames (10 s) of running players that end an out-of-play spell
    "out_margin": 1.0,         # m beyond the line before the ball counts as out (tracking noise)
    "out_min_frames": 3,       # consecutive out frames before declaring dead
    "kick_speed": 4.0,         # m/s, ball speed that marks a restart
    "kick_min_frames": 3,      # consecutive fast in-pitch frames to accept a restart
    "resume_frames": 25,       # or: this many in-pitch frames of a moving ball (kick unseen)
    "gap_dead_frames": 100,    # ball missing at least this long (4 s) ...
    "slow_players": 2.0,       # m/s, rolling player speed above this counts as "running"
    "gap_slow_ratio": 0.75,    # gap is a stoppage if players are below this x open-play speed
    "gap_slow_share": 0.6,     # ... on at least this share of the gap
    "still_ball_speed": 0.5,   # m/s, a ball this slow is "still"
    "still_dead_frames": 75,   # still ball this long (3 s) with slow players -> stoppage
    "min_poss_frames": 38,     # possessions shorter than 1.5 s between same-team spells absorbed
    "min_dead_frames": 25,     # dead stretches under 1 s inside one team's spell are dropped
}
KICKOFF_2ND = 113742  # FAQ.md: graded from 110808, all D until this kickoff

# Which rule marked a frame dead, for the breakdown printed at the end. First rule wins.
RULES = ("", "out of play", "ball missing", "ball still", "slow players", "pre-kickoff",
         "set-piece run-up")

HOME, AWAY, NONE = 0, 1, -1
DEAD, ALIVE = 1, 0


def parse_atd(raw_path, n_frames, n_slots, pitch):
    """Raw ATD -> player x/y/speed (n_frames x n_slots) and ball x/y/speed, in metres. NaN = unseen."""
    length, width = pitch
    px = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    py = np.full_like(px, np.nan)
    ps = np.full_like(px, np.nan)
    ball = np.full((n_frames, 3), np.nan, dtype=np.float32)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            f = int(parts[0]) - 1
            if not 0 <= f < n_frames:
                continue
            for slot, group in enumerate(parts[1].split(";")):
                if slot >= n_slots or not group or group[0] == "N":
                    continue
                values = group.split(",")
                px[f, slot] = float(values[0]) * length
                py[f, slot] = float(values[1]) * width
                if len(values) > 2 and values[2][:1] != "N":
                    ps[f, slot] = float(values[2])
            values = parts[2].split(",")
            if values[0][:1] != "N":
                ball[f, 0] = float(values[0]) * length
                ball[f, 1] = float(values[1]) * width
                if len(values) > 2 and values[2][:1] != "N":
                    ball[f, 2] = float(values[2])
    return px, py, ps, ball


def load_arrays(game=GAME, data_dir=None, refresh=False):
    """Parse once, then reuse the cache. The cache lives in the data folder, never in the repo."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    teams = team_of_each_track(meta_path)
    cache_dir = os.path.join(data_dir or DATA_DIR, "cache")
    cache = os.path.join(cache_dir, "atd_arrays.npz")
    if os.path.exists(cache) and not refresh:
        arrays = np.load(cache)
        return teams, arrays["px"], arrays["py"], arrays["ps"], arrays["ball"]

    print("parsing {} (first run only) ...".format(os.path.basename(raw_path)))
    px, py, ps, ball = parse_atd(raw_path, source_frames("tactical", game), len(teams),
                                 GAMES[game]["pitch"])
    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(cache, px=px, py=py, ps=ps, ball=ball)
    print("cached -> {}".format(cache))
    return teams, px, py, ps, ball


def ball_speed(ball, fps):
    """The ATD ball speed where given, else finite differences of position."""
    speed = ball[:, 2].astype(np.float64)
    step = np.full(len(ball), np.nan)
    step[1:] = np.hypot(np.diff(ball[:, 0]), np.diff(ball[:, 1])) * fps
    return np.where(np.isnan(speed), step, speed)


def runs(mask):
    """[(start, end_exclusive)] of True runs."""
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return list(zip(edges[::2], edges[1::2]))


def features(teams, px, py, ps, ball, p):
    """Per-frame nearest-player team and distance, ball speed, mean player speed."""
    bx, by = ball[:, 0:1], ball[:, 1:2]
    dist = np.hypot(px - bx, py - by)
    dist = np.where(np.isnan(dist), np.inf, dist)
    nearest = np.argmin(dist, axis=1)
    near_dist = dist[np.arange(len(dist)), nearest]
    near_team = np.where(np.isfinite(near_dist), teams[nearest], NONE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # frames with nobody tracked
        player_speed = np.nanmean(ps, axis=1)
    return near_team, near_dist, ball_speed(ball, p["fps"]), player_speed


def dead_ball(ball, bspeed, pspeed, pitch, p):
    """1 where the ball is judged dead, from rules a-e in the module docstring."""
    length, width = pitch
    n = len(ball)
    seen = ~np.isnan(ball[:, 0])
    m = p["out_margin"]
    out = seen & ((ball[:, 0] < -m) | (ball[:, 0] > length + m) |
                  (ball[:, 1] < -m) | (ball[:, 1] > width + m))
    inside = seen & ~out
    kick = inside & (bspeed > p["kick_speed"])

    # a + b: out of play until a restart kick
    out_confirmed = np.zeros(n, dtype=bool)
    for start, end in runs(out):
        if end - start >= p["out_min_frames"]:
            out_confirmed[start] = True
    kick_confirmed = np.zeros(n, dtype=bool)
    for start, end in runs(kick):
        if end - start >= p["kick_min_frames"]:
            kick_confirmed[start] = True

    tracked = ~np.isnan(pspeed)
    filled = np.where(tracked, pspeed, 0.0)
    w = p["slow_window"]
    csum = np.concatenate(([0.0], np.cumsum(filled)))
    ccnt = np.concatenate(([0], np.cumsum(tracked)))
    lo = np.clip(np.arange(n) - w // 2, 0, n)
    hi = np.clip(np.arange(n) + w // 2 + 1, 0, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        rolling = (csum[hi] - csum[lo]) / (ccnt[hi] - ccnt[lo])
    active = rolling > p["slow_players"]

    # The kick itself is often unseen (the ball is missing 41% of the time), so a dead spell also
    # ends once the ball has been seen moving inside the pitch for resume_frames frames.
    moving = inside & (bspeed > p["resume_speed"])
    dead = np.zeros(n, dtype=bool)
    reason = np.zeros(n, dtype=np.int8)
    # A third way out, so that a missed kick cannot leave the match dead indefinitely: the
    # players are running again (rolling speed above slow_players) for rerun_frames frames.
    state, moved, ran = False, 0, 0
    for f in range(n):
        if out_confirmed[f]:
            state, moved, ran = True, 0, 0
        elif state:
            moved += moving[f]
            ran += active[f]
            if kick_confirmed[f] or moved >= p["resume_frames"] or ran >= p["rerun_frames"]:
                state = False
        dead[f] = state
    reason[dead] = 1

    slow = np.nan_to_num(pspeed, nan=0.0) < p["slow_players"]
    # The players' speed in open play, per half: the reference the slow-player rules scale by.
    reference = np.full(n, np.nan)
    for _, first, last in half_video_blocks(GAME):
        half = slice(first, last + 1)
        reference[half] = np.nanmedian(rolling[half][moving[half]]) if moving[half].any() \
            else np.nanmedian(rolling[half])
    # c: long ball-missing gaps with slow players
    with np.errstate(invalid="ignore"):
        gap_slow = rolling < p["gap_slow_ratio"] * reference
    for start, end in runs(~seen):
        if end - start >= p["gap_dead_frames"] and gap_slow[start:end].mean() >= p["gap_slow_share"]:
            dead[start:end] = True
            reason[start:end][reason[start:end] == 0] = 2
    # d: a still ball with slow players for a long time
    still = seen & (bspeed < p["still_ball_speed"]) & slow
    for start, end in runs(still):
        if end - start >= p["still_dead_frames"]:
            dead[start:end] = True
            reason[start:end][reason[start:end] == 0] = 3
    # f: players slow for a sustained stretch -> stoppage, whatever the ball is doing
    with np.errstate(invalid="ignore"):
        slow_dead = rolling < p["slow_dead_ratio"] * reference
    for start, end in runs(slow_dead):
        if end - start >= p["slow_dead_frames"]:
            dead[start:end] = True
            reason[start:end][reason[start:end] == 0] = 4
    # e: published pre-kickoff stretch of the 2nd half
    second_start = half_video_blocks(GAME)[1][1]
    dead[second_start:KICKOFF_2ND] = True
    reason[second_start:KICKOFF_2ND][reason[second_start:KICKOFF_2ND] == 0] = 5
    quiet = out | ~seen | (seen & (bspeed < p["resume_speed"]) & ~active)
    return dead, kick_confirmed, quiet, out, reason


def set_piece_restarts(kick_confirmed, home_name, game=GAME):
    """[(window_start, restart_frame, team)] for every tagged set piece. team is NONE for N/A.

    The tag marks a ~5 s window around the restart, not the instant, so the restart is the first
    confirmed kick inside the window, or its midpoint when the kick itself was not seen.
    """
    out = []
    for event in load_smart_tagging.load(game):
        if (event["code"] or "").strip().lower() != "set pieces":
            continue
        start, end = event["start_frame"], max(event["end_frame"], event["start_frame"] + 1)
        kicks = np.flatnonzero(kick_confirmed[start:end])
        restart = start + int(kicks[0]) if len(kicks) else (start + end) // 2
        tag = (event["tags"].get("Team") or "").strip().lower()
        team = NONE if tag in ("", "n/a") else (HOME if tag == home_name.strip().lower() else AWAY)
        out.append((start, restart, team))
    return out


def apply_set_pieces(dead, quiet, out, restarts, p, reason):
    """Dead over the quiet frames (ball unseen, out or still) running up to each restart.

    The tagged window is only a rough bracket, so it is not copied across: the dead spell is
    whatever stretch of quiet ball directly precedes the restart, capped at setpiece_back frames.
    """
    seeds = {}
    for start, restart, team in restarts:
        back = restart
        while back > 0 and restart - back < p["setpiece_back"] and quiet[back - 1]:
            back -= 1
        reason[back:restart][~dead[back:restart]] = 6
        dead[back:restart] = True
        after = slice(restart, restart + p["post_restart"])
        dead[after] = out[after]  # the restart put the ball in play
        if team != NONE:
            seeds[restart] = team
    return dead, seeds


def possession(control, near_dist, dead, seeds, p):
    """Team per frame with hysteresis, inside each alive stretch. NONE where unknown.

    seeds maps a restart frame to the team the tagging says restarts there; an alive stretch that
    begins within a second of a seed starts with that team instead of whoever touches first.
    """
    n = len(control)
    team = np.full(n, NONE, dtype=np.int8)
    seed_frames = np.array(sorted(seeds), dtype=np.int64)
    for start, end in runs(~dead):
        current, switches = NONE, []
        if len(seed_frames):
            near = seed_frames[np.abs(seed_frames - start) <= p["fps"]]
            if len(near):
                current = seeds[int(near[np.argmin(np.abs(near - start))])]
                switches.append((start, current))
        recent = []  # control frames inside the sliding window, as (frame, team)
        for f in range(start, end):
            c = control[f]
            if c == NONE:
                continue
            if current == NONE:
                current = c
                switches.append((start, c))  # the restart belongs to whoever touches first
                continue
            recent.append((f, c))
            while recent[0][0] <= f - p["switch_window"]:
                recent.pop(0)
            if c == current:
                continue
            theirs = [g for g, t in recent if t == c]
            ours = len(recent) - len(theirs)
            if len(theirs) >= p["switch_frames"] and len(theirs) >= 2 * ours:
                touches = [g for g in theirs if near_dist[g] <= p["touch_dist"]]
                current = c
                switches.append((touches[0] if touches else theirs[0], c))
                recent = []
        bounds = [s for s, _ in switches] + [end]
        for (s, c), e in zip(switches, bounds[1:]):
            team[s:e] = c
    return team


def labels_from(team, dead):
    codes = np.where(dead, 2, np.where(team == HOME, 0, np.where(team == AWAY, 1, 3)))
    return codes.astype(np.int8)  # 0 H, 1 A, 2 D, 3 X


def clean(codes, p):
    """Absorb short possessions between same-team spells, drop short dead stretches likewise."""
    for target, min_len in ((2, p["min_dead_frames"]), (0, p["min_poss_frames"]),
                            (1, p["min_poss_frames"])):
        changed = True
        while changed:
            changed = False
            for start, end in runs(codes == target):
                if end - start >= min_len or start == 0 or end == len(codes):
                    continue
                before, after = codes[start - 1], codes[end]
                if before == after and before in (0, 1) and before != target:
                    codes[start:end] = before
                    changed = True
                elif before == after == 2 and target in (0, 1):
                    codes[start:end] = 2  # a blip of play between two stoppages
                    changed = True
    return codes


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="possession", help="output file name, use the team name")
    parser.add_argument("--no-tagging", action="store_true", help="ignore the Smart Tagging")
    parser.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results"))
    parser.add_argument("--refresh", action="store_true", help="re-parse the ATD")
    for key, value in PARAMS.items():
        parser.add_argument("--" + key.replace("_", "-"), type=type(value), default=value)
    args = parser.parse_args()
    p = {key: getattr(args, key) for key in PARAMS}
    p["fps"] = GAMES[GAME]["fps"]
    pitch = GAMES[GAME]["pitch"]

    teams, px, py, ps, ball = load_arrays(refresh=args.refresh)
    near_team, near_dist, bspeed, pspeed = features(teams, px, py, ps, ball, p)
    control = np.where(near_dist <= p["control_dist"], near_team, NONE).astype(np.int8)
    dead, kick_confirmed, quiet, out, reason = dead_ball(ball, bspeed, pspeed, pitch, p)
    seeds = {}
    if not args.no_tagging:
        restarts = set_piece_restarts(kick_confirmed, team_names(require(["atd_meta"])[0])[0])
        dead, seeds = apply_set_pieces(dead, quiet, out, restarts, p, reason)
        print("set pieces {}  with a team tag {}".format(len(restarts), len(seeds)))
    team = possession(control, near_dist, dead, seeds, p)
    codes = clean(labels_from(team, dead), p)
    k = p["shift_frames"]
    if k > 0:
        codes = np.concatenate((np.repeat(codes[:1], k), codes[:-k]))

    graded = np.zeros(len(codes), dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        graded[first:last + 1] = True
    codes[~graded] = 3
    labels = np.array(["H", "A", "D", "X"])[codes]

    path = write_submission(os.path.join(args.out, args.name + ".csv"), labels)
    g = labels[graded]
    spells = sum(1 for lab in (0, 1) for _ in runs(codes[graded] == lab))
    print("graded frames  H {}  A {}  D {}  X {}".format(*[(g == x).sum() for x in "HADX"]))
    print("possessions {}  mean {:.1f} s".format(
        spells, np.isin(g, ["H", "A"]).sum() / max(spells, 1) / p["fps"]))
    print("-> {}".format(os.path.relpath(path, ROOT)))
    rule_breakdown(codes, reason, labels)


def rule_breakdown(codes, reason, labels):
    """Which rule made the dead-ball frames of the file, per half. Aggregates only.

    Uses the labels as written (after cleaning and the shift), shifting the rule array the same
    way so each D frame is attributed to the rule that produced it.
    """
    k = PARAMS["shift_frames"] if "shift_frames" in PARAMS else 0
    shifted = np.concatenate((np.repeat(reason[:1], k), reason[:-k])) if k > 0 else reason
    set_pieces = [(e["start_frame"], e["end_frame"]) for e in load_smart_tagging.load()
                  if (e["code"] or "").strip().lower() == "set pieces"]
    print("\ndead-ball frames by rule (share of the half's D frames)")
    for half, first, last in half_video_blocks(GAME):
        sl = slice(first, last + 1)
        d = labels[sl] == "D"
        parts = []
        for r in range(len(RULES)):
            count = int((d & (shifted[sl] == r)).sum())
            if count:
                parts.append("{} {:.0f}%".format(RULES[r] or "cleaning", 100.0 * count / d.sum()))
        print("  half {}: {}".format(half, ", ".join(parts)))
        late = {}
        for start, end in set_pieces:
            f = end + 5 * GAMES[GAME]["fps"]
            if first <= start and f <= last and labels[f] == "D":
                name = RULES[shifted[f]] or "cleaning"
                late[name] = late.get(name, 0) + 1
        print("    set pieces still dead 5 s after their window, by rule: {}".format(
            ", ".join("{} {}".format(k, v) for k, v in sorted(late.items())) or "none"))


if __name__ == "__main__":
    main()
