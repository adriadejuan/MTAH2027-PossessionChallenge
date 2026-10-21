# Ball possession from automatic tracking: technical note

*Team: <team_name>. Draft, to be finished before submission.*

## Approach

A deterministic, rule-based state machine over the tactical ATD, with the Smart Tagging set
pieces as restart markers. No learning: with 17 labelled possessions there is nothing to train
on that would not overfit, and every rule below can be explained in football terms.

We chose to optimise the structure of possessions rather than per-frame accuracy, because the
ranking metric (spellF1) matches whole possessions. The guiding check throughout was whether the
output has the shape of a real match: the right number and length of possessions and a plausible
share of time with the ball in play.

## Assets

| asset | used | how |
|---|---|---|
| tactical ATD | yes | ball and player positions and speeds; the only per-frame input |
| Smart Tagging | yes, set pieces only | restart markers, and the restarting team when tagged |
| homography | no | the ATD is already in pitch coordinates |
| tactical / broadcast video | no | TODO: say whether anyone tried it |
| broadcast ATD | no | not synchronised to the tactical timeline |

## Possession rules

- **Dead ball.** Out of play until a restart; ball missing with players clearly slower than their
  open-play speed; ball still with slow players; players slow for a sustained stretch; the
  published pre-kickoff stretch of the second half. "Open-play speed" is measured per half so the
  thresholds follow the players as they tire.
- **Restarts.** A tagged set piece ends a stoppage; the 10 s after it are live unless the ball is
  out, so a ball lost in a crowded box after a corner is not a stoppage.
- **Control.** The nearest tracked player within 2 m of a visible ball.
- **Change of possession.** Only when the other team holds the ball for 15 of the last 50 frames
  and twice as often as the holder, dated to their first touch (within 1 m). Loose balls,
  deflections and aerial duels therefore stay with the last team in control, as the challenge
  defines it.
- **Clean-up.** Very short possessions between same-team spells and very short stoppages inside a
  spell are absorbed.
- **Timing calibration.** Labels are shifted 12 frames later; the reference's changes landed a
  consistent 10-18 frames after ours.

## How we got here

| version | change | public macroF1 | public spellF1 | full match: ball in play (1st / 2nd) |
|---|---|---|---|---|
| organisers' baseline (smooth50) | nearest player + majority vote | 0.621 | 0.326 | 66.9% / 68.5% |
| v1 | stoppages + hysteresis | 0.710 | 0.444 | |
| v2 | Smart Tagging restarts, touch dating | 0.711 | 0.456 | |
| v3 | slow players = stoppage, timing shift | 0.762 | 0.432 | |
| v4 | restarts protect play, sliding-window switches | 0.802 | 0.494 | 52.7% / 48.9% |
| v6 | thresholds relative to open-play speed | 0.792 | 0.480 | 55.2% / 50.5% |
| v7 (submitted) | stricter ball-missing rule | 0.806 | 0.494 | 57.7% / 51.6% |

Tried and dropped: a fixed slow-player threshold (fires more as players tire); a per-half
threshold taken over all frames (stoppages drag it down and it missed a 20 s stoppage); the
Smart Tagging intervals copied as labels (they are phases, not possessions).

The public window cannot separate these versions statistically (bootstrap intervals overlap), so
each change was kept for a football reason checked against the public window's segments and the
full-match plausibility checks, never for its public score alone.

## Limitations

- Possession changes still land late on some transitions (up to ~3 s), mostly with pressing
  defenders near the ball.
- Very short possessions (under 1.5 s) are smoothed away by design.
- In the second half, set pieces are still labelled dead shortly after their tagged window more
  often than in the first (44% vs 19% of them), mostly from the ball-missing rule.
- Throw-ins carry no team in the tagging, so the team after a throw-in comes from the first touch.
- TODO: what we would do with more time (pass/touch detection from ball velocity changes;
  using the video where the ball is missing).

## Reproduction

See `team/README.md`. `python team/possession.py --name <team_name>` regenerates the submitted
file exactly.
