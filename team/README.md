# Team possession pipeline

Our entry for More than a Hack 2027, Phase 1. It sits in `team/` inside a copy of the organisers'
repository and reuses its loaders (`games.py`, `starter/baseline.py`, `starter/load_smart_tagging.py`).

## Setup (once per machine)

```
git clone <our team repo>
cd <repo>
conda create -n fcb python=3.11 -y
conda activate fcb
pip install -r team/requirements-lock.txt
conda env config vars set MTAH_DATA_DIR="<folder containing tactical\ and broadcast\>"
conda deactivate && conda activate fcb
```

The match data is never committed. Each member downloads it and points `MTAH_DATA_DIR` at it.

## Reproduce the submission

```
python team/possession.py --name <team_name>
python evaluation/validate_submission.py evaluation/results/<team_name>.csv
```

Same data + same code = the same file, byte for byte (numpy only, no randomness).

## Check a change

```
python evaluation/run_evaluation.py --bootstrap 1000                 # public 5 minutes: correctness only
python team/sanity_full_match.py evaluation/results/<new>.csv evaluation/results/<team_name>.csv
python team/diagnose_public.py evaluation/results/<new>.csv          # per-segment view of the 5 minutes
```

The public window has 17 possessions and cannot rank versions (bootstrap intervals overlap).
A change is worth keeping when it has a football reason, does not hurt the public window, and makes
the full-match checks more plausible. All three scripts print aggregates only.

## Working together

One branch per idea, a pull request per change, and paste the three outputs above in the PR.
Parameters live in `PARAMS` at the top of `team/possession.py` and can be overridden from the
command line, e.g. `--switch-frames 20`, so most experiments need no code change.
