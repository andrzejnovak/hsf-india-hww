# Hackathon: find the Higgs in H → WW

A Higgs boson of unknown mass and strength, decaying to two W bosons and then two leptons and two neutrinos, on
top of the WW continuum, in simulated 13 TeV data. You build a pipeline on the one public dataset, with the
simulation beside it to train on; the organisers run it, unchanged, on sealed datasets with hidden truths and
score whether your mass and signal strength follow the truth with honest error bars.
The pitch: [the slides](https://andrzejnovak.github.io/talks/2026-10-13-hsf-india/index.html?profile=hack).
Everything else, the three levels, the loop with the agent, where the classifier goes, the physics in pictures,
debugging, what gets scored, the rules: [`GUIDE.md`](GUIDE.md). The task itself: [`PROMPT.md`](PROMPT.md),
[`DATASET.md`](DATASET.md), [`EVAL_CONTRACT.md`](EVAL_CONTRACT.md).

## Setup

Anywhere with git and Python 3 (numpy and scipy; `pip install numpy scipy matplotlib scikit-learn` if missing).
The data is simulated and all of it is in this repository.

```bash
git clone https://github.com/andrzejnovak/hsf-india-hww.git && cd hsf-india-hww
python3 -m sbh check --task hww --submission examples/stub_submission --dataset datasets/public/seed_000 --determinism
bash new_team.sh <team-name>        # -> ~/hack-<team-name>: the task files, the public dataset, submission/ with the stub
```

The second line must end with `determinism (second clean run): PASS`. Your pipeline is `~/hack-<team-name>/submission/`:
a `run.sh <dataset_dir> <output_dir>` that writes `<output_dir>/RESULT.json`, Python 3 with numpy and scipy only.
`examples/template_fit/` is a working start (a kinematic fit); `examples/nn_fit/` is the same fit with a
classifier plugged in. The self-check with `--submission ~/hack-<team-name>/submission` tells you whether
yours runs.

## Submit

```bash
cd ~/hack-<team-name> && git add -A && git commit -m "our submission" && git push   # to a public GitHub repository of yours
cd ~/hsf-india-hww && bash submit.sh <team-name> <git url>                         # the self-check, is it pushed, then one line
```

Post that line, `<team-name>: <git url> @ <commit>`, as a comment on
[issue 1](https://github.com/andrzejnovak/hsf-india-hww/issues/1) (`submit.sh` posts it for you when `gh` is
logged in). Every comment starts the sealed run: your commit on datasets nobody has seen, and the table below a few
minutes later. Submit as often as you like; the last comment per team before the deadline is the one that counts.
Pushing from a Binder session needs a GitHub token as the password (github.com → Settings → Developer settings →
Personal access tokens); from your laptop it is your usual push.

## Leaderboard

<!-- leaderboard -->
_Sealed run of 2026-10-14 07:50 UTC: 21 datasets nobody has seen, every submission on the [issue](https://github.com/andrzejnovak/hsf-india-hww/issues/1) at its last comment. Recovery plots follow after the deadline._

| # | team     | level | slope mH | slope μ | pull width mH | pull width μ | false disc. | power | recovery | self-check | submitted                                      |
|---|----------|-------|----------|---------|---------------|--------------|-------------|-------|----------|------------|------------------------------------------------|
| 1 | demo-ref | 2     | 1.03     | 1.08    | 1.07          | 0.97         | 0/5         | 1.00  | 21/21    | PASS       | bb580e9 (2026-10-14T07:48:50Z by andrzejnovak) |
<!-- /leaderboard -->
