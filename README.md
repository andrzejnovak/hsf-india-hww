# Hackathon: how much Higgs is in the data?

A Higgs boson decaying to two W bosons and then two leptons and two neutrinos, on top of the WW continuum, in
simulated 13 TeV data; how much of it there is, is unknown. You build a pipeline on the one public dataset,
with the simulation beside it to train on; the organisers run it, unchanged, on sealed datasets with hidden
truths and score whether your signal strength follows the truth with honest error bars.
The pitch: [the slides](https://andrzejnovak.github.io/talks/2026-10-13-hsf-india/index.html?profile=hack).
Everything else, the big picture for non-physicists, the three levels, the loop with the agent, where the
classifier goes, the physics in pictures, debugging, what gets scored, the rules: [`GUIDE.md`](GUIDE.md). The task itself: [`PROMPT.md`](PROMPT.md),
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
`examples/template_fit/` is a working start (a fit of the transverse and dilepton masses for the Higgs mass, the
signal strength and the background rate; the leaderboard's `baseline-mt`); `examples/tutorial_nn/` plugs the ML
tutorial's network into the same fit (`baseline-nn`), and `examples/nn_fit/` does the same with better features. The self-check with `--submission ~/hack-<team-name>/submission` tells you whether
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
_Sealed run of 2026-10-15 05:02 UTC: 21 datasets nobody has seen, every submission on the [issue](https://github.com/andrzejnovak/hsf-india-hww/issues/1) at its last comment. Recovery plots follow after the deadline. A submission gets at most 40 minutes per run; the datasets left when they run out count as not run._

| # | team          | level μ | level mass | slope μ | pull width μ | slope mass | pull width mass | false disc. | self-check | submitted                                      |
|---|---------------|---------|------------|---------|--------------|------------|-----------------|-------------|------------|------------------------------------------------|
| 1 | thursday-demo | 2       | 1          | 1.01    | 0.93         | 1.05       | 2.80            | 0/3         | PASS       | a71c823 (2026-10-15T04:53:09Z by andrzejnovak) |
| 2 | Horizon       | 2       | 1          | 1.03    | 0.90         | 1.15       | 1.55            | 0/3         | PASS       | d8bc180 (2026-10-15T04:50:58Z by AnushSingla)  |
| 3 | baseline-mt   | 2       | 1          | 1.03    | 0.90         | 1.15       | 1.55            | 0/3         | PASS       | 52ad132 (2026-10-14T08:25:55Z by andrzejnovak) |
| 4 | baseline-nn   | 2       | 1          | 1.04    | 0.84         | 1.11       | 1.95            | 0/3         | PASS       | c549c28 (2026-10-14T08:26:00Z by andrzejnovak) |
<!-- /leaderboard -->
