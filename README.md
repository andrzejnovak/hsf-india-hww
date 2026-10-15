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

Anywhere with git and Python 3 (numpy and scipy, and the CPU build of PyTorch for a network evaluated with torch:
`pip install numpy scipy matplotlib scikit-learn` and `pip install torch --index-url https://download.pytorch.org/whl/cpu` if missing).
The data is simulated and all of it is in this repository.

```bash
git clone https://github.com/andrzejnovak/hsf-india-hww.git && cd hsf-india-hww
python3 -m sbh check --task hww --submission examples/stub_submission --dataset datasets/public/seed_000 --determinism
bash new_team.sh <team-name>        # -> ~/hack-<team-name>: the task files, the public dataset, submission/ with the stub
```

The second line must end with `determinism (second clean run): PASS`. Your pipeline is `~/hack-<team-name>/submission/`:
a `run.sh <dataset_dir> <output_dir>` that writes `<output_dir>/RESULT.json`, Python 3 with numpy, scipy and torch (CPU, for
inference) at run time.
`examples/template_fit/` is a working start (a fit of the transverse and dilepton masses for the Higgs mass, the
signal strength and the background rate; the leaderboard's `baseline-mt`); `examples/tutorial_nn/` plugs the ML
tutorial's network into the same fit (`baseline-nn`), `examples/nn_fit/` does the same with better features, and
`examples/torch_nn/` trains that network with PyTorch and evaluates it either way, with numpy from `model.json` or with torch
from the state_dict as JSON (`fit_torch.py`).
The self-check with `--submission ~/hack-<team-name>/submission` tells you whether yours runs.

## Submit

```bash
cd ~/hack-<team-name> && git add -A && git commit -m "our submission" && git push   # to a public GitHub repository of yours
cd ~/hsf-india-hww && bash submit.sh <team-name> <git url>                         # the self-check, is it pushed, then one line
```

Post that line, `<team-name>: <git url> @ <commit>`, as a comment on
[issue 1](https://github.com/andrzejnovak/hsf-india-hww/issues/1) (`submit.sh` posts it for you when `gh` is
logged in; the team name is letters, digits, `-` and `_` only, no spaces, and a comment without a line in that
form gets a reply saying so instead of a leaderboard row). Every comment starts the sealed run: your commit on
datasets nobody has seen, and the table below a few minutes later. Submit as often as you like; the last comment
per team before the deadline is the one that counts.
Pushing from a Binder session needs a GitHub token as the password (github.com → Settings → Developer settings →
Personal access tokens); from your laptop it is your usual push.

## Leaderboard

<!-- leaderboard -->
_Sealed run of 2026-10-15 08:23 UTC: 51 datasets nobody has seen, every submission on the [issue](https://github.com/andrzejnovak/hsf-india-hww/issues/1) at its last comment; μ scored on the 26 of them whose Higgs mass stays at 125 GeV, the score 100 × the template fit's calibrated error bar over yours (see below). The same with the mass, on all 51: [LEADERBOARD_MASS.md](LEADERBOARD_MASS.md). Each team's plot on those datasets is folded at the bottom. A submission gets at most 60 minutes per run; the datasets left when they run out count as not run._

| # | team          | score | error μ | slope μ | pull width μ | submitted                                         |
|---|---------------|-------|---------|---------|--------------|---------------------------------------------------|
| 1 | Hack-404      | –     | –       | 1.03    | 0.99         | 0909838 (2026-10-15T07:18:42Z by shaurya-cd)      |
| 2 | thursday-demo | –     | –       | 1.03    | 0.99         | a71c823 (2026-10-15T04:53:09Z by andrzejnovak)    |
| 3 | Horizon       | –     | –       | 1.02    | 1.13         | d8bc180 (2026-10-15T04:50:58Z by AnushSingla)     |
| 4 | Pilots        | –     | –       | 1.02    | 1.13         | 3be629b (2026-10-15T06:57:34Z by ishandhull2953)  |
| 5 | baseline-mt   | –     | –       | 1.02    | 1.13         | 52ad132 (2026-10-14T08:25:55Z by andrzejnovak)    |
| 6 | team_x        | –     | –       | 1.02    | 1.13         | 4bc4ba0 (2026-10-15T07:12:25Z by gauriaggarwal31) |
| 7 | baseline-nn   | –     | –       | 1.02    | 1.24         | c549c28 (2026-10-14T08:26:00Z by andrzejnovak)    |

<details>
<summary><b>Recovery plots</b>, per team: reported against planted μ on the fixed-mass datasets, each point coloured by how far it sits from the truth in units of its own error bar</summary>

**Hack-404** (score –)

![Hack-404](results/Hack-404.png)

**thursday-demo** (score –)

![thursday-demo](results/thursday-demo.png)

**Horizon** (score –)

![Horizon](results/Horizon.png)

**Pilots** (score –)

![Pilots](results/Pilots.png)

**baseline-mt** (score –)

![baseline-mt](results/baseline-mt.png)

**team_x** (score –)

![team_x](results/team_x.png)

**baseline-nn** (score –)

![baseline-nn](results/baseline-nn.png)

</details>
<!-- /leaderboard -->

### Reading the table

One row per team, its last submitted commit, run on the same sealed datasets every time (a fixed suite, so runs
are comparable). Each dataset has its own planted signal strength μ and background rate; on half of them the
Higgs mass stays at 125 GeV (within 2 GeV), on the other half it is anywhere between 100 and 150 GeV. This table
scores μ on the fixed-mass datasets alone: how precisely the μ you report follows the planted μ, with error bars
that mean what they say. The three levels of the [GUIDE](GUIDE.md) are milestones for when you present, not
columns here. The mass is not scored here, so a wrong mass cannot drag μ along; μ and the mass together, on
every dataset, is a page of its own, [LEADERBOARD_MASS.md](LEADERBOARD_MASS.md). The two plots below show what is
measured, for the three example pipelines on 30 datasets of their own (`make_plots.py`); yours look the same with
`score_suite.py`, see the end.

- **score**, the one number the table is ordered by: 100 × the template fit's calibrated error on μ over yours.
  The template fit (`baseline-mt`) scores 100; 115 means your honest error bar is 15 % tighter than its. No
  self-check pass (hygiene, the contract, determinism on the public dataset: the check `submit.sh` runs), or
  nothing run, is no score.
- **error μ**: your calibrated error bar, the mean error bar you quote times the pull width where that is
  above 1: the error bar you would have had to quote to be honest. A genuinely more precise measurement, its
  pulls still of width 1, is a smaller calibrated error and a higher score: that is the game. Quoting a smaller
  error bar than you have gains nothing, the pull width grows by the same factor; padding it costs you directly;
  a bias or a wrong slope widens the pulls and costs too.
- **slope**: a line fitted through reported against planted (the recovery plot below). 1 means your
  measurement moves one-for-one with the truth; 0 that you report the same value whatever was planted.
- **pull width**: the spread of (reported − planted) / your error bar across the datasets (the pulls plot
  below). 1 means your error bars are honest; above it they are too small, below it too large.
- **submitted**: the commit that ran, and when and by whom it was posted.

![reported against planted: the three example pipelines on 30 datasets](figs/recovery.png)

![the pulls of the same run: (reported − planted) / error, against a width-1 Gaussian](figs/pulls.png)

Your own plots and both tables' numbers, on datasets you make (the sealed ones are drawn from the same priors,
with a seed you don't have; `make_suite.py` plants the same mix of fixed and moving masses):

```bash
python3 make_suite.py suites/mine --n 10 --seed 7
python3 score_suite.py ~/hack-<team>/submission suites/mine --plot recovery.png --plot-main mu.png
```

Each team's plot on the sealed datasets themselves, μ on the fixed-mass ones, is folded under the table after
every run; the one with the mass on every dataset is on the [mass page](LEADERBOARD_MASS.md). They show the
planted values of the sealed suite, the same suite every run: use them to see where your score is lost, not to
fit a correction to them; a correction tuned to those points is a hard-coded answer, against the rules and plain
to see in your code.
