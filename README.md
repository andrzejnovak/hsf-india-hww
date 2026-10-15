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

Every team's last submitted commit, on the same sealed datasets each run. Each dataset has its own planted
signal strength μ and background rate, and the Higgs mass at 125 GeV (within 2 GeV): a measurement of μ at a known
mass. The mass you report (optional in RESULT.json) is for the record, not scored.

- **score** = 100 × **sensitivity** × **robustness**, the order of the table. The template fit, `baseline-mt`,
  with honest error bars would score 100.
- **sensitivity**: the template fit's error bar on μ over yours, averaged over the scored datasets (**σ μ** is
  yours). Bigger means a tighter measurement.
- **robustness**: 1 over your pull width where that is above 1. The pulls are (reported − planted) / σ, and
  their spread (**pull width μ**) is 1 when the error bars are honest, so robustness 1.00 is honest; a wrong
  slope, a bias, or error bars smaller than the measurement supports all widen the pulls and lower it. Shrinking
  σ alone moves nothing: what it gains in sensitivity it loses in robustness.
- **slope μ**: a line through reported against planted; 1 is a measurement, 0 a constant.

Each team's plot, reported against planted on the scored datasets and coloured by the pull, is folded under
the table. Yours, on datasets you make: `python3 make_suite.py suites/mine --n 10 --seed 7`, then
`python3 score_suite.py ~/hack-<team>/submission suites/mine --plot-main mu.png`. The sealed plots show the
planted values: for seeing where your score is lost, not for fitting a correction to (a hard-coded answer,
against the rules). The three levels of the [GUIDE](GUIDE.md) are milestones for when you present, not columns here.
