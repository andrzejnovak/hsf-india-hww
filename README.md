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
_Sealed run of 2026-10-15 07:08 UTC: 31 datasets nobody has seen, every submission on the [issue](https://github.com/andrzejnovak/hsf-india-hww/issues/1) at its last comment. The table scores μ on the 16 datasets whose Higgs mass stays at 125 GeV; folded under it, μ and the mass on all 31, the mass moving between 100 and 150 GeV. Each team's recovery plot, reported against planted and coloured by the pull, is folded at the bottom. A submission gets at most 60 minutes per run; the datasets left when they run out count as not run._

| # | team          | level | slope μ | pull width μ | false disc. | self-check | submitted                                        |
|---|---------------|-------|---------|--------------|-------------|------------|--------------------------------------------------|
| 1 | thursday-demo | 2     | 1.07    | 1.12         | 0/2         | PASS       | a71c823 (2026-10-15T04:53:09Z by andrzejnovak)   |
| 2 | Horizon       | 2     | 1.05    | 1.15         | 0/2         | PASS       | d8bc180 (2026-10-15T04:50:58Z by AnushSingla)    |
| 3 | baseline-mt   | 2     | 1.05    | 1.15         | 0/2         | PASS       | 52ad132 (2026-10-14T08:25:55Z by andrzejnovak)   |
| 4 | baseline-nn   | 2     | 1.06    | 1.14         | 0/2         | PASS       | c549c28 (2026-10-14T08:26:00Z by andrzejnovak)   |
| 5 | Pilots        | 0     | –       | –            | –           | FAIL       | a63b823 (2026-10-15T06:37:19Z by ishandhull2953) |

<details>
<summary><b>With the mass</b>: μ and the mass on every dataset, the mass moving</summary>

| # | team          | level μ | level mass | slope μ | pull width μ | slope mass | pull width mass | false disc. | self-check | submitted                                        |
|---|---------------|---------|------------|---------|--------------|------------|-----------------|-------------|------------|--------------------------------------------------|
| 1 | thursday-demo | 2       | 2          | 1.04    | 0.98         | 0.94       | 1.02            | 0/10        | PASS       | a71c823 (2026-10-15T04:53:09Z by andrzejnovak)   |
| 2 | Horizon       | 2       | 2          | 1.02    | 1.07         | 0.95       | 1.39            | 0/10        | PASS       | d8bc180 (2026-10-15T04:50:58Z by AnushSingla)    |
| 3 | baseline-mt   | 2       | 2          | 1.02    | 1.07         | 0.95       | 1.39            | 0/10        | PASS       | 52ad132 (2026-10-14T08:25:55Z by andrzejnovak)   |
| 4 | baseline-nn   | 2       | 2          | 1.02    | 1.20         | 0.98       | 1.37            | 0/10        | PASS       | c549c28 (2026-10-14T08:26:00Z by andrzejnovak)   |
| 5 | Pilots        | 0       | 0          | –       | –            | –          | –               | –           | FAIL       | a63b823 (2026-10-15T06:37:19Z by ishandhull2953) |

</details>

<details>
<summary><b>Recovery plots</b>, per team: reported against planted on the sealed datasets, each point coloured by how far it sits from the truth in units of its own error bar</summary>

**thursday-demo** (level 2)

![thursday-demo](results/thursday-demo.png)

**Horizon** (level 2)

![Horizon](results/Horizon.png)

**baseline-mt** (level 2)

![baseline-mt](results/baseline-mt.png)

**baseline-nn** (level 2)

![baseline-nn](results/baseline-nn.png)

</details>
<!-- /leaderboard -->

### Reading the tables

One row per team, its last submitted commit, run on the same sealed datasets every time (a fixed suite, so runs
are comparable). Each dataset has its own planted signal strength μ and background rate; on half of them the
Higgs mass stays at 125 GeV (within 2 GeV), on the other half it is anywhere between 100 and 150 GeV. The table
scores μ on the fixed-mass datasets alone: whether the μ you report follows the planted μ, with error bars that
mean what they say; the mass is not scored there, so a wrong mass cannot drag μ along. Folded under it, the
table with the mass: μ and the mass, each by the same rule, on every dataset, where the two have to move
together. The two plots below show what is measured, for the three example pipelines on 30 datasets of their
own (`make_plots.py`); yours look the same with `score_suite.py`, see the end.

- **level**, the score. **0**, no measurement: the self-check fails, fewer than 90 % of the datasets run, or
  the slope is below 0.8 (the stub reports a constant: slope 0). **1**, follows the truth: slope ≥ 0.8. **2**,
  and honest: pull width within [0.5, 1.5] and no discovery claimed on a dataset without a Higgs. **3** is
  judged when you present: how the agent got there. Rows are ordered by level, then by how close the slope is
  to 1 and the pull width to 1.
- **slope**: a line fitted through reported against planted (the recovery plot below). 1 means your
  measurement moves one-for-one with the truth; 0 that you report the same value whatever was planted.
- **pull width**: the spread of (reported − planted) / your error bar across the datasets (the pulls plot
  below). 1 means your error bars are honest; above 1.5 they are too small, below 0.5 too large, and both fail
  level 2.
- **false disc.**: datasets without a Higgs on which you claimed a discovery, out of all such datasets; any
  fails level 2.
- **self-check**: hygiene, the contract and determinism on the public dataset, the check `submit.sh` runs;
  FAIL is level 0.
- **submitted**: the commit that ran, and when and by whom it was posted.
- **With the mass**, the folded table: **level μ** by the rule above on every dataset, and **level mass** the
  same rule for the optional `mH` (0 when it is not reported on every dataset); ordered by level μ, then level
  mass, then the distance.

![reported against planted: the three example pipelines on 30 datasets](figs/recovery.png)

![the pulls of the same run: (reported − planted) / error, against a width-1 Gaussian](figs/pulls.png)

Your own plots and both tables' numbers, on datasets you make (the sealed ones are drawn from the same priors,
with a seed you don't have; `make_suite.py` plants the same mix of fixed and moving masses):

```bash
python3 make_suite.py suites/mine --n 10 --seed 7
python3 score_suite.py ~/hack-<team>/submission suites/mine --plot recovery.png
```

Each team's recovery plot on the sealed datasets themselves is folded under the tables after every run. It shows
the planted values of the sealed suite, the same suite every run: use it to see where your score is lost, not to
fit a correction to it; a correction tuned to those points is a hard-coded answer, against the rules and plain to
see in your code.
