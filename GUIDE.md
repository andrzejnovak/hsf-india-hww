# The guide

The long version of the [README](README.md): the big picture, the levels, the loop, where the classifier goes,
the physics in pictures, debugging, what gets scored and the rules. Read, in this order: [`PROMPT.md`](PROMPT.md)
(the task), [`DATASET.md`](DATASET.md) (the data), [`EVAL_CONTRACT.md`](EVAL_CONTRACT.md) (what you deliver, how
it is run and scored).

## The big picture, for anyone who has never done a physics analysis

A collider analysis is a loop with four stages. **Collisions become rows**: each proton-proton collision that
the detector records is reduced to a handful of numbers, here two leptons, the missing energy and the jets,
twenty columns. **Rows become histograms**: millions of rows are too many to look at, so you count them in bins
of a quantity you chose, the transverse mass, say, or a classifier's output. **Simulation gives the shapes**:
you cannot label a real collision, but you can simulate the processes you expect, the Higgs and the
backgrounds, and histogram them the same way, so you know what shape each one makes. **A fit reads off the
physics**: the data histogram is modelled as a sum of the simulated shapes with a few free numbers, how much
Higgs, how much background, and the numbers that make the observed counts most probable, with their error
bars, are the measurement.

Machine learning enters at the second stage: a classifier trained on the simulation turns twenty columns into
one number, "how Higgs-like", and histogramming that number separates signal from background far better than
any one column. Agents enter at every stage: the whole loop is code, and the question of this week is whether
an agent can write that code, and how we would know it got it right.

That last question is the hackathon. The organisers hold datasets with the answers hidden; your pipeline runs on
them unchanged; the score is whether your μ follows the planted one, and whether your error bars are honest.
Nothing in this repository is beyond a CS undergraduate: it is histograms, a maximum-likelihood fit of a
two-parameter model, and a small neural network.

## Three levels

1. **A pipeline.** A `run.sh` that measures μ from the data, passes the self-check and follows the truth on a
   suite you made (slope ≈ 1). `examples/template_fit` is one: a fit of the transverse-mass distribution, two
   parameters, what the real analyses do. The agent can write such a pipeline from `PROMPT.md`.
2. **The classifier, plugged in, and still honest.** Train on the simulation, export the model as text, score
   the data in `run.sh`, fit the score distribution. `examples/nn_fit` is the plain version: on the public
   dataset it takes the error on μ from 0.17 to 0.11 and the significance from 6.3σ to 9.2σ. The level is
   reached when the error bars are honest on the sealed datasets (pulls of width ≈ 1) and nothing is claimed on
   the empty ones. Better features, a better network, a two-dimensional score × m<sub>T</sub> fit, a treatment of
   the template statistics are all open.
3. **The agent does it.** A brief, a check it must pass, a reviewer, a second model: whatever gets
   `deepseek-v4-flash` or `glm-5` from the bare prompt to a level-2 pipeline on its own. How much context that
   takes is an open question; keep the prompts and the runs, they are the result.

The metric: **sensitivity** (the slope of reported vs planted μ; 1 is a measurement, 0 is a constant) while keeping
**calibration** (the width of your pulls with your own error bars; 1 is honest, bigger means error bars too small,
smaller means padded), and **no false discoveries** on the datasets without a Higgs.

## The loop

1. **Build** `~/hack-<team>/submission/`: a `run.sh <dataset_dir> <output_dir>` that writes `<output_dir>/RESULT.json`
   (format in EVAL_CONTRACT.md §2). Python 3 with **numpy and scipy only** at run time. Start from
   `examples/template_fit/` (copy `fit.py`, `hww.py`, `run.sh`) or `examples/nn_fit/` (plus `train.py` and
   `model.json`).
2. **With the agent:** `cd ~/hack-<team> && opencode`, then paste:
   > Read PROMPT.md, DATASET.md and EVAL_CONTRACT.md in this folder. Build the submission in `submission/`:
   > a `run.sh <dataset_dir> <output_dir>` that writes `<output_dir>/RESULT.json`. The public dataset is
   > `datasets/public/seed_000`. Python 3 with numpy and scipy only at run time. Do not hard-code the public
   > dataset's answer.
3. **Self-check** (hygiene, the contract, determinism), from the kit folder:
   ```bash
   python3 -m sbh check --task hww --submission ~/hack-<team>/submission --dataset datasets/public/seed_000 --determinism
   ```
4. **Score yourself** on suites you make, truth included, before the sealed one:
   ```bash
   python3 make_suite.py suites/mine --n 10 --seed 7        # 1 public + 9 hidden-truth datasets
   python3 score_suite.py ~/hack-<team>/submission suites/mine --plot recovery.png
   ```
   Change `--seed` for a fresh suite. The sealed suite is drawn from the same priors with a seed you don't have.
5. **Submit**: your workspace is a git repository; push it to GitHub and post one line on the
   [submissions issue](https://github.com/andrzejnovak/hsf-india-hww/issues/1):
   ```bash
   cd ~/hack-<team> && git add -A && git commit -m "our submission" && git push    # a public repository of yours (git remote add origin <url> the first time)
   cd ~/hsf-india-hww && bash submit.sh <team-name> <git url>                      # the self-check, is it pushed, then the line to post
   ```
   The line is `<team-name>: <git url> @ <commit>`; `submit.sh` posts it for you when `gh` is logged in, else
   paste it as a comment. Submit as often as you like: a new comment replaces your earlier one, and the last
   comment per team before the deadline is the commit that gets run. A submission that fails the self-check
   is not submitted (`--no-check` if the check cannot run on your machine; the organisers run it anyway).
   Pushing from a Binder session needs a GitHub token as the password (github.com → Settings → Developer
   settings → Personal access tokens); from your laptop it is your usual push.

## Where the classifier goes

The pool in every dataset (`mc/pool.npz`) is the training set: 577k simulated events with labels and weights, the
same events as the machine-learning tutorial's file. Train whatever you like on it, on your own machine, with
whatever library; then export the model as text (JSON weights; `examples/nn_fit/train.py` does it for a
multilayer perceptron) and evaluate it in `run.sh` with numpy. In the measurement the score is the fitted
observable: histogram it for the data and for the simulated signal and background, and fit the two rates
(`examples/nn_fit/fit.py`). The score cut, the binning, a second axis such as m<sub>T</sub>, the features, the
network itself: all yours.

## The physics, in five pictures

`python3 make_plots.py` draws all of them from the kit (`figs/`; `--fast` skips the one that fits 60 datasets).

1. **[`figs/data.png`](figs/data.png) — the data.** Each event is two leptons, missing transverse energy and jets
   from a proton-proton collision: the final state of two W bosons each decaying to a lepton and a neutrino.
   Most come from the continuum production of two W bosons; some from a Higgs boson decaying to two W bosons.
   The two neutrinos escape, so there is no mass peak: the Higgs shows as a soft bump in the transverse mass
   m<sub>T</sub> and a low dilepton mass m<sub>ll</sub> (the two leptons come out close together, a consequence of
   the Higgs having no spin). The pool holds the simulation of both, with weights normalised to the expected
   events at 1000 fb⁻¹: 618 signal events at μ = 1 on 15 969 background.
2. **[`figs/score.png`](figs/score.png) — the classifier.** A small network trained on the simulation gives each
   event a score between 0 and 1. Signal piles up at high score, background at low; the same score on the data
   is what the network pipeline fits.
3. **[`figs/fit.png`](figs/fit.png) — a fit.** A measurement is a model with a few knobs turned until it matches
   the data: here the signal strength μ and the background rate b (maximum likelihood: the knobs that make the
   observed counts most probable, bin by bin). Move μ away from the best value and the match gets worse; the width
   of that valley, where it is less than ½ unit worse, is the error bar. The valley is narrower when the
   fitted histogram is the network score than when it is m<sub>T</sub>: that is what the classifier buys. An
   honest error bar covers the truth about two times in three; that is what the sealed run tests.
4. **[`figs/recovery.png`](figs/recovery.png) — the truth moves.** Every sealed dataset has its own planted μ
   (one dataset in five has none) and background rate. Both example pipelines follow the diagonal on 30 of them;
   answer 1 every time and your slope is 0.
5. **[`figs/pulls.png`](figs/pulls.png) — calibration.** pull = (reported − planted) / your error bar; a
   calibrated pipeline has pulls of width 1. Both examples are.

The real analysis this is a toy of: the HSF-India project
[H → WW with CMS Open Data NanoAOD](https://github.com/davidlange6/H-to-WW-NanoAOD-analysis), which fits the
same transverse mass with nine backgrounds, jet-binned signal regions, control regions and a statistical model
in Combine. Here: one background, one region, a two-parameter fit, and the simulation from the
[machine-learning tutorial](https://github.com/gordonwatts/courses-hsf-india-October2026).

## Debugging

```bash
bash ~/hack-<team>/submission/run.sh suites/mine/seed_003 out && cat out/RESULT.json    # one dataset by hand
python3 explore.py suites/mine/seed_003 --result out/RESULT.json --out seed_003.png     # the data, the truth, your answer
```
`score_suite.py` prints a per-dataset table (status, pulls, and the last lines of `run.sh`'s output when it fails);
the self-check prints the traceback of a failed run. `seed_000` of any suite is the public truth, so a pipeline that
is right there and wrong elsewhere has hard-coded something.

## What gets scored (EVAL_CONTRACT.md §3, in one line each)

- **sensitivity**: slope of reported vs planted μ — 1 is a measurement, 0 is a memorised constant (the stub scores 0).
- **calibration**: width of your pulls `(reported − planted) / your total` — too small *and* too large both fail.
- **bias**, **recovery** (fraction of datasets that run).
- **false-discovery rate** on datasets with no signal; **discovery power** on strong ones.

`score_suite.py` prints the same aggregates on your own suites. The organisers' leaderboard is the sealed run.

## Rules

- numpy + scipy at run time; no other packages, no network, ≤ 600 s per dataset, no data files inside
  `submission/` (a model as text is fine).
- `run.sh` must be deterministic (fix your random seeds) and read only `<dataset_dir>` and its own folder.
- Nothing from the public truth (μ = 1, nominal background) may be hard-coded: the sealed datasets differ in
  both.
- The data are drawn from the simulation you train on. Matching data events back to pool events to read off
  their labels is cheating, not analysis; the organisers read every submission.
- Teams of three or so. Model variance is real: run the agent twice before you trust a design.
- The leaderboard: level 0, 1 or 2 as above (level 1: slope ≥ 0.8 with the self-check passing and every dataset
  run; level 2: pull width within [0.7, 1.4] and no false discovery), then closest to slope 1 and width 1 first.
  Level 3 is judged when you present: show the prompts and the runs.

## The day

1. Setup: the README's commands pass; teams and workspaces.
2. Build, self-check, score yourself; submit early and again whenever it gets better: every comment on the
   issue re-runs the sealed run (a GitHub Actions job, every submission on 20 fresh datasets nobody has seen)
   and updates the table in the README within a few minutes.
3. The deadline is announced in the room; the last comment per team on the issue before it counts.
4. The final sealed run: the table with recovery plots, one slide per team, then each team on how it got there
   (level 3 is judged there).
