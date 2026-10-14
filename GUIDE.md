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
Higgs, how heavy it is, how much background, and the numbers that make the observed counts most probable, with
their error bars, are the measurement.

Machine learning enters at the second stage: a classifier trained on the simulation turns twenty columns into
one number, "how Higgs-like", and histogramming that number separates signal from background far better than
any one column. Agents enter at every stage: the whole loop is code, and the question of this week is whether
an agent can write that code, and how we would know it got it right.

That last question is the hackathon. The organisers hold datasets with the answers hidden; your pipeline runs on
them unchanged; the score is whether your μ follows the planted one, and whether your error bars are honest.
Nothing in this repository is beyond a CS undergraduate: it is histograms, a maximum-likelihood fit of a model
with a few parameters, and a small neural network.

## Three levels

1. **A pipeline.** A `run.sh` that measures μ from the data, passes the self-check and follows the truth on a
   suite you made (slope ≈ 1). `examples/template_fit` is one: a fit of the (m<sub>T</sub>, m<sub>ll</sub>)
   histogram, the leaderboard's `baseline-mt` row. The agent can write such a pipeline from `PROMPT.md`; make
   your own suites (step 4 of the loop) and look at what they plant before you trust it.
2. **The classifier, plugged in, and still honest.** Train on the simulation, export the model as text, score
   the data in `run.sh`, fit the score together with a quantity that carries the mass. `examples/tutorial_nn`
   is the tutorial's own network plugged in unchanged (six raw columns, three hidden layers), fitted in bins of
   score × m<sub>T</sub>: that is the leaderboard's `baseline-nn` row. On the public dataset it is *not* better
   than the template fit (μ ± 0.16 against ± 0.14, the mass ± 7.3 GeV against ± 5.3): a score trained to tell
   signal from background at 125 GeV compresses away the shape information the mass lives in. `examples/nn_fit`
   is one step past it, ten physics features instead of six raw ones (± 0.14, ± 5.9 GeV), still not past the
   template fit: the room is real. The level is
   reached when the error bars on μ are honest on the sealed datasets (pulls of width ≈ 1) and nothing is
   claimed on the empty ones; the optional `mH` of the contract, reported on every dataset, gets its own level
   by the same rule, scored independently of μ. Better features, a classifier that keeps the mass information, a
   three-dimensional score × m<sub>T</sub> × m<sub>ll</sub> fit, a treatment of the template statistics are
   all open.
3. **The agent does it.** A brief, a check it must pass, a reviewer, a second model: whatever gets
   `deepseek-v4-flash` or `glm-5` from the bare prompt to a level-2 pipeline on its own. How much context that
   takes is an open question; keep the prompts and the runs, they are the result.

The metric, for μ (and for the mass when you report it): **sensitivity** (the slope of reported vs planted; 1 is
a measurement, 0 is a constant) while keeping **calibration** (the width of your pulls with your own error bars;
1 is honest, bigger means error bars too small, smaller means padded), and **no false discoveries** on the
datasets without a Higgs.

## The loop

1. **Build** `~/hack-<team>/submission/`: a `run.sh <dataset_dir> <output_dir>` that writes `<output_dir>/RESULT.json`
   (format in EVAL_CONTRACT.md §2). Python 3 with **numpy and scipy only** at run time. Start from
   `examples/template_fit/` (copy `fit.py`, `hww.py`, `run.sh`) or `examples/tutorial_nn/` (plus `train.py`
   and `model.json`; `examples/nn_fit/` is the same with better features).
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

## What the network should be optimising

The measurement is a fit of a histogram that has the score as one axis, so the quantity to make small is **the
error on μ and on the mass that the histogram gives**, not the AUC. Three facts set the direction. First, for
μ alone the optimal one-number summary of an event is the likelihood ratio p(x | signal) / p(x | background),
and a binary classifier trained with cross-entropy on the simulation, with the event weights as sample
weights, approximates exactly that: so the spec is a probabilistic signal-vs-background classifier, nothing
exotic. Second, the mass is *not* in that number: a score trained at 125 GeV is a cut, not a ruler, and the mass
lives in the shapes of m<sub>T</sub> and m<sub>ll</sub>; that is why the examples fit score × m<sub>T</sub> and
not the score alone, and why the tutorial's network plugged in naively (`examples/tutorial_nn`, μ ± 0.16, the
mass ± 7.3 GeV on the public dataset) does worse than the plain (m<sub>T</sub>, m<sub>ll</sub>) template fit
(± 0.14, ± 5.3 GeV); ten physics features (`examples/nn_fit`) get to ± 0.14 and ± 5.9 GeV. Third, were the mass
known, the ceiling is close: with it fixed at 125 a fit of m<sub>T</sub> alone gives σ<sub>μ</sub> ≈ 0.17, the
2D template fit 0.12, the tutorial's network 0.13, ten features 0.11 and the best classifier we found (gradient
boosting on 17 features) 0.108; a bigger network buys nothing by itself. What moves the numbers:

- **the features**: the derived kinematics (m<sub>ll</sub>, m<sub>T</sub>, p<sub>T</sub><sup>ll</sup>,
  Δφ<sub>ll</sub>) do most of the work; the lepton flavours (same-flavour pairs carry the ZZ part of the
  background), the jets and the charges add a few per cent;
- **the weights**: train with the event weights (the background's are far from uniform) or the score is tuned
  to the wrong mixture;
- **the binning** of the score histogram: finer bins where the signal sits, enough simulated events per bin that
  the templates are not noise;
- **the axes beside the score**: m<sub>T</sub> (the examples) keeps what the score compresses away; m<sub>ll</sub>
  as a third axis, or a score trained to be *independent* of the mass, or a classifier parametrised in the
  mass hypothesis (train on the morphed signal at several masses, the mass as an input), or a second network
  that regresses the mass: all ways to give the fit back the ruler;
- **honesty**: pulls of width 1 on your own suites come before another per cent of precision. A classifier that is
  overtrained, or templates with too few simulated events, show up as error bars that are too small.

Measure your own numbers: `score_suite.py` on a suite of 30 gives the pull widths, and the mean quoted errors are
σ<sub>μ</sub> and σ<sub>m</sub>. For the ambitious: a classifier trained for the measurement directly (a loss that is the
expected σ<sub>μ</sub> of the resulting histogram, differentiable through a soft histogram) is a research
topic, not a hackathon afternoon, but the pieces are all here.

## Where the classifier goes

The pool in every dataset (`mc/pool.npz`) is the training set: 577k simulated events with labels and weights, the
same events as the machine-learning tutorial's file. Train whatever you like on it, on your own machine, with
whatever library; then export the model as text (JSON weights; `examples/tutorial_nn/train.py` and
`examples/nn_fit/train.py` do it for a multilayer perceptron) and evaluate it in `run.sh` with numpy. In the measurement the score is one axis of the
fitted histogram: histogram (score, m<sub>T</sub>) for the data, for the simulated background and for the
simulated signal at every mass of a grid (the 125 GeV sample morphed, then scored by the same network, so the
classifier's response to a heavier or lighter Higgs is part of the model), and fit the mass, μ and the
background rate (`examples/nn_fit/fit.py`). The binning, the axes, the features, the network itself: all
yours.

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
   the data: the signal strength μ, the background rate b, whatever else the shapes depend on (maximum
   likelihood: the knobs that make the observed counts most probable, bin by bin). Move one knob away from its
   best value, re-tune the others, and the match gets worse; the width of that valley, where it is less than ½
   unit worse, is the error bar. Left, the valley in μ for the three example pipelines; right, the valley in
   the other parameter the examples fit. An honest error bar covers the truth about two times in three; that
   is what the sealed run tests.
4. **[`figs/recovery.png`](figs/recovery.png) — the truth moves.** Every sealed dataset has its own planted μ
   (one dataset in five has none) and background rate. The example pipelines follow the diagonal on 30 of
   them; answer the public values every time and your slope is 0.
5. **[`figs/pulls.png`](figs/pulls.png) — calibration.** pull = (reported − planted) / your error bar; a
   calibrated pipeline has pulls of width 1.

The real analysis this is a toy of: the HSF-India project
[H → WW with CMS Open Data NanoAOD](https://github.com/davidlange6/H-to-WW-NanoAOD-analysis), which fits the
same transverse mass with nine backgrounds, jet-binned signal regions, control regions and a statistical model
in Combine. Here: one background, one region, a three-parameter fit, and the simulation from the
[machine-learning tutorial](https://github.com/gordonwatts/courses-hsf-india-October2026).

## Debugging

```bash
bash ~/hack-<team>/submission/run.sh suites/mine/seed_003 out && cat out/RESULT.json    # one dataset by hand
python3 explore.py suites/mine/seed_003 --result out/RESULT.json --out seed_003.png     # the data, the truth, your answer
```
`score_suite.py` prints a per-dataset table (status, pulls, and the last lines of `run.sh`'s output when it fails);
the self-check prints the traceback of a failed run. `seed_000` of any suite is the public truth, so a pipeline that
is right there and wrong elsewhere has hard-coded something.

## What gets scored: a level

The leaderboard is levels, from two tests on the sealed datasets (EVAL_CONTRACT.md §3 has the full list of
aggregates; they are all in the sealed run's `results/`):

- **level 0, no measurement**: the self-check fails, a dataset does not run, or the slope of reported vs
  planted μ is below 0.8 (the stub's slope is 0).
- **level 1, follows the truth**: slope ≥ 0.8.
- **level 2, and honest**: the pulls `(reported − planted) / your total` have a width within [0.5, 1.5] (too
  small *and* too large both fail), and no discovery is claimed on a dataset without a Higgs.
- **level 3** is judged when you present: how the agent got there.

The contract's optional `mH`, reported on every scored dataset, gets its own level by the same rule, scored
independently of μ (0 when it is not reported); the leaderboard orders by the μ level first. `score_suite.py`
prints both levels on your own suites.

## Rules

- numpy + scipy at run time; no other packages, no network, ≤ 600 s per dataset, no data files inside
  `submission/` (a model as text is fine).
- `run.sh` must be deterministic (fix your random seeds) and read only `<dataset_dir>` and its own folder.
- Nothing from the public truth (125 GeV, μ = 1, nominal background) may be hard-coded: the sealed datasets
  differ in all three.
- The data are drawn from the simulation you train on. Matching data events back to pool events to read off
  their labels is cheating, not analysis; the organisers read every submission.
- Teams of three or so. Model variance is real: run the agent twice before you trust a design.
- The leaderboard: the μ level, then the mass level, then closest to slope 1 and width 1 first. Level 3 is
  judged when you present: show the prompts and the runs.

## The day

1. Setup: the README's commands pass; teams and workspaces.
2. Build, self-check, score yourself; submit early and again whenever it gets better: every comment on the
   issue re-runs the sealed run (a GitHub Actions job, every submission on 20 fresh datasets nobody has seen)
   and updates the table in the README within a few minutes.
3. The deadline is announced in the room; the last comment per team on the issue before it counts.
4. The final sealed run: the table with recovery plots, one slide per team, then each team on how it got there
   (level 3 is judged there).
