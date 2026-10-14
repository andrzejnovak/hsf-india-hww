# The guide

The long version of the [README](README.md): the levels, the loop, where the classifier goes, the physics in
pictures, debugging, what gets scored and the rules. Read, in this order: [`PROMPT.md`](PROMPT.md) (the task),
[`DATASET.md`](DATASET.md) (the data), [`EVAL_CONTRACT.md`](EVAL_CONTRACT.md) (what you deliver, how it is run
and scored).

## Three levels

1. **A pipeline.** A `run.sh` that measures the Higgs mass and the signal strength from the data, passes the
   self-check and follows the truth on a suite you made (slope ≈ 1 for both). `examples/template_fit` is one: a
   fit of templates on the (m<sub>T</sub>, m<sub>ll</sub>) plane, the signal template at any mass by morphing the
   simulated 125 GeV sample. The agent can write such a pipeline from `PROMPT.md`.
2. **Sharper, still honest.** Plug a classifier in and beat the reference's error bars while keeping pulls of
   width ≈ 1 and no false discoveries on the empty datasets. `examples/nn_fit` is the plain version: a small
   network trained on the pool, exported as text, evaluated with numpy, and a fit in bins of score and
   m<sub>T</sub>. On the public dataset it takes the mass error from 6.0 to 5.6 GeV and the strength error from
   0.137 to 0.133: a start, not the end. Better features, a mass-parametrised classifier trained on the
   morphed signal, a better binning, a systematic treatment of the template statistics are all open.
3. **The agent does it.** A brief, a check it must pass, a reviewer, a second model: whatever gets
   `deepseek-v4-flash` or `glm-5` from the bare prompt to a level-2 pipeline on its own. How much context that
   takes is an open question; keep the prompts and the runs, they are the result.

The metric: **sensitivity** (the slope of reported vs planted; 1 is a measurement, 0 is a constant) while keeping
**calibration** (the width of your pulls with your own error bars; 1 is honest, bigger means error bars too small,
smaller means padded), then **precision** against the reference, awarded only while calibration holds.

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
multilayer perceptron) and evaluate it in `run.sh` with numpy. In the measurement the score is an axis of the
templates (`examples/nn_fit/fit.py`: score × m<sub>T</sub>, the score for purity, m<sub>T</sub> for the mass) or
a selection before the kinematic fit. The signal template at mass m is the morphed sample scored by the same
network, so the classifier's response to a heavier or lighter Higgs is in the model; a classifier trained at
125 GeV is not wrong at 140, only weaker, and one trained on morphed samples with the mass as an input is the
natural next step.

## The physics, in five pictures

`python3 make_plots.py` draws all of them from the kit (`figs/`; `--fast` skips the one that fits 30 datasets).

1. **[`figs/data.png`](figs/data.png) — the data.** Each event is two leptons, missing transverse energy and jets
   from a proton-proton collision: the final state of two W bosons each decaying to a lepton and a neutrino.
   Most come from the continuum production of two W bosons; some from a Higgs boson decaying to two W bosons.
   The two neutrinos escape, so there is no mass peak: the Higgs shows as a soft bump in the transverse mass
   m<sub>T</sub> and a low dilepton mass m<sub>ll</sub> (the two leptons come out close together, a consequence of
   the Higgs having no spin). The pool holds the simulation of both, with weights normalised to the expected
   events at 1000 fb⁻¹: 618 signal events at μ = 1 on 15 969 background.
2. **[`figs/morph.png`](figs/morph.png) — a Higgs of another mass.** The simulation exists at 125 GeV only. A
   heavier Higgs gives its decay products more momentum, so a signal of mass m is defined as the 125 GeV sample
   with lepton momenta and missing energy scaled by m / 125 (`hww.py`, `morph`). The planted datasets were made
   with this rule, so templates built with it are exact. It is a kinematic approximation of real physics (it
   ignores how the off-shell W changes with the mass) and the truth lives inside it.
3. **[`figs/fit.png`](figs/fit.png) — a fit.** A measurement is a model with a few knobs turned until it matches
   the data: here the mass m<sub>H</sub>, the signal strength μ and the background normalisation b (maximum
   likelihood: the knobs that make the observed counts most probable, bin by bin). Move m<sub>H</sub> away from
   the best value and the match gets worse; the width of that valley, where it is less than ½ unit worse, is the
   error bar. An honest error bar covers the truth about two times in three; that is what the sealed run tests.
4. **[`figs/recovery.png`](figs/recovery.png) — the truth moves.** Every sealed dataset has its own planted mass
   between 100 and 150 GeV, its own μ (one dataset in five has none) and background normalisation. The reference
   fit on 30 of them follows the diagonal; answer 125 every time and your slope is 0.
5. **[`figs/pulls.png`](figs/pulls.png) — calibration.** pull = (reported − planted) / your error bar; a
   calibrated pipeline has pulls of width 1. The reference's are; a Hessian error on a flat likelihood was not,
   which is why it uses the profile likelihood for the mass.

The real analysis this is a toy of: the HSF-India project
[H → WW with CMS Open Data NanoAOD](https://github.com/davidlange6/H-to-WW-NanoAOD-analysis), nine backgrounds,
jet-binned signal regions, control regions, a statistical model in Combine. Here: one background, one region,
a three-parameter fit, and the simulation from the [machine-learning tutorial](https://github.com/gordonwatts/courses-hsf-india-October2026).

## Debugging

```bash
bash ~/hack-<team>/submission/run.sh suites/mine/seed_003 out && cat out/RESULT.json    # one dataset by hand
python3 explore.py suites/mine/seed_003 --result out/RESULT.json --out seed_003.png     # the data, the truth, your answer
```
`score_suite.py` prints a per-dataset table (status, pulls, and the last lines of `run.sh`'s output when it fails);
the self-check prints the traceback of a failed run. `seed_000` of any suite is the public truth, so a pipeline that
is right there and wrong elsewhere has hard-coded something.

## What gets scored (EVAL_CONTRACT.md §3, in one line each)

- **sensitivity**: slope of reported vs planted — 1 is a measurement, 0 is a memorised constant (the stub scores 0).
- **calibration**: width of your pulls `(reported − planted) / your total` — too small *and* too large both fail.
- **bias**, **precision** (vs the reference, only once calibration passes), **recovery** (fraction of datasets that run).
- **false-discovery rate** on datasets with no signal; **discovery power** on strong ones.

`score_suite.py` prints the same aggregates on your own suites. The organisers' leaderboard is the sealed run.

## Rules

- numpy + scipy at run time; no other packages, no network, ≤ 600 s per dataset, no data files inside
  `submission/` (a model as text is fine).
- `run.sh` must be deterministic (fix your random seeds) and read only `<dataset_dir>` and its own folder.
- Nothing from the public truth (mass 125, μ = 1, nominal background) may be hard-coded: the sealed datasets
  differ in all of it.
- The data are drawn from the simulation you train on. Matching data events back to pool events to read off
  their labels is cheating, not analysis; the organisers read every submission.
- Teams of three or so. Model variance is real: run the agent twice before you trust a design.
- The leaderboard: level 0, 1 or 2 as above (level 1: slope of m<sub>H</sub> ≥ 0.8 with the self-check passing and every
  dataset run; level 2: pull widths within [0.7, 1.4] for both and no false discovery), then closest to slope 1 and
  width 1 first. Level 3 is judged when you present: show the prompts and the runs.

## The day

1. Setup: the README's commands pass; teams and workspaces.
2. Build, self-check, score yourself; submit early and again whenever it gets better: every comment on the
   issue re-runs the sealed run (a GitHub Actions job, every submission on 20 fresh datasets nobody has seen)
   and updates the table in the README within a few minutes.
3. The deadline is announced in the room; the last comment per team on the issue before it counts.
4. The final sealed run: the table with recovery plots, one slide per team, then each team on how it got there
   (level 3 is judged there).
