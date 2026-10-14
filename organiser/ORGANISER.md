# Organiser notes

What is here: the H → WW toy on the participant bundle of `jfc-mit/slopbench_harness` (`sbh/`, the self-check;
`tasks/hww/`, the task stub), the shared physics (`hww.py`: kinematics, the mass morphing, templates), the
generator (`make_suite.py`, truth beside each dataset), the scorer
(`score_suite.py`, the contract's aggregates on a home-made suite, no sandbox), three example pipelines
(`examples/template_fit`, the reference: a three-parameter fit of the (m_T, m_ll) histogram, the posted
`baseline-mt`; `examples/tutorial_nn`: the ML tutorial's network on its six raw columns in a score × m_T fit,
the posted `baseline-nn`; `examples/nn_fit`: the same network on ten physics features, the worked step past
the baseline), `explore.py`, `make_plots.py` (the figures of
GUIDE.md and the slides), `submit.sh`, `new_team.sh`, and the sealed run (`organiser/sealed_run.py` and
`.github/workflows/sealed_run.yml`).

## The data

`datasets/public/seed_000/mc/pool.npz` is Gordon Watts' `dataWW_d1.root` (ATLAS Open Data 13 TeV, the file
of the machine-learning tutorial; processes 345324 ggF H → WW, 345323 VBF H → WW, 363492 qq → ℓℓνν), reduced
to events with exactly two leptons and positive weight, the columns of DATASET.md as float32, and weights
scaled from the file's 10 fb⁻¹ to **1000 fb⁻¹**: 618 signal events at μ = 1 on 15 969 background. The lepton
and jet energies of the original are in MeV while its momenta are in GeV; the pool keeps momenta only.

Pseudo-data are drawn from the pool with probability proportional to the weight: at 1000 fb⁻¹ about 8 % of
the background draws repeat an event already drawn. The truths: the mass uniform in [100, 150] GeV, μ uniform
in [0, 3] with a fifth at 0, the background normalisation uniform within ±10 %. A signal of mass m is the
125 GeV sample with lepton pT and missing ET scaled by m / 125 (`hww.morph`), the rule DATASET.md states, so
templates built the same way are exact. `expected_significance` in truth.json is the Asimov significance of
the 2D (m_T, m_ll) templates at the truth; datasets with ≥ 5 are the scored ones. The mass is in the task
because without it the task is too easy: an agent reached level 2 from the bare prompt in one session; with
it, the pipeline that fixes the mass at the public 125 GeV has a μ slope of 1.3 and pulls of width 2.6, so it
stays at level 1. Reporting the mass is optional (`score_suite.OPTIONAL_POIS`): scored by the same rule when
it is on every scored dataset, as a level of its own, independent of the μ level; the leaderboard orders by
the μ level first.

The data events are copies of pool events (morphed for the signal); a determined cheater could match them
back to the labels. The rule in GUIDE.md and reading the submissions is the mitigation; a held-out pool
(events the students never see) would remove it, and `make_suite.py --pool` takes one.

## Validate in 3 minutes

```bash
git clone https://github.com/andrzejnovak/hsf-india-hww.git && cd hsf-india-hww
python3 -m sbh check --task hww --submission examples/stub_submission --dataset datasets/public/seed_000 --determinism   # PASS
python3 -m sbh check --task hww --submission examples/template_fit --dataset datasets/public/seed_000 --determinism     # PASS, mH ≈ 123 ± 5, mu ≈ 0.88 ± 0.14, 6 s
python3 -m sbh check --task hww --submission examples/tutorial_nn --dataset datasets/public/seed_000 --determinism     # PASS, mH ≈ 129 ± 7, mu ≈ 1.05 ± 0.16, 30 s
python3 -m sbh check --task hww --submission examples/nn_fit --dataset datasets/public/seed_000 --determinism           # PASS, mH ≈ 128 ± 6, mu ≈ 1.01 ± 0.14, 15 s
python3 make_suite.py suites/t --n 6 --seed 3 && python3 score_suite.py examples/template_fit suites/t --plot r.png    # slope ≈ 1, FDR 0
python3 explore.py suites/t/seed_001 --out e.png && python3 make_plots.py --fast --out /tmp/figs                       # the pictures regenerate (the full run fits 93 datasets, 15 min)
bash new_team.sh test && ls ~/hack-test                                                                                 # then rm -rf ~/hack-test
```
On 31 datasets: the template fit has μ slope 0.99 and pull width 0.62 (the mass: 1.01 and 1.00), the
tutorial network's score fit 1.05 and 0.99 (the mass: 0.97 and 1.01), no false discoveries; the error bars on
the public dataset are 0.14 and 0.16 on μ, 5 and 7 GeV on the mass. The leaderboard posts the first two as `baseline-mt` and `baseline-nn`
(branches `hack-baseline-mt`, `hack-baseline-nn`): the ten-feature network stays in the kit as the worked
step past the baseline, so the students have something to beat. `examples/*/model.json` are trained by each
example's `train.py` (scikit-learn, under a minute); retrain them if the pool changes.

The calibration window is [0.5, 1.5] rather than the [0.7, 1.4] of the toy kit because honest pipelines
over-cover on μ here: on 31 datasets the (m_T, m_ll) template fit has a pull width of 0.62, and 0.74 even with
the mass fixed at its planted value, with the scatter of μ̂ (0.095) below its Poisson error (0.125); the score
fits sit at 0.8–1.0. The pseudo-data are a weighted draw with replacement from a finite pool, so a bin that a
few heavy background events dominate fluctuates less than Poisson; the fit's likelihood does not know that.
The window still fails the stub (width 4) and a padded "always ± 1" (width 0.1). `examples/*/hww.py` are copies of the root `hww.py`: keep them in step.

## Before the start

```bash
python3 -c 'import secrets; print(secrets.randbelow(2**31))'   # a seed, only if you want two sealed runs comparable; the CI makes its own
```
Students submit by pushing their team workspace to GitHub and posting `<team>: <url> @ <commit>` as a comment
on the [submissions issue](https://github.com/andrzejnovak/hsf-india-hww/issues/1); `bash submit.sh <team> <url>`
runs the self-check, checks the commit is pushed and prints (or, with `gh`, posts) that line. The last comment
per team before the deadline counts.

## The sealed run in CI (the default)

`.github/workflows/sealed_run.yml` runs `organiser/sealed_run.py` on a GitHub-hosted runner on every comment on
the submissions issue and commits the table into `README.md` (the `<!-- leaderboard -->` markers). The suite is
drawn from a seed made up in the job and never printed; the submissions run as `nobody` with the truths
unreadable (`--as nobody`) and a scrubbed environment. After the deadline: Actions → "sealed run" → Run
workflow with plots = true (and a larger n if you like) for the final table with the recovery plots, committed
under `results/`; the plots show the planted values per dataset, which is why they wait for the deadline. A
fixed seed input makes two runs comparable. About a minute per team for either example (21 datasets plus the self-check).

## The sealed run by hand

```bash
python3 organiser/sealed_run.py --seed <secret>              # reads issue 1, clones each team at its commit, ~/hack-sealed: 21 datasets, truths kept apart; --as nobody where sudo allows it
cat ~/hack-sealed/results/leaderboard.md                     # per team: results/<team>/recovery.png, score.txt, check.txt
```
`--team red` for one team, `--teams FILE` for a hand-made list of `<team>: <url> @ <commit>` lines, `--timeout`
is the contract's 600 s per dataset. The levels and the order are in the script's docstring; `level()` is the
one function to change if the rule changes. The submissions run as you with no sandbox unless `--as` is given:
a throwaway session or a VM, not your own account.

The private harness is the sandboxed alternative, on the clones `sealed_run.py` made: a `hww` task module
there would need the generator above; it does not exist yet, the kit is self-contained.

## The real analysis this is a toy of

David Lange's HSF-India project, [H → WW with CMS Open Data NanoAOD](https://github.com/davidlange6/H-to-WW-NanoAOD-analysis):
the eμ channel, nine backgrounds, jet-binned signal regions and control regions, Combine. Its cutflow at
36 fb⁻¹ has 285 ggH events in the signal regions on about 15 000 background, the same scale as this toy at
1000 fb⁻¹ with one background. The CMS NanoAOD samples are reachable from a GitHub runner
(`https://opendata.cern.ch/eos/opendata/cms/...`), so a CMS pool with a held-out split is a possible
follow-up; the ATLAS Open Data originals were not reachable from the session that built this.
