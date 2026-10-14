# Organiser notes

What is here: the H → WW toy on the participant bundle of `jfc-mit/slopbench_harness` (`sbh/`, the self-check;
`tasks/hww/`, the task stub), the shared physics (`hww.py`: kinematics, templates; a mass-morphing function that
the task no longer uses), the generator (`make_suite.py`, truth beside each dataset), the scorer
(`score_suite.py`, the contract's aggregates on a home-made suite, no sandbox), two example pipelines
(`examples/template_fit`, the reference: a fit of the m_T distribution; `examples/nn_fit`: a fit of a small
network's score), `explore.py`, `make_plots.py` (the figures of
GUIDE.md and the slides), `submit.sh`, `new_team.sh`, and the sealed run (`organiser/sealed_run.py` and
`.github/workflows/sealed_run.yml`).

## The data

`datasets/public/seed_000/mc/pool.npz` is Gordon Watts' `dataWW_d1.root` (ATLAS Open Data 13 TeV, the file
of the machine-learning tutorial; processes 345324 ggF H → WW, 345323 VBF H → WW, 363492 qq → ℓℓνν), reduced
to events with exactly two leptons and positive weight, the columns of DATASET.md as float32, and weights
scaled from the file's 10 fb⁻¹ to **1000 fb⁻¹**: 618 signal events at μ = 1 on 15 969 background. The lepton
and jet energies of the original are in MeV while its momenta are in GeV; the pool keeps momenta only.

Pseudo-data are drawn from the pool with probability proportional to the weight: at 1000 fb⁻¹ about 8 % of
the background draws repeat an event already drawn. The truths: μ uniform in [0, 3] with a fifth at 0, the
background normalisation uniform within ±10 %; the mass is 125 GeV everywhere (`hww.morph` can make a signal
of another mass by scaling lepton pT and missing ET, which an earlier version of the task planted; the
generator no longer does). `expected_significance` in truth.json is the Asimov significance of the 2D
(m_T, m_ll) templates at the truth; datasets with ≥ 5 are the scored ones.

The data events are copies of pool events (morphed for the signal); a determined cheater could match them
back to the labels. The rule in GUIDE.md and reading the submissions is the mitigation; a held-out pool
(events the students never see) would remove it, and `make_suite.py --pool` takes one.

## Validate in 3 minutes

```bash
git clone https://github.com/andrzejnovak/hsf-india-hww.git && cd hsf-india-hww
python3 -m sbh check --task hww --submission examples/stub_submission --dataset datasets/public/seed_000 --determinism   # PASS
python3 -m sbh check --task hww --submission examples/template_fit --dataset datasets/public/seed_000 --determinism     # PASS, mu ≈ 1.07 ± 0.17, 1 s
python3 -m sbh check --task hww --submission examples/nn_fit --dataset datasets/public/seed_000 --determinism           # PASS, mu ≈ 0.98 ± 0.11, 2 s
python3 make_suite.py suites/t --n 6 --seed 3 && python3 score_suite.py examples/template_fit suites/t --plot r.png    # slope ≈ 1, FDR 0
python3 explore.py suites/t/seed_001 --out e.png && python3 make_plots.py --fast --out /tmp/figs                       # the pictures regenerate
bash new_team.sh test && ls ~/hack-test                                                                                 # then rm -rf ~/hack-test
```
On 30 datasets: the m_T fit has μ slope 0.99 and pull width 0.8, the score fit slope 0.97 and width 0.7,
no false discoveries, and the score fit's error bars are a third smaller (0.11 against 0.17 on the public
dataset). `examples/nn_fit/model.json` is trained by `train.py` (scikit-learn, 10 s); retrain
it if the pool changes. `examples/*/hww.py` are copies of the root `hww.py`: keep them in step.

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
