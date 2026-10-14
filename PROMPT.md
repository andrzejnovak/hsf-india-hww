# The task

Measure the Higgs boson in the data: its mass and how much of it there is.

The dataset (`DATASET.md`) holds events with two leptons, missing transverse energy and jets. Most are the
continuum production of two W bosons. Some are a Higgs boson decaying to two W bosons and then two leptons and
two neutrinos: a signal of unknown mass m_H and unknown strength μ, the number of signal events relative to the
Standard Model expectation (μ = 1 is the Standard Model, μ = 0 is no Higgs at all). The background rate is known
to about 10 % only.

Beside the data is the simulation: the signal at 125 GeV and the background, with labels, and weights
normalised to the expected events in the data. Train on it, build your templates from it, do with it what a
real analysis does. A signal of another mass is the 125 GeV sample with the lepton momenta and the missing
energy scaled by m_H / 125 (`DATASET.md`); that is the rule the data follow.

Deliver a pipeline, `run.sh <dataset_dir> <output_dir>`, that reads one dataset and writes
`<output_dir>/RESULT.json` with:

- `mu`, the signal strength, with its uncertainty;
- optionally `mH`, the Higgs mass in GeV, with its uncertainty (a bonus when it is right);
- `significance_obs`, how incompatible the data are with no signal, in standard deviations;
- `discovery`, whether you claim the signal is there.

The format, the rules and how it is scored: `EVAL_CONTRACT.md`. The public dataset has a Higgs of mass
125 GeV at μ = 1 and the nominal background; the sealed datasets, on which the pipeline is run unchanged, have
a mass between 100 and 150 GeV, a signal strength between 0 and 3 or no signal at all, and a background
normalisation within 10 % of nominal. A pipeline that assumes the public values will be wrong on them.
