# Leaderboard, with the mass

The [README](README.md)'s table scores μ alone, on the sealed datasets whose Higgs mass stays at 125 GeV. This
page scores μ and the mass, each by the same rule, on every sealed dataset, where the mass moves between 100 and
150 GeV and the two have to be measured together. The mass (`mH` in RESULT.json) is optional: its level is 0 when
it is not reported on every dataset, and it never costs anything in the README's table.

<!-- leaderboard -->
<!-- /leaderboard -->

### Reading the table

- **level μ**: the README's rule, on every dataset: **0** no measurement, **1** follows the truth (slope ≥ 0.8),
  **2** and honest (pull width within [0.5, 1.5], no discovery claimed on a dataset without a Higgs).
  **level mass**: the same rule for `mH`. Rows are ordered by level μ, then level mass, then how close the μ slope
  is to 1 and the μ pull width to 1.
- **slope**, **pull width**, **false disc.**, **self-check**, **submitted**: as in the README; the μ columns over
  every dataset, the mass columns over every dataset where the mass is reported.
- In the plots, filled points are the fixed-mass datasets the README scores and hollow ones the moving-mass
  datasets; the colour is the pull, as in the README. Your own: `python3 score_suite.py <submission> <suite> --plot recovery.png`.
