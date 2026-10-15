# Leaderboard, with the mass

The [README](README.md)'s table scores μ alone, on the sealed datasets whose Higgs mass stays at 125 GeV. This
page scores μ and the mass, each by the same rule, on every sealed dataset, where the mass moves between 100 and
150 GeV and the two have to be measured together. The mass (`mH` in RESULT.json) is optional: it counts here when
it is reported on every dataset, and it never costs anything in the README's table.

<!-- leaderboard -->
<!-- /leaderboard -->

### Reading the table

- **score**, the order: as in the README, 100 × the template fit's calibrated error over yours, here averaged
  over μ and the mass where you report it (μ alone where you don't), both on every dataset. The template fit
  scores 100.
- **error μ**, **error mass**: your calibrated error bars, the mean error bar you quote times the pull width
  where that is above 1, on every dataset.
- **slope**, **pull width**, **submitted**: as in the README; the μ columns over every dataset, the mass columns
  over every dataset where the mass is reported.
- In the plots, filled points are the fixed-mass datasets the README scores and hollow ones the moving-mass
  datasets; the colour is the pull, as in the README. Your own: `python3 score_suite.py <submission> <suite> --plot recovery.png`.
