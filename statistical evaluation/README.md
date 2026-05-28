This script reproduces all paired permutation tests reported in the paper.
It reads per-sample quality scores from CSV files, computes paired permutation
tests with Bonferroni correction, and outputs the consolidated results table.

Statistical method:
  - Two-sided paired permutation test (sign-flip variant)
  - 99,999 resamples per test
  - Bonferroni correction for family-wise error rate control
  - Minimum attainable p-value: 2 / 100,000 = 2e-5

Requirements:
  - Python >= 3.8
  - numpy >= 1.20
  - scipy >= 1.9 (for scipy.stats.permutation_test)
  - pandas >= 1.3

Usage:
  python statistical_tests.py --data-dir ./data

The data directory should contain the following CSV files:
  - RL_syntactic_quality.csv
  - RL_pragmatic_quality.csv
  - RL_semantic_quality.csv
  - sft_syntactic_quality.csv
  - sft_pragmatic_quality.csv
  - sft_semantic_quality.csv

Each CSV uses semicolons as delimiters. The first column is the BPMN sample
identifier; remaining columns are model configuration names with per-sample
quality scores (empty cells indicate invalid/missing outputs).

Reference:
  Good, P. I. (2005). Permutation, Parametric, and Bootstrap Tests of
  Hypotheses (3rd ed.). Springer. doi:10.1007/b138696


