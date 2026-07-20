# Excel Templates

Excel template files (`.xltx`) supporting the thesis workflow. Double-clicking
a `.xltx` in Excel (or LibreOffice Calc) creates a new untitled workbook from
it, so the template itself is never overwritten. In both templates, **yellow
cells with blue text are the cells you fill in** — everything else is computed
by formulas.

## `dataset-entry-template.xltx`

For collecting a new daily-returns dataset in the same layout as
[`../DATASETS/2001-2017-ATHEX-RETURNS.csv`](../DATASETS/2001-2017-ATHEX-RETURNS.csv).

- **Data** sheet: one trading day per row — `Date` (dd-mmm-yy) and `Returns`
  as a decimal fraction (e.g. `-0.0206` = −2.06 %). An example row shows the
  expected format; rows are pre-formatted down to row 5001 (~20 years of
  daily data).
- **Summary** sheet: auto-computed descriptive statistics (observations, mean,
  standard deviation, annualised volatility, min/max, date range,
  positive/negative day counts).
- When finished, export the Data sheet as CSV (keeping the header row) to use
  it with `ATHEXNEURALNETS.ipynb` or `train_models.py`.

## `experiment-results-template.xltx`

For logging training runs when retraining or extending the models
(`train_models.py`).

- **Experiment Log** sheet: one row per run — model, look-back window, epochs,
  batch size, and the four metrics reported by the notebook (train/test MAE
  and RMSE, per Twomey & Smith 1995). The *overfit gap* column
  (Test MAE − Train MAE) is computed automatically. Two example rows record
  the scores of the original thesis models (`MLP2.h5`, `LSTM.h5`) from
  `ATHEXNEURALNETS.ipynb`.
- **Comparison** sheet: auto-computed best run per metric (via
  `INDEX`/`MATCH`) and average test scores per model.
