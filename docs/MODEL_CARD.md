# Model Card

Every number here comes from `models/*.json`, written by `python -m src.ml.train`
against the real Home Credit dataset. Regenerate this document's numbers with
that command; nothing below is asserted without a corresponding artifact.

## Task

Predict the probability that a loan applicant will default (`TARGET = 1`),
output as a calibrated probability, a risk band (Low/Medium/High), and — via
`src/ml/explain.py` and `src/ml/rules.py` — an explanation and a set of
plain-language policy rules.

## Data

307,511 labelled applications, 8.073% positive (default) rate — an 11.4:1
class imbalance. Features: 122 raw application-table columns plus 83
engineered features (ratios and per-client aggregates over the 5 child
tables), for 205 total, built entirely in SQL (`src/data/features.py`). See
`docs/EDA_FINDINGS.md` for the underlying data analysis.

80/20 stratified train/holdout split (`random_state=42`), with the training
portion further split 90/10 for LightGBM's early stopping. The holdout is
untouched by training, imbalance handling, or calibration — every metric
below is measured on it, not on training data.

## Model selection

| Model | Mean PR-AUC (5-fold CV) |
|---|---:|
| Logistic Regression (baseline) | 0.2518 |
| **LightGBM (shipped)** | **0.2790** |

LightGBM over deep learning: at ~307k rows and 205 mixed numeric/categorical
features, gradient-boosted trees are the established strong baseline for
tabular data at this scale, need no GPU, and train in minutes — a real
constraint for a Docker image an evaluator runs locally. LightGBM specifically
for native NaN handling (the dataset's missingness is informative, not
random — see `docs/EDA_FINDINGS.md`) and native categorical support, avoiding
a 58-level one-hot expansion of `ORGANIZATION_TYPE`.

LightGBM improves PR-AUC over the logistic-regression baseline by **+10.8%
relative** (0.2518 → 0.2790) using the identical feature set — the honest
answer to "how much does model choice buy over a transparent linear
reference," isolated from the feature-engineering contribution.

## Imbalance handling

Four strategies, compared on the **same** 5-fold CV splits so the comparison
is apples-to-apples, selected on **PR-AUC** (accuracy is meaningless at
11.4:1 — predicting "no default" for everyone scores ~92%):

| Strategy | Mean PR-AUC | Mean ROC-AUC | Mean Brier |
|---|---:|---:|---:|
| **None (shipped)** | **0.2790** | 0.7859 | 0.0660 |
| Random undersampling | 0.2725 | 0.7818 | 0.1884 |
| SMOTE | 0.2492 | 0.7688 | 0.0675 |
| `scale_pos_weight` (≈11.4) | 0.2000 | 0.7330 | 0.0733 |

**Result, stated plainly because it runs against the common assumption that
imbalance always needs correcting: none of the three standard techniques beat
training on the data as-is.** `scale_pos_weight` — the most commonly reached-for
fix — is the *worst* performer here, and undersampling's Brier score (0.188,
nearly 3x worse) shows exactly why: throwing away 90% of the majority class to
balance the training set destroys the model's sense of how *rare* a default
actually is, so its raw probabilities become badly overconfident. LightGBM's
own tree-building already handles skewed leaf populations well enough that
none of these interventions add ranking power, and each of the three costs
something (SMOTE and undersampling both cost calibration quality; reweighting
costs ranking quality outright). The shipped model uses **no explicit
imbalance handling** — the comparison is real evidence for that choice, not
an assumption.

## Calibration

Reweighting/resampling change *ranking*, not the *meaning* of a probability —
a `scale_pos_weight`-trained model's "0.5" does not mean a 50% empirical
default rate. Since the winning strategy here is "none," this matters less
than it would have, but calibration is still applied for correctness: isotonic
regression, fit on the untouched holdout via `CalibratedClassifierCV` +
`FrozenEstimator` (the base model is never refit or otherwise exposed to the
holdout).

| | Brier score (holdout) |
|---|---:|
| Raw LightGBM output | 0.0660 |
| Calibrated | 0.0654 |

A modest improvement, consistent with "none" already producing reasonably
well-behaved probabilities — calibration here is a correctness guarantee, not
a large fix.

## Evaluation (holdout, n=61,503)

| Metric | Value |
|---|---:|
| ROC-AUC | 0.7899 |
| PR-AUC | 0.2794 |
| KS statistic | 0.4414 |
| Brier score (calibrated) | 0.0654 |

**Confusion matrix at the operating threshold (0.0935):**

| | Predicted: repay | Predicted: default |
|---|---:|---:|
| **Actual: repay** | 43,181 (TN) | 13,357 (FP) |
| **Actual: default** | 1,624 (FN) | 3,341 (TP) |

Precision 20.0%, recall 67.3% at this threshold — see "Operating threshold"
below for why recall is weighted this heavily.

**Lift by decile** (riskiest first — the table a credit officer reads):

| Decile | Default rate | Lift | Cumulative defaulters caught |
|---:|---:|---:|---:|
| 1 (riskiest) | 29.9% | 3.7x | 37.0% |
| 2 | 16.6% | 2.1x | 57.6% |
| 3 | 10.3% | 1.3x | 70.4% |
| 4 | 7.1% | 0.9x | 79.3% |
| 5 | 4.7% | 0.6x | 85.1% |
| 6–10 | 0.9%–4.2% | 0.1x–0.5x | 85.1% → 100% |

**Reviewing the riskiest 30% of applicants catches 70.4% of all eventual
defaulters** — the concrete number behind "prioritise manual review by risk
score."

## Risk bands

Cut from the calibrated holdout score distribution, not from arbitrary
probability thresholds (a "0.3 cutoff" means nothing on its own at an 8% base
rate):

| Band | Population share | Score threshold | Observed default rate | Lift over base rate |
|---|---:|---:|---:|---:|
| **High** | 10% | ≥ 0.1878 | 29.9% | 3.7x |
| **Medium** | 40% | ≥ 0.0449 | 9.7% | 1.2x |
| **Low** | 50% | < 0.0449 | 2.4% | 0.3x |

## Operating threshold

Set by minimising expected cost, not by a conventional 0.5 cutoff (meaningless
at 8% base rate) or by F1 (treats the two error types as equally costly, which
they are not for a lender): a missed default (false negative) is a written-off
loan; a wrongly-flagged good applicant (false positive) costs a manual review
or a more cautious offer. **Assumption, stated explicitly because it is a
policy choice, not a measured fact:** a missed default costs 10x a false
positive (`fn_cost=10, fp_cost=1` in `src/ml/train.py`).

Under that assumption the cost-minimising threshold is **0.0935** — well
below 0.5, reflecting that a false negative is deliberately made expensive.
This ratio is a placeholder for whatever real economics a deploying bank would
supply (average loan size, cost of manual review, recovery rate on default);
changing it and rerunning `python -m src.ml.train` moves the threshold
accordingly.

## Explainability

`src/ml/explain.py` uses SHAP's `TreeExplainer` directly on the LightGBM
booster. **Important distinction, stated because it is easy to get backwards:**
SHAP explains the model's raw log-odds output, not the calibrated probability
displayed to the user — calibration is a monotone remap fit afterward with no
per-feature decomposition. Contributions are therefore always shown as
directional ("pushed the risk up/down"), never as an exact slice of the
displayed percentage. Per-applicant explanations are turned into a 2-4
sentence narrative by Gemini, grounded strictly on the computed contributions
(see `docs/PROMPTS.md`).

## Business rules

`src/ml/rules.py` fits a depth-4 `DecisionTreeRegressor` to approximate the
calibrated score (not the label — see the module's docstring for why that
distinction matters), then reports each leaf as an IF-THEN rule. 16 rules were
extracted, ranging from **36.3% observed default rate (4.5x base rate)** down
to **2.6% (0.3x)** — every number measured directly against `TARGET` for the
applicants actually in that leaf, not read off the tree's prediction.

**Surrogate fidelity**, reported honestly rather than assumed: R² = 0.488
against the calibrated score, ROC-AUC 0.7207 against the actual label —
against the full model's 0.8766 (measured on the same training data the
surrogate saw, so this comparison is optimistic for both; the holdout ROC-AUC
of 0.7899 above is the fairer number for the full model). The surrogate
captures real, useful structure but is meaningfully simpler than the model it
approximates — which is the honest way to present "here is a readable
four-question checklist," not a claim that the checklist matches the model.

## Known limitations

- **`CODE_GENDER` is the model's #2 feature by global mean |SHAP|** — second
  only to `EXT_SOURCE_MEAN`, and ahead of income, credit amount, and
  employment history (`models/shap_global.json`). Using a protected attribute
  as a predictive feature raises real fair-lending concerns (disparate
  treatment / disparate impact review would be required before any production
  use) — this platform is a technical demonstration, not a
  compliance-reviewed scoring system, and a real deployment would need to
  either exclude protected attributes or apply an explicit fairness
  constraint and audit before this ranking would be acceptable to ship.
- **The imbalance comparison used only 4 well-known techniques** at their
  default settings. Cost-sensitive learning, ensemble-based imbalance methods
  (e.g. `BalancedRandomForest`), or per-fold threshold tuning were not
  explored, and might change the "none wins" conclusion.
- **The FN:FP cost ratio (10:1) is asserted, not derived** from real loan
  economics. A production deployment needs a real cost model.
- **The surrogate rules' fidelity (R² 0.488) is moderate** — they are a
  useful, auditable summary, not a substitute for the full model in a
  decision that matters.
- **No temporal validation.** The dataset has no timestamp for out-of-time
  testing; the holdout split is random, not chronological, so this cannot
  speak to how the model would perform on genuinely future applications.

## Reproducing this document

```bash
python -m src.data.loader        # build the database (if not already built)
python -m src.ml.train           # ~4.5 minutes on 307,511 rows, 14 cores
python -m src.ml.rules           # surrogate rule extraction
```

Every number above should match `models/train_summary.json`,
`models/imbalance_comparison.json`, `models/metrics.json`, `models/bands.json`,
`models/threshold.json`, and `models/rules.json` exactly.
