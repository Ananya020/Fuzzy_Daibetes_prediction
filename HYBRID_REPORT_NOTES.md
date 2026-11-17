# Hybrid GA + NN + Fuzzy: report notes

All numbers below come from `hybrid_experiment.py` (10 seeds, test rows are original
data only). Full tables: `results_hybrid.md`, `results_hybrid.csv`. Figures: `fig_hybrid_*.png`,
`fig_fuzzy_system.png`, `fig_synthetic_realism.png`, `fig_risk_levels.png`.

## 1. Architecture

```
8 raw features -> scaler -> BPN / GA-BPN -------- p_nn ---------\
      |                                                           +-> fused = (1-a)*p_nn + a*p_fuzzy
      +-> Glucose, BMI, Age, DPF -> 14 fuzzy rules -> p_fuzzy ---/
                                    (Sugeno, AND = min)             |
                                                                    +-> fuzzy output sets -> Low / Moderate / High / Very high
```
* **NN (Unit 1):** 8-12-1 sigmoid network, backprop with momentum (unchanged).
* **GA (Units 4, 5):** (a) finds start weights for the network (unchanged `run_ga`);
  (b) a second GA over 23 genes tunes 14 rule weights, one shift and one stretch of the
  membership partition per input, and the fusion weight alpha.
* **Fuzzy (Unit 3):** trapezoidal membership functions anchored on clinical cut-offs
  (2-h glucose, BMI, age bands), zero-order Sugeno rules, linguistic output levels.
* **Why late fusion:** the network is untouched, the fuzzy layer can be switched off
  (alpha = 0), so each component can be ablated cleanly; and the fuzzy output sets give
  an interpretable risk level plus "which rules fired" for any patient.

## 2. Data handling
* **Bug fixed:** `diabetes.csv` has no header; `pd.read_csv` default dropped patient 1
  (767 rows). Now 768 rows (500/268). Earlier numbers in the handoff came from 767 rows.
* Split 70/15/15 stratified (538/115/115) of the ORIGINAL rows; medians, scaler and the
  synthetic generator use training rows only. Validation: threshold, alpha, fuzzy-GA fitness.
  Test: reporting only.
* **Synthetic data:** one multivariate Gaussian per class (log1p for Insulin, DPF, Age,
  Pregnancies), integer rounding, clipping to the training range, 1 synthetic row per real
  row per class (class rate identical: 0.349). Added to TRAIN only; val/test stay original.
  Realism: means match (<1%), std of Insulin/DPF/Age 8-16% too small, correlation matrices
  differ by 0.036 on average; worst KS distance is Insulin (0.26) because the real column has
  a median-imputation spike a Gaussian cannot copy.
* Different split code than `diabetes_ga_bpn.py` (numpy instead of sklearn), so numbers are
  not directly comparable with the earlier experiment tables.

## 3. Measured results (not a success story)
Test, mean over 10 seeds, threshold tuned on validation:

| Model | F1 | Recall | AUC |
|---|---|---|---|
| BPN | 0.694 | 0.790 | 0.844 |
| GA + BPN | 0.679 | 0.792 | 0.846 |
| BPN + Fuzzy | 0.678 | 0.815 | 0.844 |
| GA + BPN + Fuzzy | 0.684 | 0.710 | 0.844 |
| Fuzzy only (no network) | 0.660 | 0.780 | 0.824 |

* **No configuration is significantly better than plain BPN** on F1, AUC or log-loss
  (F1 and AUC p > 0.28, log-loss p > 0.07; std across seeds is 0.03-0.05, larger than every difference).
* The expert fuzzy rules alone reach AUC 0.824 with zero training, i.e. the rules encode
  real signal, but fusing them with the network adds nothing measurable.
* The only differences with p < 0.05 are: full hybrid has lower recall than GA+BPN
  (-0.082, p = 0.020), and synthetic data hurts in some cells (below). With 45 uncorrected
  tests (Bonferroni level 0.0011) none of these should be called established.
* **Synthetic data did not help.** Mean F1 and AUC are lower or equal in all four
  configurations; BPN F1@tuned -0.053 (p = 0.006, uncorrected). Plausible reason: the
  generator only re-samples the mean and covariance already estimated from the training rows,
  so it adds no information and distorts the Insulin spike. Not tested further.
* **Fuzzy GA overfits validation:** alpha moves to 0.71 (near the 0.8 cap) and the rule
  weights differ strongly between seeds (std 0.3), while test log-loss gets no better.
  A 23-gene search on 115 validation rows is too flexible.
* **Interpretability is the real gain.** Linguistic levels are monotone with observed
  outcome for the full hybrid: Low 0.11, Moderate 0.37, High 0.62, Very high 0.81 diabetes
  rate (BPN probability binned the same way: 0.12 / 0.41 / 0.60 / 0.81, so the levels are as
  informative, not more). The fuzzy layer additionally supplies rule explanations.

## 4. Why a GA + NN + Fuzzy hybrid MAY outperform a plain network (hypotheses)
Phrase these as motivation, then state the measured outcome above.
1. The GA searches globally and needs no gradient, so it can optimise non-differentiable
   objectives and avoid poor random starts; backprop then refines locally.
2. Fuzzy rules inject domain knowledge (glucose, BMI, age thresholds) that a small noisy
   dataset (768 rows) may not let the network learn on its own.
3. Fusion combines two models with different error patterns; if their errors are only
   partly correlated, the average has lower variance.
4. Fuzzy output sets turn a probability into graded, human-readable risk levels.
On this dataset, (1)-(3) did not produce a measurable accuracy gain; (4) held.

## 5. Limitations (state them)
* 115 validation and test rows per seed; seed-to-seed std is 0.03-0.05.
* Fuzzy anchors were set from clinical thresholds before the experiment, but the full
  dataset's glucose/BMI/age distribution is public knowledge to the author; membership
  functions were not re-tuned after seeing results.
* Same compute budgets as the original project (BPN 200 epochs; GA-BPN GA + 100 epochs).
* `hybrid_model.npz` is the seed-0 full hybrid on original data, saved without choosing by
  test score. The Streamlit app does not use it yet.
