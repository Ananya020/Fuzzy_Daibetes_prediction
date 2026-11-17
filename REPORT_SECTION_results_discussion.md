# 7. Results and Discussion

## 7.1 Experimental protocol

All results use the Pima Indians Diabetes dataset (768 patients, 8 features, 500 non-diabetic and 268 diabetic). Zero values in Glucose, Blood Pressure, Skin Thickness, Insulin and BMI were treated as missing and replaced by the median of the training rows. (While preparing these experiments we found that the data loader skipped the first row of the CSV file because the file has no header; this was corrected, so every experiment uses all 768 rows.)

To avoid data leakage, the following rules were applied in every experiment:

1. **Final holdout.** 154 rows (20%) were set aside and used exactly once, for the final chosen model. The remaining 614 rows form the development set.
2. **Cross-validation.** The development set was evaluated with stratified 5-fold cross-validation repeated 3 times, giving 15 test folds of 122 rows. Results are reported as mean ± standard deviation over these 15 folds.
3. **Inner validation split.** Within each training portion, 20% of the rows form a validation set. It is used only to choose the decision threshold (and the fuzzy fusion weight). Test folds are never used for tuning.
4. **Training-only statistics.** Medians, feature scaling, the Gaussian generator and SMOTE use training rows only. Synthetic rows are added to the training part only; validation and test rows are always original patients.
5. **GA fitness.** The genetic algorithm's fitness is computed by 3-fold cross-validation on the inner-training rows, so the validation rows stay unseen by the GA.

Statistical comparisons use the Nadeau-Bengio corrected resampled paired t-test, which accounts for the fact that cross-validation training sets overlap and therefore makes the test more conservative than an ordinary paired t-test. The 12 comparisons were not corrected for multiple testing.

## 7.2 Models compared

| Model | Description |
|---|---|
| Logistic regression | Simplest baseline, no hidden layer |
| BPN | 8-12-1 sigmoid network, backpropagation with momentum |
| GA-BPN | GA finds the starting weights, backpropagation fine-tunes |
| BPN + Fuzzy | BPN fused with an expert fuzzy rule base (14 rules on Glucose, BMI, Age, DPF) |
| BPN + Gaussian / + SMOTE | BPN trained with Gaussian-synthetic rows or SMOTE rows added |
| BPN + Eng | Adds engineered features: Glucose×BMI, Age×BMI, Age×Glucose, log(DPF), Glucose², BMI², Pregnancies/Age |
| GA-Opt | GA selects the feature subset and the network hyperparameters (hidden units, learning rate, momentum, epochs) |

## 7.3 Main results

Table 7.1 gives test-fold performance when the classification threshold is tuned for **accuracy** on the validation set.

**Table 7.1: Threshold tuned for accuracy (mean ± std over 15 test folds)**

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic regression | 0.755 ± 0.034 | 0.737 | 0.506 | 0.584 | 0.838 |
| BPN | 0.769 ± 0.036 | 0.718 | 0.571 | 0.627 | 0.838 |
| BPN + Fuzzy | 0.765 ± 0.041 | 0.710 | 0.570 | 0.622 | 0.840 |
| BPN + Gaussian | 0.766 ± 0.034 | 0.765 | 0.501 | 0.595 | 0.838 |
| BPN + SMOTE | 0.769 ± 0.042 | 0.694 | 0.637 | 0.655 | 0.838 |
| BPN + Eng | 0.765 ± 0.040 | 0.722 | 0.553 | 0.619 | 0.836 |
| GA-BPN | 0.761 ± 0.037 | 0.697 | 0.560 | 0.614 | 0.838 |
| GA-Opt (features + hyperparameters) | 0.757 ± 0.032 | 0.730 | 0.504 | 0.587 | 0.834 |

**Table 7.2: Default threshold 0.5 (selected models)**

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic regression | 0.774 ± 0.039 | 0.737 | 0.556 | 0.631 | 0.838 |
| BPN | 0.767 ± 0.044 | 0.695 | 0.595 | 0.639 | 0.838 |
| BPN + SMOTE | 0.754 ± 0.039 | 0.624 | 0.751 | 0.679 | 0.838 |
| GA-BPN | 0.762 ± 0.035 | 0.694 | 0.578 | 0.627 | 0.838 |

(See `results_accuracy.md` for all 12 models; Figure `fig_accuracy_models.png` plots accuracy for both threshold rules.)

**Key observation.** All models lie in a narrow band of about 0.75 to 0.77 accuracy and about 0.83 to 0.84 AUC. The fold-to-fold standard deviation (0.03 to 0.04) is larger than every difference between models. None of the 12 paired comparisons on accuracy was statistically significant (all p ≥ 0.33).

## 7.4 Where changes in performance come from

**Threshold tuning.** Tuning the threshold for accuracy on the validation set did not beat the default 0.5 (difference +0.002, p = 0.86). The validation set has only about 98 rows, so the accuracy-maximising threshold is largely determined by chance. On the final holdout, for example, the BPN's validation-tuned threshold was 0.70, which sharply lowered its recall (0.54) without improving accuracy beyond chance variation.

**SMOTE.** SMOTE raised recall from 0.595 to 0.751 and F1 from 0.639 to 0.679 at threshold 0.5, while accuracy fell slightly (0.767 to 0.754). The AUC is unchanged (0.838). Since AUC measures how well the model *ranks* patients regardless of threshold, this shows that SMOTE did not make the network a better classifier; it balanced the classes to 50:50, which shifts the predicted probabilities upward, so more patients cross the 0.5 threshold. In other words, SMOTE changes the recall-precision trade-off, not the discriminating ability. This is useful when missing a diabetic patient is costlier than a false alarm, but it is not an accuracy improvement. SMOTE and Gaussian augmentation did not differ significantly on accuracy (p = 0.90).

**Gaussian augmentation.** The Gaussian generator preserves the class ratio and feature correlations of the training data (mean correlation difference 0.036), but it only resamples statistics already estimated from the same training rows, so it adds no new information. It also reproduced the Insulin feature poorly (KS distance 0.26), because the real column contains a spike of median-imputed values that a Gaussian cannot copy. In the first study (Section 7.6) it slightly lowered performance.

**Engineered features.** Adding interaction and transformed features did not help (−0.004, p = 0.80), for either the network or logistic regression. A plausible explanation, not tested separately, is that the hidden layer can already form such products, and that the added features are strongly correlated with the originals, giving 8 additional inputs with little new information on 614 training rows.

**GA for feature selection and hyperparameters.** The GA found configurations that scored well on its own cross-validation fitness, but these did not generalise better to the test folds (−0.012 vs BPN, p = 0.33). The selected feature subsets were unstable: Glucose was kept in 6 of 15 folds, and Age in only 1 of 15, even though both are clinically important. A search that picks different features on different samples of the same data indicates that it is fitting noise in the fitness estimate (about 400 rows), a known risk of wrapper-based selection on small datasets.

**GA-based starting weights.** In line with the first project experiment, GA initialisation did not change final accuracy (0.761 vs 0.769, p = 0.43). Its benefit is a lower starting loss and faster early convergence, at the price of 1,600 extra forward passes.

**Model complexity.** Logistic regression, with no hidden layer, scored within 0.014 of the BPN (p = 0.46). The neural network's extra capacity therefore gives no measurable accuracy gain on this dataset.

## 7.5 Choice of the simplest defensible model

We selected the best model by mean cross-validated accuracy (BPN + SMOTE, 0.769), then chose the model with the fewest components whose accuracy was not significantly lower than the best (corrected t-test, p > 0.05). Every model qualified, so the rule selects **logistic regression**. On the untouched 154-row holdout it achieved accuracy 0.766 (95% Wilson interval 0.693 to 0.826), compared with 0.779 for the BPN and 0.766 for BPN + SMOTE.

This conclusion has two limits. First, failing to find a significant difference is weak evidence that models are equivalent; with 15 correlated folds the test has low power. Second, the holdout interval spans about ±7 percentage points, so the holdout cannot rank the models. The defensible statement is that, with the available data, **no tested model is distinguishable from the simplest one in accuracy**.

## 7.6 Fuzzy layer and hybrid model (first study)

The fuzzy layer is placed after the network: the final score is a weighted average of the network output and a zero-order Sugeno rule base score, and a second set of fuzzy output sets converts it into Low, Moderate, High and Very high risk levels. In an earlier experiment (10 random 70/15/15 splits, 115 test rows each) four configurations were compared:

| Model | F1 (tuned) | Recall | AUC |
|---|---|---|---|
| BPN | 0.694 | 0.790 | 0.844 |
| GA + BPN | 0.679 | 0.792 | 0.846 |
| BPN + Fuzzy | 0.678 | 0.815 | 0.844 |
| GA + BPN + Fuzzy (full hybrid) | 0.684 | 0.710 | 0.844 |
| Fuzzy rules only (no network) | 0.660 | 0.780 | 0.824 |

No configuration was significantly better than the plain BPN on F1, AUC or log-loss. Two findings are worth noting:

* The fuzzy rules, which were set from clinical thresholds and not trained, reach an AUC of 0.824 alone, close to the trained network (0.844). The rules therefore encode real signal, but combining them with the network added nothing measurable.
* When the GA tuned the rule weights, membership shifts and fusion weight on only 115 validation rows, the learned rule weights varied strongly between splits (standard deviation about 0.3), a sign of overfitting.

The measurable benefit of the fuzzy layer is **interpretability**. The risk levels correspond to observed outcomes: the observed diabetes rate was 0.11 (Low), 0.37 (Moderate), 0.62 (High) and 0.81 (Very high) for the full hybrid (`fig_risk_levels.png`). Binning the BPN probability the same way gives similar rates (0.12, 0.41, 0.60, 0.81), so the levels are as informative as the probability, and in addition the system can state which rules produced a prediction.

## 7.7 Discussion

The expected motivation for a hybrid GA + neural network + fuzzy system is that the GA avoids poor random starting weights and can search non-differentiable objectives, fuzzy rules add domain knowledge that a small dataset may not let the network learn, and fusing two models with different error patterns can reduce variance. In our experiments the first three effects did not produce a measurable accuracy gain. The consistent explanation is that accuracy on this dataset is limited by the data and not by the model: with 768 rows, noisy measurements and many imputed values, even simple logistic regression reaches an AUC of 0.838, and all more elaborate models reach the same value. The practical contributions of the hybrid are therefore (i) an adjustable recall-precision trade-off through the threshold or SMOTE, and (ii) interpretable, rule-based risk levels.

## 7.8 Limitations

* Each test fold has about 122 rows and the final holdout 154, so differences below roughly 3 to 5 percentage points cannot be detected reliably.
* The 15 cross-validation folds are not independent; the corrected t-test partly accounts for this, but p-values should be read as indicative.
* The 12 comparisons were not corrected for multiple testing. A Bonferroni-adjusted level would be 0.004; no comparison approached it.
* The membership functions and rules were set before the experiments from clinical cut-offs and were not changed afterwards, but they remain a design choice of the authors.
* The first study and the accuracy study use different data splits and are not directly comparable in absolute numbers.
* The system is an educational prototype and is not a clinically validated diagnostic tool.
