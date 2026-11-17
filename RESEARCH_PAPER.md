# A Genetic-Algorithm-Optimised Neural-Fuzzy System for Diabetes Risk Estimation: An Honest Evaluation of Hybrid Soft-Computing Components on the Pima Indians Dataset

**[Author name]**, B.Tech, [Department], [Institution]
Guide: [Teacher name] · Subject: Neuro-Fuzzy and Genetic Programming · [Roll number] · [Month Year]

---

## Abstract

Hybrid soft-computing systems are often presented as improvements over a plain neural network, but on small clinical datasets the claim is rarely tested under strict leakage control. We build a diabetes risk estimator from three techniques: a backpropagation neural network (BPN), a real-coded genetic algorithm (GA) and a zero-order Sugeno fuzzy inference layer, all implemented from scratch in NumPy. The GA is used to initialise network weights and to select features and hyperparameters; the fuzzy layer fuses 14 clinical rules with the network output and converts the score into linguistic risk levels. We also test Gaussian and SMOTE data augmentation, engineered interaction features and threshold tuning. On the Pima Indians Diabetes dataset (768 patients), we evaluate every variant with a held-out set, stratified 5-fold cross-validation repeated three times, validation-only tuning, and a corrected resampled t-test. All models reach about 0.75 to 0.77 accuracy and 0.83 to 0.84 AUC; none of the 12 paired accuracy comparisons is significant (p ≥ 0.33), and logistic regression is statistically indistinguishable from the best model. SMOTE raises recall from 0.595 to 0.751 at threshold 0.5 without changing AUC, so it shifts the decision boundary and does not improve ranking. The fuzzy layer adds no measurable accuracy but yields risk levels whose observed diabetes rates rise monotonically (0.11, 0.37, 0.62, 0.81) and gives rule-based explanations. We conclude that on this dataset accuracy is limited by data, not model sophistication, and that the practical value of the hybrid is interpretability and an adjustable recall/precision trade-off.

**Keywords:** genetic algorithm, backpropagation neural network, fuzzy inference, diabetes prediction, SMOTE, feature selection, cross-validation, hybrid soft computing

---

## I. Introduction

Early identification of people at risk of type 2 diabetes relies on a few routine measurements such as plasma glucose, body mass index and age. Machine-learning models can combine these measurements, but a clinical user also needs to know why a score is high. Neural networks are accurate but opaque, genetic algorithms can optimise non-differentiable objectives, and fuzzy logic expresses expert knowledge in readable rules. Combining the three is a natural idea, and many studies report improvements from such hybrids.

Small clinical datasets, however, make improvements hard to demonstrate. With a few hundred rows, test-fold standard deviations are comparable to the differences between models, and tuning on the test set, selecting the best of many configurations, or leaking preprocessing statistics can easily manufacture an apparent gain [12]. This paper therefore has two goals. First, to build a transparent GA + neural network + fuzzy system for diabetes risk. Second, to measure honestly what each component contributes under a protocol that never touches test data during tuning.

**Contributions.**
1. A complete hybrid pipeline written from scratch: backpropagation network, GA (for start weights, and for feature and hyperparameter selection), Sugeno fuzzy layer with linguistic risk levels, Gaussian and SMOTE augmentation.
2. A leakage-controlled evaluation: final holdout, repeated stratified cross-validation, validation-only threshold tuning, and a corrected resampled t-test [9].
3. An ablation showing that none of the tested components improves accuracy significantly, together with explanations of where the observable changes (recall, F1) come from.
4. A working application that presents the model, the fuzzy explanations and all results.

---

## II. Background and Related Work

**Backpropagation networks.** Multilayer perceptrons trained with error backpropagation [2] are the standard nonlinear classifier for tabular medical data. Their weights start from random values, so training can end in different local solutions.

**Genetic algorithms.** A GA evolves a population of candidate solutions using selection, crossover and mutation [3], [4]. For neural networks, GAs have been used to determine weights directly [5], which is the idea behind "GA-based weight determination" in hybrid systems, and to select feature subsets [10], [11]. Wrapper-based feature selection is known to overfit the performance estimate used as its fitness when data are scarce [10], [12].

**Fuzzy inference.** Fuzzy sets [6] let a value belong partly to several linguistic categories ("glucose is high"). In zero-order Sugeno systems [7] each rule has a constant consequent and the output is a weighted average of consequents, which is simple and differentiable-free, so it can be tuned by a GA.

**Class imbalance and augmentation.** SMOTE [8] oversamples the minority class by interpolating between a minority sample and one of its nearest minority neighbours. It changes the class prior seen during training and therefore shifts predicted probabilities.

**Statistical comparison.** Overlapping training sets make standard paired t-tests over cross-validation folds too optimistic. The Nadeau–Bengio corrected resampled t-test [9] adds a correction term for this dependence; comparisons of classifiers on a single dataset are discussed in [14].

---

## III. Dataset and Preprocessing

**Data.** We use the Pima Indians Diabetes dataset [1] from the National Institute of Diabetes and Digestive and Kidney Diseases: 768 women aged 21 or older, eight predictors and a binary outcome (diabetes within five years). The classes are 500 negative and 268 positive (34.9% positive). The predictors are number of pregnancies, plasma glucose (2-hour oral glucose tolerance test), diastolic blood pressure, triceps skin-fold thickness, 2-hour serum insulin, BMI, diabetes pedigree function (DPF) and age.

**Missing values.** In Glucose, Blood Pressure, Skin Thickness, Insulin and BMI, a value of 0 is physiologically impossible and denotes a missing measurement (Insulin is missing for about half of the patients). Zeros in these columns are set to missing and imputed with the median of the **training** rows.

**Data-handling defect found and fixed.** The CSV file used has no header row. The initial loader used the default `header=0`, silently consuming the first patient as column names and leaving 767 rows. The loader now detects a header explicitly, and all reported results use all 768 rows.

**Scaling.** Features are standardised with the mean and standard deviation of training rows only.

---

## IV. Methods

### A. Backpropagation neural network

The network has 8 inputs, 12 sigmoid hidden units and one sigmoid output interpreted as P(diabetes):

h = σ(X·W₁ + b₁),  p = σ(h·W₂ + b₂)

It has 8×12 + 12 + 12 + 1 = 121 weights, stored in one flat vector. Training minimises binary cross-entropy by full-batch gradient descent with momentum (learning rate 0.1, momentum 0.9, 200 epochs). With a sigmoid output the output delta is (p − y)/N, and the hidden delta is (δ_out·W₂ᵀ) ⊙ h ⊙ (1 − h). Initial weights are drawn from N(0, 0.5).

### B. Genetic algorithm

The GA is real-coded with tournament selection (k = 3), uniform crossover (probability 0.8), Gaussian mutation (per-gene probability 0.1, σ = 0.3) and elitism (the two best individuals are copied unchanged). It is used in three ways.

1. **GA-based start weights (GA-BPN).** The chromosome is the 121-weight vector; fitness is 1/(1 + MSE) on the training rows; population 40, 40 generations (1,600 fitness evaluations). The best individual initialises the network, which backpropagation then fine-tunes for 100 epochs.
2. **GA feature and hyperparameter selection (GA-Opt).** The chromosome has 19 genes in [0, 1]: 15 feature on/off genes (8 original and 7 engineered features), hidden units (4–24), learning rate (0.01–0.5, logarithmic), momentum (0–0.95) and epochs (50–300). Fitness is the mean accuracy of 3-fold cross-validation on the inner-training rows, minus 0.003 per selected feature, plus a small log-loss tie-breaker (0.02/(1 + log-loss)). One individual is seeded with the default configuration so the GA cannot finish below it on its own fitness. Population 16, 12 generations.
3. **GA tuning of the fuzzy layer (first study only).** The chromosome has 23 genes: 14 rule weights, one shift and one stretch of the membership partition for each of four inputs, and the fusion weight. Fitness is 1/(1 + validation log-loss) of the fused score; population 40, 40 generations, mutation probability 0.15 (σ = 0.12).

### C. Fuzzy inference layer

A zero-order Sugeno system uses four inputs: Glucose, BMI, Age and DPF. Each input has overlapping trapezoidal membership functions (for example Glucose: Low, Medium, High). Anchors follow clinical cut-offs for 2-hour glucose (WHO: below 140 mg/dL normal, 140–199 impaired [15]) and BMI (25 and 30). The membership functions and rules were fixed before the experiments and not modified afterwards.

Fourteen rules have the form *IF Glucose is High AND BMI is High THEN risk = 0.95*, with AND implemented as the minimum. For rule strengths s_r, rule weights w_r and consequents c_r, the fuzzy score is

p_fuzzy = Σ w_r s_r c_r / Σ w_r s_r  (a prior of 0.35 is returned if no rule fires).

**Fusion.** The final score is p = (1 − a)·p_NN + a·p_fuzzy, with a ∈ [0, 0.8]. In the "expert" variant a is chosen on the validation set (a = 0 disables the fuzzy layer); in the "GA-tuned" variant a is a gene. Placing the fuzzy layer **after** the network (late fusion) leaves the network and its training untouched, and a = 0 gives an exact ablation.

**Linguistic output.** Four overlapping output sets map the fused score to *Low*, *Moderate*, *High* and *Very high*. The most active rules are reported as the explanation of an individual prediction.

### D. Data augmentation

**Gaussian augmentation.** One multivariate Gaussian per class is fitted to the training rows (Insulin, DPF, Age and Pregnancies on a log1p scale to reduce skew), preserving class means and the feature covariance. Samples are converted back, integer features rounded, and all values clipped to the range seen in training. One synthetic row is generated per real row of each class, so the class ratio is preserved.

**SMOTE.** For each new row, a minority sample and one of its five nearest minority neighbours (distance on standardised features) are chosen, and the new row is placed at a random point on the segment between them. This balances the classes to 50:50.

In all experiments synthetic rows are generated from, and added to, the **training** part only.

### E. Engineered features

Seven features are computed row by row: Glucose×BMI, Age×BMI, Age×Glucose, log(DPF), Glucose², BMI² and Pregnancies/Age. Together with the eight originals they form a pool of 15 candidate features.

---

## V. Experimental Design

### A. Protocol

**Study 1 (hybrid study).** Ten random stratified 70/15/15 splits (538/115/115 rows). Four configurations plus ablations are compared: BPN; GA + BPN; BPN + fuzzy; GA + BPN + fuzzy (GA-tuned fuzzy layer); and fuzzy rules alone, each with and without Gaussian synthetic training rows.

**Study 2 (accuracy study).** Twenty percent of the rows (154) are set aside as a **final holdout**, never used for tuning or selection. The remaining 614 rows are evaluated with stratified 5-fold cross-validation repeated three times (15 outer test folds of 122 rows). Inside every outer split the training rows are divided 80/20 into inner-training and validation sets. Models are fitted on the inner-training rows (plus synthetic rows generated from them); the validation set is used only for the decision threshold and the fusion weight. The GA fitness uses 3-fold cross-validation on inner-training rows only. Medians and scalers come from the outer-training rows. Compared models are:

| Model | Description |
|---|---|
| LogReg | Logistic regression, 8 features |
| LogReg+Eng | Logistic regression, 15 features |
| BPN | 8-12-1 network |
| BPN+Fuzzy | BPN + expert fuzzy layer (a chosen on validation) |
| BPN+Gauss / BPN+SMOTE | BPN trained with Gaussian / SMOTE rows added |
| BPN+Eng (+Gauss / +SMOTE) | BPN with 15 features |
| GA-BPN | GA start weights, 8 features |
| GA-Opt (+SMOTE) | GA-selected features and hyperparameters |

### B. Thresholds and metrics

Predictions use either the default threshold 0.5 or a threshold from 0.30 to 0.70 (step 0.02) that maximises accuracy on the validation set (ties go to the value closest to 0.5). We report accuracy, precision, recall, F1 and AUC (computed by pairwise comparison of scores, tie-aware). Study 1 also reports log-loss.

### C. Statistical testing

Study 2 uses the Nadeau–Bengio corrected resampled paired t-test [9] on the 15 outer-fold accuracies, with correction term (1/k + n_test/n_train). Study 1 uses an exact paired sign-flip permutation test over the ten splits. No correction for multiple comparisons is applied (12 contrasts in Study 2, 45 in Study 1); the Bonferroni level would be 0.004 and 0.0011 respectively, and p-values are treated as indicative.

### D. Model selection rule

The best model is chosen by mean cross-validated accuracy at the validation-tuned threshold. Among all models whose accuracy is not significantly lower than the best (corrected t-test, p > 0.05), the one with the fewest components is selected (components counted as network, threshold tuning, fuzzy layer, augmentation, engineered features, GA). The chosen model is then evaluated once on the holdout.

### E. Implementation

All components are written in NumPy; the experiments are seeded and reproducible. The fuzzy layer, the augmentation and the experiment scripts are separate modules, and a Streamlit application presents the model and all results.

---

## VI. Results

### A. Accuracy study (Study 2)

Table I gives test-fold performance with the threshold tuned for accuracy on the validation set (mean ± standard deviation over 15 test folds).

**Table I. Threshold tuned for accuracy on validation.**

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| LogReg | 0.755 ± 0.034 | 0.737 | 0.506 | 0.584 | 0.838 |
| LogReg+Eng | 0.753 ± 0.040 | 0.724 | 0.515 | 0.588 | 0.838 |
| BPN | 0.769 ± 0.036 | 0.718 | 0.571 | 0.627 | 0.838 |
| BPN+Fuzzy | 0.765 ± 0.041 | 0.710 | 0.570 | 0.622 | 0.840 |
| BPN+Gauss | 0.766 ± 0.034 | 0.765 | 0.501 | 0.595 | 0.838 |
| BPN+SMOTE | 0.769 ± 0.042 | 0.694 | 0.637 | 0.655 | 0.838 |
| BPN+Eng | 0.765 ± 0.040 | 0.722 | 0.553 | 0.619 | 0.836 |
| BPN+Eng+Gauss | 0.767 ± 0.029 | 0.769 | 0.499 | 0.595 | 0.839 |
| BPN+Eng+SMOTE | 0.756 ± 0.038 | 0.661 | 0.631 | 0.639 | 0.835 |
| GA-BPN | 0.761 ± 0.037 | 0.697 | 0.560 | 0.614 | 0.838 |
| GA-Opt | 0.757 ± 0.032 | 0.730 | 0.504 | 0.587 | 0.834 |
| GA-Opt+SMOTE | 0.762 ± 0.047 | 0.678 | 0.649 | 0.654 | 0.835 |

**Table II. Default threshold 0.5 (selected models).**

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| LogReg | 0.774 ± 0.039 | 0.737 | 0.556 | 0.631 | 0.838 |
| BPN | 0.767 ± 0.044 | 0.695 | 0.595 | 0.639 | 0.838 |
| BPN+SMOTE | 0.754 ± 0.039 | 0.624 | 0.751 | 0.679 | 0.838 |
| GA-BPN | 0.762 ± 0.035 | 0.694 | 0.578 | 0.627 | 0.838 |
| GA-Opt+SMOTE | 0.751 ± 0.036 | 0.624 | 0.737 | 0.674 | 0.835 |

All models lie in a narrow band, 0.75 to 0.77 accuracy and 0.83 to 0.84 AUC, and the fold-to-fold standard deviation (0.03 to 0.04) exceeds every difference between models. Table III lists the paired contrasts on accuracy.

**Table III. Paired contrasts on accuracy (corrected t-test).**

| Contrast (A vs B) | Mean difference | p |
|---|---|---|
| BPN, tuned threshold vs 0.5 threshold | +0.002 | 0.857 |
| BPN+Gauss vs BPN | −0.003 | 0.867 |
| BPN+SMOTE vs BPN | 0.000 | 0.999 |
| BPN+SMOTE vs BPN+Gauss | +0.003 | 0.898 |
| BPN+Eng vs BPN | −0.004 | 0.798 |
| LogReg+Eng vs LogReg | −0.002 | 0.872 |
| BPN+Fuzzy vs BPN | −0.003 | 0.705 |
| GA-BPN vs BPN | −0.008 | 0.433 |
| GA-Opt vs BPN | −0.012 | 0.330 |
| GA-Opt vs BPN+Eng | −0.008 | 0.607 |
| GA-Opt+SMOTE vs GA-Opt | +0.005 | 0.779 |
| BPN vs LogReg | +0.014 | 0.462 |

**Threshold tuning.** Tuning the threshold for accuracy on about 98 validation rows did not improve on 0.5 (+0.002, p = 0.86).

**SMOTE.** At threshold 0.5, SMOTE raises recall from 0.595 to 0.751 and F1 from 0.639 to 0.679, while accuracy decreases slightly (0.767 to 0.754) and AUC is unchanged at 0.838.

**Engineered features.** Interaction and transformed features did not help the network (−0.004) or logistic regression (−0.002).

**GA feature and hyperparameter selection.** GA-Opt did not outperform the default network (−0.012, p = 0.33). The GA kept on average 4.5 of 15 features, with hidden units 16.1 ± 5.5, learning rate 0.217, momentum 0.53 and 225 epochs. The selection was unstable across folds: Glucose was kept in 6 of 15 folds, BMI in 6, DPF in 6, log(DPF) in 8 and Age in only 1.

### B. Model selection and holdout

The best mean accuracy is BPN + SMOTE (0.769). Every other model is not significantly lower (p from 0.25 to 1.0), so the selection rule chooses **logistic regression** as the simplest defensible model. Table IV gives holdout performance (154 rows, evaluated once).

**Table IV. Final holdout.**

| Model | Threshold | Accuracy (95% CI) | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|---|
| LogReg (chosen) | 0.52 | 0.766 (0.693–0.826) | 0.680 | 0.630 | 0.654 | 0.829 |
| BPN + SMOTE | 0.58 | 0.766 (0.693–0.826) | 0.645 | 0.741 | 0.690 | 0.838 |
| BPN | 0.70 | 0.779 (0.707–0.837) | 0.763 | 0.537 | 0.630 | 0.838 |

The Wilson intervals span about ±7 percentage points, so the holdout cannot rank the models; its role is to confirm that the chosen model's accuracy is consistent with the cross-validated estimate.

### C. Hybrid study with fuzzy layer (Study 1)

Table V gives results over ten splits with the threshold tuned on validation F1.

**Table V. Study 1, test rows, mean over 10 splits.**

| Model | Accuracy | Recall | F1 | AUC |
|---|---|---|---|---|
| BPN | 0.757 | 0.790 | 0.694 | 0.844 |
| GA + BPN | 0.740 | 0.792 | 0.679 | 0.846 |
| BPN + Fuzzy | 0.723 | 0.815 | 0.678 | 0.844 |
| GA + BPN + Fuzzy | 0.775 | 0.710 | 0.684 | 0.844 |
| Fuzzy rules only | 0.724 | 0.780 | 0.660 | 0.824 |

No configuration is significantly better than the plain BPN on F1 (p ≥ 0.29), AUC (p ≥ 0.35) or log-loss (p ≥ 0.07; the smallest, 0.072, is for GA + BPN, which has slightly lower loss than BPN). The only difference with p < 0.05 among the four main contrasts is the full hybrid's lower recall than GA + BPN (−0.082, p = 0.020), which does not survive correction for the 45 tests. Adding Gaussian synthetic rows did not help in Study 1 either: mean F1 and AUC were lower or equal for all four configurations (for BPN, F1 −0.053, p = 0.006, uncorrected).

**Fuzzy rules alone.** The untrained expert rules achieve AUC 0.824 against 0.844 for the network, so the clinical rules carry real signal, but fusing them with the network adds nothing measurable.

**GA tuning of the fuzzy layer.** The GA moved the fusion weight to a = 0.71 ± 0.15 (expert variant: 0.16), and the learned rule weights differed strongly between splits (standard deviation about 0.3), while test log-loss did not improve (0.473 against 0.466 for the BPN). This pattern suggests overfitting of 23 genes to 115 validation rows, which motivated moving the GA to feature and hyperparameter selection in Study 2.

**Risk levels.** Table VI shows the observed diabetes rate among test rows (1,150 pooled) for each fuzzy risk level of the full hybrid.

**Table VI. Observed diabetes rate per risk level.**

| Level | Rows | Observed rate (full hybrid) | Observed rate (BPN probability binned the same way) |
|---|---|---|---|
| Low | 542 | 0.11 | 0.12 |
| Moderate | 246 | 0.37 | 0.41 |
| High | 217 | 0.62 | 0.60 |
| Very high | 145 | 0.81 | 0.81 |

The levels rise monotonically with the observed rate, but binning the raw network probability gives similar rates, so the levels are as informative as the probability, not more.

**Synthetic data realism.** Gaussian synthetic rows match the real means within 1%, preserve the positive rate (0.349) and the correlation structure (mean absolute difference 0.036), but understate the standard deviation of Insulin, DPF and Age by 8% to 16%. The largest distribution mismatch is Insulin (Kolmogorov–Smirnov distance 0.26), because the real column contains a spike of median-imputed values that a Gaussian cannot reproduce.

---

## VII. Discussion

**Why no accuracy gain?** Every model, from logistic regression to the full hybrid, produces the same ranking quality (AUC 0.83 to 0.84), and accuracy at a given threshold follows from the ranking. With 768 rows, noisy measurements and many imputed values, the signal that the eight features carry appears to be almost fully captured by a linear model, so extra capacity (a hidden layer), extra inputs (engineered features), better initialisation (GA) or extra knowledge (fuzzy rules) has little left to add. The seed-to-seed standard deviation (0.03 to 0.04) would mask any gain smaller than about three to five points.

**SMOTE and thresholds change trade-offs, not ranking.** SMOTE balances the classes, which raises predicted probabilities, so more patients cross the 0.5 threshold. Recall and F1 rise and precision and accuracy fall, while AUC, which ignores the threshold, stays at 0.838. Threshold tuning acts the same way. This is useful when a missed case is costlier than a false alarm, but it is not an improvement in discrimination.

**Gaussian augmentation.** It resamples the mean and covariance already estimated from the training rows and so adds no new information; it also misrepresents the imputation spike in Insulin. It was neutral to harmful.

**Engineered features.** A hidden layer can form products such as Glucose×BMI itself, and the engineered columns are strongly correlated with the originals. A separate test of this explanation was not run.

**GA for feature and hyperparameter selection.** The GA improved its own fitness but not test accuracy, and its feature choices varied across folds. Fitness computed from about 400 rows is noisy enough for a search to fit that noise, a known weakness of wrapper selection on small data [10], [12].

**GA start weights.** In line with the first project experiment, GA initialisation did not change final accuracy (0.761 against 0.769, p = 0.43). Its benefit, a lower starting loss and faster early convergence, costs 1,600 additional forward passes.

**Value of the fuzzy layer.** Its measurable contributions are (i) interpretability: each prediction can be traced to named rules and a linguistic level; (ii) clinical knowledge that is useful on its own (AUC 0.824 with no training). It did not raise accuracy, F1 or AUC when fused with the network.

**Model choice.** Because no model is distinguishable from the simplest one, the defensible choice is logistic regression. The hybrid remains justified only if interpretability or an adjustable recall/precision trade-off is required.

---

## VIII. Limitations

* Each test fold has about 122 rows and the holdout 154, so differences below roughly three to five accuracy points cannot be detected.
* Cross-validation folds are not independent. The corrected t-test partly accounts for this, but p-values are indicative.
* Twelve (Study 2) and 45 (Study 1) comparisons were made without multiple-testing correction.
* Failing to reject equality is weak evidence of equivalence; the rule that selects logistic regression favours simplicity because the test has low power.
* The membership functions and rules were set from clinical cut-offs by the authors before the experiments and not tuned, but they remain a design choice.
* Studies 1 and 2 use different data splits and are not directly comparable in absolute numbers.
* Compute budgets follow the original project settings (200 epochs for BPN; GA plus 100 epochs for GA-BPN) and were not matched.
* The system is an educational prototype, evaluated on one dataset from one population; it is not clinically validated and must not be used for diagnosis.

---

## IX. Conclusion and Future Work

We built and evaluated a hybrid GA + neural network + fuzzy system for diabetes risk estimation under strict leakage control. On the Pima Indians dataset, none of the components (GA initialisation, GA feature and hyperparameter selection, fuzzy fusion, Gaussian or SMOTE augmentation, engineered features, threshold tuning) improved accuracy significantly, and logistic regression is the simplest model statistically indistinguishable from the best. Observable changes in recall and F1 come from moving the decision boundary (SMOTE, threshold), not from better ranking. The fuzzy layer adds interpretability: monotone linguistic risk levels and rule-based explanations. The main lesson is methodological: on small noisy clinical datasets, gains claimed for hybrid models should be tested with held-out data, validation-only tuning and appropriate statistics.

**Future work.** Larger and additional datasets with external validation; nested cross-validation for the GA; a matched-compute baseline for GA initialisation; probability calibration and a cost-based threshold chosen with clinicians; and a fuzzy GA restricted to fewer genes or tuned by cross-validation to reduce overfitting.

---

## Reproducibility

All code is in NumPy and seeded. Main files: `diabetes_ga_bpn.py` (network, GA), `fuzzy_layer.py`, `augment.py` (Gaussian generator and SMOTE), `features.py`, `hybrid_experiment.py` (Study 1), `accuracy_experiment.py` (Study 2), `app.py` (application). Generated results are in `results_hybrid.md` and `results_accuracy.md`. Repository: [link to be added].

**Suggested figures** (files in `figs/`): Fig. 1 system architecture (draw from Section IV); Fig. 2 `fig_fuzzy_system.png` (membership functions, learned rule weights, GA convergence); Fig. 3 `fig_accuracy_models.png` (accuracy per model); Fig. 4 `fig_hybrid_comparison.png`; Fig. 5 `fig_risk_levels.png`; Fig. 6 `fig_synthetic_realism.png`; Fig. 7 `fig_ga_feature_selection.png`.

---

## References

[1] J. W. Smith, J. E. Everhart, W. C. Dickson, W. C. Knowler and R. S. Johannes, "Using the ADAP learning algorithm to forecast the onset of diabetes mellitus," in *Proc. Symp. Computer Applications in Medical Care*, 1988, pp. 261–265.

[2] D. E. Rumelhart, G. E. Hinton and R. J. Williams, "Learning representations by back-propagating errors," *Nature*, vol. 323, pp. 533–536, 1986.

[3] J. H. Holland, *Adaptation in Natural and Artificial Systems*. Univ. of Michigan Press, 1975.

[4] D. E. Goldberg, *Genetic Algorithms in Search, Optimization and Machine Learning*. Addison-Wesley, 1989.

[5] D. J. Montana and L. Davis, "Training feedforward neural networks using genetic algorithms," in *Proc. IJCAI*, 1989, pp. 762–767.

[6] L. A. Zadeh, "Fuzzy sets," *Information and Control*, vol. 8, no. 3, pp. 338–353, 1965.

[7] T. Takagi and M. Sugeno, "Fuzzy identification of systems and its applications to modeling and control," *IEEE Trans. Systems, Man, and Cybernetics*, vol. 15, no. 1, pp. 116–132, 1985.

[8] N. V. Chawla, K. W. Bowyer, L. O. Hall and W. P. Kegelmeyer, "SMOTE: Synthetic minority over-sampling technique," *J. Artificial Intelligence Research*, vol. 16, pp. 321–357, 2002.

[9] C. Nadeau and Y. Bengio, "Inference for the generalization error," *Machine Learning*, vol. 52, pp. 239–281, 2003.

[10] R. Kohavi and G. H. John, "Wrappers for feature subset selection," *Artificial Intelligence*, vol. 97, pp. 273–324, 1997.

[11] J. Yang and V. Honavar, "Feature subset selection using a genetic algorithm," *IEEE Intelligent Systems*, vol. 13, no. 2, pp. 44–49, 1998.

[12] G. C. Cawley and N. L. C. Talbot, "On over-fitting in model selection and subsequent selection bias in performance evaluation," *J. Machine Learning Research*, vol. 11, pp. 2079–2107, 2010.

[13] S. Rajasekaran and G. A. Vijayalakshmi Pai, *Neural Networks, Fuzzy Logic and Genetic Algorithms: Synthesis and Applications*. PHI Learning.

[14] T. G. Dietterich, "Approximate statistical tests for comparing supervised classification learning algorithms," *Neural Computation*, vol. 10, no. 7, pp. 1895–1923, 1998.

[15] World Health Organization and International Diabetes Federation, *Definition and Diagnosis of Diabetes Mellitus and Intermediate Hyperglycaemia: Report of a WHO/IDF Consultation*, 2006.
