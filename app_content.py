"""Explanatory text and small helpers for the tabs of app.py.

Results/Discussion text is NOT stored here: app.py reads it from
REPORT_SECTION_results_discussion.md, results_accuracy.md and results_hybrid.md,
so the app always shows the same numbers as the report and the experiments.
"""
import os
import re

# ------------------------------------------------------------------ helpers
def read_text(path):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return None


def split_sections(text, marker):
    """Split markdown on lines starting with `marker` (e.g. '## ' or '### ').
    Returns an ordered list of (heading, body)."""
    if not text:
        return []
    parts = re.split(rf"(?m)^{re.escape(marker)}", text)
    out = []
    for p in parts[1:]:
        head, _, body = p.partition("\n")
        out.append((head.strip(), body.strip()))
    return out


# ------------------------------------------------------------------ overview
OVERVIEW_MD = """
### The problem
Diabetes is detected from a handful of clinical measurements. Given eight values for a
patient (pregnancies, glucose, blood pressure, skin thickness, insulin, BMI, family
history score and age), can a model estimate whether the patient has diabetes, and can it
explain *why*?

### What this project builds
A **hybrid intelligent system** that combines three soft-computing techniques:

| Technique | Role in the system | Syllabus unit |
|---|---|---|
| **Neural network** (back-propagation) | Learns the relation between the 8 inputs and the outcome | Unit 1 |
| **Genetic algorithm** | Finds good starting weights; selects features and hyperparameters | Unit 4, Unit 5 (GA-based weight determination) |
| **Fuzzy logic** | Clinical IF-THEN rules; converts the score into Low / Moderate / High / Very high risk | Unit 3 |

### Objectives
1. Build a back-propagation network from scratch in NumPy (every step explainable).
2. Use a genetic algorithm, also written from scratch, to improve the network.
3. Add a fuzzy layer that is interpretable and can be switched off for ablation.
4. Test synthetic-data augmentation (Gaussian and SMOTE) and engineered features.
5. Compare all variants honestly, with proper train / validation / test separation and
   statistical tests, and say which components actually help.

### How the pieces fit together
"""

PIPELINE_DOT = """
digraph G {
  rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#7657D6", fillcolor="#F3EFFF", fontcolor="#20233A"];
  edge [color="#85899E", fontname="Helvetica", fontsize=10];
  raw   [label="8 clinical\\ninputs"];
  prep  [label="Impute zeros\\n(train median)\\n+ scale"];
  nn    [label="Neural network\\n8-12-1 (BPN)", fillcolor="#DFF5EB"];
  ga    [label="Genetic algorithm\\n(start weights /\\nfeatures / hyperparameters)", fillcolor="#FFF4D6"];
  fz    [label="Fuzzy rules\\n(Glucose, BMI,\\nAge, DPF)", fillcolor="#FFE3E3"];
  fuse  [label="Fusion\\n(1-a)*NN + a*fuzzy"];
  lvl   [label="Fuzzy risk level\\nLow / Moderate /\\nHigh / Very high"];
  raw -> prep -> nn;
  ga -> nn [style=dashed, label="initialises"];
  raw -> fz;
  nn -> fuse [label="p_nn"];
  fz -> fuse [label="p_fuzzy"];
  fuse -> lvl;
}
"""

OVERVIEW_TAIL_MD = """
**Why the fuzzy layer sits after the network (late fusion):** the network and its training
stay untouched, the fuzzy part can be switched off (a = 0) so each component can be ablated
cleanly, and the rules give a human-readable explanation of every prediction.

### Headline findings (details in the *Results* and *Discussion* tabs)
* All models, from plain logistic regression to the full hybrid, reach about **0.75-0.77
  accuracy and 0.83-0.84 AUC**. No tested change improves accuracy significantly.
* The hybrid's real contributions are **interpretability** (fuzzy risk levels and rule
  explanations) and an **adjustable recall/precision trade-off** (threshold or SMOTE).
* On this small, noisy dataset accuracy is limited by the data, not by model sophistication.

> This is an educational project, not a medical device or a clinically validated tool.
"""

# ------------------------------------------------------------------ data
FEATURES = [
    ("Pregnancies", "Number of times pregnant", "count"),
    ("Glucose", "Plasma glucose 2 h into an oral glucose tolerance test", "mg/dL"),
    ("BloodPressure", "Diastolic blood pressure", "mm Hg"),
    ("SkinThickness", "Triceps skin-fold thickness", "mm"),
    ("Insulin", "2-hour serum insulin", "mu U/mL"),
    ("BMI", "Body mass index", "kg/m²"),
    ("DiabetesPedigreeFunction", "Score of diabetes family history", "unitless"),
    ("Age", "Age", "years"),
]

DATA_MD = """
### Dataset
The **Pima Indians Diabetes** dataset (National Institute of Diabetes and Digestive and
Kidney Diseases; Smith et al., 1988): **768 women**, aged 21 or older, of Pima Indian
heritage, with 8 measurements and a binary outcome (diabetes within five years).
Class balance is **500 negative / 268 positive** (34.9 % positive), so plain accuracy
rewards a model that favours the majority class; this is why recall, precision, F1 and AUC
are reported as well.

### Preprocessing
1. **Missing values.** In five columns (Glucose, BloodPressure, SkinThickness, Insulin,
   BMI) a value of 0 is physiologically impossible and means "not measured". These zeros
   are treated as missing and replaced by the **median of the training rows**.
   (Insulin is missing for almost half of the patients, which creates a spike at the median.)
2. **Splitting.** Stratified splits keep the 34.9 % positive rate in every part.
3. **Scaling.** Every feature is standardised (mean 0, std 1) using training-row statistics.
4. **No leakage.** Medians, scaler and any synthetic-data generator are computed from
   training rows only. Validation and test rows are always original patients.

### Data-handling issue found and fixed
`diabetes.csv` has **no header row**. The original loader used the default `header=0`,
which silently treated the first patient as column names and dropped that row (767 rows
instead of 768). The loader now detects a header explicitly. Numbers from before the fix
differ very slightly.
"""

# ------------------------------------------------------------------ methods
NN_MD = """
### Architecture: 8-12-1 feed-forward network (Unit 1)
* **Input layer:** 8 standardised features.
* **Hidden layer:** 12 neurons with sigmoid activation.
* **Output neuron:** 1 sigmoid unit = predicted probability of diabetes.
* **Weights:** 8x12 + 12 + 12 + 1 = **121** numbers, stored as one flat vector.
  This vector is exactly what the genetic algorithm later treats as a chromosome.

### Forward pass
```
hidden = sigmoid(X · W1 + b1)
output = sigmoid(hidden · W2 + b2)
```

### Learning: back-propagation with momentum
* **Loss:** binary cross-entropy. With a sigmoid output its gradient is simply
  `delta_out = (output - target) / N`.
* **Hidden delta (error back-propagated):** `delta_hid = (delta_out · W2ᵀ) * h * (1 - h)`.
* **Update:** `velocity = momentum * velocity - lr * gradient;  weight += velocity`
* **Settings:** full-batch gradient descent, learning rate 0.1, momentum 0.9,
  200 epochs, initial weights drawn from N(0, 0.5).

### Output
The probability is compared with a **threshold** (0.5 by default). The threshold can be
moved on validation data to trade precision against recall.
"""

GA_MD = """
### Genetic algorithm (Units 4 and 5), written from scratch
A GA searches by evolving a population of candidate solutions (**chromosomes**) using
selection, crossover and mutation. It needs no gradient, so it can optimise anything we can
score.

| Element | Choice |
|---|---|
| Encoding | Real-coded chromosome |
| Selection | Tournament (k = 3) |
| Crossover | Uniform, probability 0.8 |
| Mutation | Gaussian noise on each gene with probability 0.1 (sigma 0.3) |
| Elitism | The 2 best individuals copied unchanged |
| Population / generations | 40 / 40 (= 1,600 fitness evaluations) |

### Use 1: GA finds the starting weights (GA-based weight determination, Unit 5)
* **Chromosome:** the 121 network weights.
* **Fitness:** `1 / (1 + MSE)` on the training set (higher is better).
* **Then:** back-propagation fine-tunes for 100 epochs. The GA gives a good *global*
  starting region; back-propagation finishes the *local* search.

### Use 2: GA selects features and hyperparameters
* **Chromosome (19 genes in [0,1]):** 15 feature on/off genes (8 original + 7 engineered
  features), then hidden units (4-24), learning rate (0.01-0.5, log scale), momentum
  (0-0.95) and epochs (50-300).
* **Fitness:** mean accuracy of 3-fold cross-validation on the inner-training rows, minus a
  small penalty per selected feature (0.003), plus a tiny smooth log-loss tie-breaker.
  One individual is seeded with the default setup so the GA cannot end below it.
* Population 16, generations 12.

### Use 3 (first study): GA tunes the fuzzy layer
23 genes: 14 rule weights, 4 shifts and 4 stretches of the membership partitions (one per
input) and the fusion weight a. Fitness = `1/(1 + validation log-loss)`.
"""

FUZZY_MD = """
### Fuzzy inference layer (Unit 3)
A **zero-order Sugeno** system on four clinically meaningful inputs:
**Glucose, BMI, Age and Diabetes Pedigree Function (DPF)**.

1. **Fuzzification.** Each input gets overlapping trapezoidal membership functions
   (e.g. Glucose is Low / Medium / High). Anchors follow clinical cut-offs: 2-hour glucose
   (WHO: < 140 normal, 140-199 impaired, >= 200 diabetic) and BMI (< 25, 25-30, >= 30).
   A value can belong to two sets at once, e.g. glucose 130 is partly Medium and partly High.
2. **Rules.** 14 IF-THEN rules, each with a risk value, e.g.
   *IF Glucose is High AND BMI is High THEN risk = 0.95*. AND is the minimum of the
   memberships.
3. **Inference.** `p_fuzzy = sum(w * strength * risk) / sum(w * strength)`, where `w` is a
   rule weight (1 for the expert setting).
4. **Fusion.** `final = (1 - a) * p_network + a * p_fuzzy`. In the expert variant `a` is
   chosen on the validation set; a = 0 means "no fuzzy layer".
5. **Defuzzified output as linguistic levels.** The fused score is mapped by overlapping
   output sets to **Low / Moderate / High / Very high**, and the most active rules are
   reported as the explanation.

The membership functions and rules below are drawn from the live code, so they are exactly
what the model uses.
"""

AUG_MD = """
### Synthetic data and feature engineering

**Gaussian augmentation.** One multivariate Gaussian per class is fitted on the training
rows (mean = typical level of each feature, covariance = relationships between features;
skewed features are modelled on a log scale). New rows are drawn, integer features are
rounded and every value is clipped to the range seen in training. The class ratio is
preserved. It re-samples what is already known, so it adds no new information.

**SMOTE.** For each new row, pick a random minority (diabetic) row, pick one of its 5
nearest minority neighbours and place the new row at a random point on the line between
them. SMOTE **balances the classes to 50:50**, which pushes predicted probabilities up and
changes the precision/recall balance.

**Original vs synthetic.** Synthetic rows are only ever added to the *training* part. The
validation and test sets contain only real patients, and synthetic rows are generated from
training rows only.

**Engineered features** (computed row by row, no leakage): Glucose x BMI, Age x BMI,
Age x Glucose, log(DPF), Glucose², BMI², Pregnancies / Age.
"""

REFERENCES_MD = """
1. Smith, J. W., Everhart, J. E., Dickson, W. C., Knowler, W. C., Johannes, R. S. (1988).
   *Using the ADAP learning algorithm to forecast the onset of diabetes mellitus.*
   Proc. Symposium on Computer Applications in Medical Care, 261-265.
2. Pima Indians Diabetes Database, UCI Machine Learning Repository / NIDDK.
3. Goldberg, D. E. (1989). *Genetic Algorithms in Search, Optimization and Machine Learning.*
   Addison-Wesley.
4. Rajasekaran, S., Vijayalakshmi Pai, G. A. *Neural Networks, Fuzzy Logic and Genetic
   Algorithms: Synthesis and Applications.* PHI Learning.
5. Chawla, N. V., Bowyer, K. W., Hall, L. O., Kegelmeyer, W. P. (2002). SMOTE: Synthetic
   Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, 16, 321-357.
6. Nadeau, C., Bengio, Y. (2003). Inference for the generalization error.
   *Machine Learning*, 52, 239-281. (Corrected resampled t-test.)
7. World Health Organization: classification of BMI and of glucose tolerance (cut-offs used
   to anchor the fuzzy membership functions).
"""
