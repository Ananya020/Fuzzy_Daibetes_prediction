"""
Fuzzy inference layer for the diabetes project (numpy only).

WHERE IT SITS (late fusion, after the neural network):

    8 raw features --> scaler --> BPN / GA-BPN ---------- p_nn ----------\
         |                                                               +--> fused score
         +--> Glucose, BMI, Age, DPF --> fuzzy rules --> p_fuzzy -------/
                                                           (clinical knowledge)
    fused = (1 - alpha) * p_nn + alpha * p_fuzzy
    fused --> fuzzy OUTPUT sets --> Low / Moderate / High / Very high  (+ degrees)

Why late fusion: the neural network and its training are untouched (the existing
GA-BPN pipeline is preserved), the fuzzy part can be switched off (alpha = 0),
and each component can be ablated separately.

Fuzzy system type: zero-order Sugeno.
    rule r:  IF Glucose is High AND BMI is High THEN risk = 0.95
    strength_r = min(memberships of the IF-part)          (AND = min)
    p_fuzzy    = sum(w_r * strength_r * c_r) / sum(w_r * strength_r)
where c_r is the rule's risk value and w_r in [0,1] is a rule WEIGHT (importance).

What the GA can tune here (23 genes, each in [0,1]):
    14 rule weights, 4 shifts + 4 stretches of the membership partitions
    (one shift/stretch per input), and the fusion weight alpha.
"""
import numpy as np

INF = 1e9
PRIOR = 0.35                      # fallback risk if no rule fires (~class rate)

# ---------------------------------------------------------------- membership
def trap(x, a, b, c, d):
    """Trapezoid: 0 below a, rises to 1 on [b, c], falls to 0 at d.
    a == b (or c == d) makes a vertical edge, i.e. a 'shoulder' set."""
    x = np.asarray(x, dtype=float)
    left = np.where(b > a, (x - a) / (b - a + 1e-12), (x >= b).astype(float))
    right = np.where(d > c, (d - x) / (d - c + 1e-12), (x <= c).astype(float))
    return np.clip(np.minimum(left, right), 0.0, 1.0)


# input name -> (column in the 8-feature vector, reference centre, sets)
# Sets are (a, b, c, d) trapezoids in raw units. Anchors follow clinical
# cut-offs: 2-h OGTT glucose (WHO: <140 normal, 140-199 impaired, >=200
# diabetic), BMI (WHO: <25, 25-30, >=30), age bands; the Pima data are mostly
# overweight, so the BMI 'High' set starts at 31.
INPUTS = {
    'Glucose': (1, 127.0, {
        'Low':    (-INF, -INF, 95, 120),
        'Medium': (100, 127, 127, 155),
        'High':   (140, 170, INF, INF)}),
    'BMI': (5, 30.0, {
        'Low':    (-INF, -INF, 24, 28),
        'Medium': (25, 30, 30, 35),
        'High':   (31, 36, INF, INF)}),
    'Age': (7, 40.0, {
        'Young':  (-INF, -INF, 27, 35),
        'Middle': (30, 40, 40, 52),
        'Old':    (45, 55, INF, INF)}),
    'DPF': (6, 0.5, {
        'Low':    (-INF, -INF, 0.25, 0.5),
        'High':   (0.35, 0.8, INF, INF)}),
}
INPUT_NAMES = list(INPUTS)
MAX_SHIFT = np.array([20.0, 4.0, 8.0, 0.15])    # max |shift| per input (raw units)

# (IF-part as [(input, set), ...], risk value c_r)
RULES = [
    ([('Glucose', 'High'), ('BMI', 'High')],                       0.95),
    ([('Glucose', 'High')],                                        0.80),
    ([('Glucose', 'High'), ('Age', 'Middle')],                     0.90),
    ([('Glucose', 'High'), ('Age', 'Old')],                        0.90),
    ([('Glucose', 'Medium'), ('BMI', 'High')],                     0.70),
    ([('Glucose', 'Medium'), ('Age', 'Middle')],                   0.55),
    ([('Glucose', 'Medium'), ('Age', 'Old')],                      0.65),
    ([('Glucose', 'Medium'), ('DPF', 'High')],                     0.60),
    ([('Glucose', 'Medium'), ('BMI', 'Low')],                      0.25),
    ([('Glucose', 'Medium'), ('BMI', 'Medium'), ('Age', 'Young')], 0.30),
    ([('Glucose', 'Low'), ('BMI', 'Low')],                         0.05),
    ([('Glucose', 'Low'), ('Age', 'Young')],                       0.08),
    ([('Glucose', 'Low'), ('BMI', 'High')],                        0.30),
    ([('Glucose', 'Low'), ('DPF', 'High')],                        0.30),
]
N_RULES = len(RULES)
N_GENES = N_RULES + 4 + 4 + 1
ALPHA_MAX = 0.8                   # the neural net keeps at least 20 % of the say


def rule_text(r):
    cond = ' AND '.join(f'{i} is {s}' for i, s in RULES[r][0])
    return f'IF {cond} THEN risk = {RULES[r][1]:.2f}'


# ---------------------------------------------------------------- the system
class FuzzySystem:
    """Zero-order Sugeno system. Defaults = expert setting (no tuning)."""

    def __init__(self, rule_w=None, shift=None, stretch=None):
        self.rule_w = np.ones(N_RULES) if rule_w is None else np.asarray(rule_w)
        self.shift = np.zeros(4) if shift is None else np.asarray(shift)
        self.stretch = np.ones(4) if stretch is None else np.asarray(stretch)

    def memberships(self, X):
        """X: raw (n, 8). Returns {input: {set: (n,) array}}."""
        out = {}
        for k, name in enumerate(INPUT_NAMES):
            col, ref, sets = INPUTS[name]
            # shift the partition, then stretch it around its reference centre
            u = ref + (X[:, col] - self.shift[k] - ref) / self.stretch[k]
            out[name] = {s: trap(u, *p) for s, p in sets.items()}
        return out

    def strengths(self, X):
        """Rule firing strengths (n, N_RULES), AND = min."""
        mem = self.memberships(X)
        cols = [np.min([mem[i][s] for i, s in cond], axis=0)
                for cond, _ in RULES]
        return np.stack(cols, axis=1)

    def score(self, X):
        """Sugeno output: weighted average of rule risk values, in [0,1]."""
        f = self.strengths(X) * self.rule_w
        c = np.array([c for _, c in RULES])
        den = f.sum(1)
        return np.where(den > 1e-6, (f * c).sum(1) / np.maximum(den, 1e-12),
                        PRIOR)

    def explain(self, x, top=3):
        """Most active rules for ONE patient (raw 8-vector)."""
        f = (self.strengths(x.reshape(1, -1)) * self.rule_w)[0]
        order = np.argsort(-f)[:top]
        return [(rule_text(r), float(f[r])) for r in order if f[r] > 1e-3]


def decode(genes):
    """Chromosome in [0,1]^23 -> (FuzzySystem, alpha)."""
    g = np.clip(np.asarray(genes, dtype=float), 0.0, 1.0)
    rule_w = g[:N_RULES]
    shift = (g[N_RULES:N_RULES + 4] * 2 - 1) * MAX_SHIFT
    stretch = 0.75 + 0.5 * g[N_RULES + 4:N_RULES + 8]
    alpha = ALPHA_MAX * g[-1]
    return FuzzySystem(rule_w, shift, stretch), alpha


def default_genes(alpha=0.0):
    """Gene vector that reproduces the untuned expert system."""
    g = np.concatenate([np.ones(N_RULES), np.full(4, 0.5), np.full(4, 0.5),
                        [alpha / ALPHA_MAX]])
    return g


def fuse(p_nn, p_fz, alpha):
    return (1.0 - alpha) * np.ravel(p_nn) + alpha * np.ravel(p_fz)


# ---------------------------------------------------- linguistic risk levels
LEVELS = {'Low':       (-INF, -INF, 0.20, 0.35),
          'Moderate':  (0.20, 0.40, 0.40, 0.60),
          'High':      (0.40, 0.60, 0.70, 0.85),
          'Very high': (0.70, 0.85, INF, INF)}


def risk_levels(p):
    """Fuzzy output sets over the fused score.
    Returns (labels, memberships dict). Ties go to the higher level."""
    p = np.ravel(p)
    names = list(LEVELS)
    M = np.stack([trap(p, *LEVELS[n]) for n in names], axis=1)
    idx = M.shape[1] - 1 - np.argmax(M[:, ::-1], axis=1)
    return np.array(names)[idx], {n: M[:, i] for i, n in enumerate(names)}


# ---------------------------------------------------------------- GA (generic)
def ga_maximize(fit, n_genes, rng, seed_ind=None, pop=40, gens=40, pc=0.8,
                pm=0.15, sigma=0.12, elite=2, k=3):
    """Same GA as run_ga() in diabetes_ga_bpn.py (tournament selection,
    uniform crossover, Gaussian mutation, elitism) but for any fitness
    function on genes in [0,1]. One individual may be seeded with the
    untuned system, so the GA can never end below it on the fitness data."""
    def tournament(P, f):
        idx = rng.integers(0, len(P), k)
        return P[idx[np.argmax(f[idx])]]

    P = rng.random((pop, n_genes))
    if seed_ind is not None:
        P[0] = seed_ind
    hist = []
    for _ in range(gens):
        f = np.array([fit(g) for g in P])
        hist.append(f.max())
        new = [P[i].copy() for i in np.argsort(-f)[:elite]]
        while len(new) < pop:
            p1, p2 = tournament(P, f), tournament(P, f)
            if rng.random() < pc:
                child = np.where(rng.random(n_genes) < 0.5, p1, p2)
            else:
                child = p1.copy()
            m = rng.random(n_genes) < pm
            child = np.clip(child + m * rng.normal(0, sigma, n_genes), 0, 1)
            new.append(child)
        P = np.array(new)
    f = np.array([fit(g) for g in P])
    return P[np.argmax(f)], hist
