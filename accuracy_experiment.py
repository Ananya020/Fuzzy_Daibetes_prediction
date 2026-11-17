"""
Accuracy-focused study (numpy only).   Run:  python accuracy_experiment.py

Questions answered
  * Does tuning the threshold for ACCURACY (on validation) help?
  * Gaussian augmentation vs SMOTE vs none.
  * Do engineered features (Glucose*BMI, Age*BMI, Age*Glucose, log DPF, ...) help?
  * GA used for FEATURE SELECTION + NETWORK HYPERPARAMETERS (not fuzzy tuning).
  * Which is the simplest model that is statistically as good as the best?

Leakage control (nothing is tuned on a test row):
  1. 20 % of the rows are set aside as a FINAL HOLDOUT, never used in any
     tuning or model selection. It is evaluated once, for the chosen model.
  2. The other 80 % ("DEV", 614 rows) go through stratified 5-fold CV,
     repeated N_REPEATS times (new fold assignment each repeat).
     In every outer split the TEST fold is only used for scoring.
  3. Inside every outer split, the outer-training rows are split 80/20 into
     inner-train / validation. Models are fitted on inner-train (plus
     synthetic rows, which are generated from inner-train only), and
     validation is used only for the threshold (and the fuzzy alpha).
  4. The GA's fitness uses 3-fold CV on inner-train rows only, so validation
     rows are untouched by the GA.
  5. Medians (imputation) and the feature scaler come from the outer-training
     rows (never the test fold).
"""
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from diabetes_ga_bpn import BPN, run_ga, bce, sigmoid
from hybrid_experiment import load_xy, stratified_split, auc
import augment
import features
import fuzzy_layer as fz

K_OUTER, N_REPEATS, HOLDOUT_SEED = 5, 3, 12345
THR_GRID = np.round(np.arange(0.30, 0.701, 0.02), 2)
GA_POP, GA_GENS = 16, 12
NP = len(features.POOL_NAMES)                     # 15 candidate features
DEF_HP = dict(hid=12, lr=0.1, mom=0.9, ep=200)    # the project's usual settings
ALPHA_GRID = np.round(np.arange(0.0, 0.81, 0.1), 2)
METRICS = ['Acc', 'Prec', 'Rec', 'F1', 'AUC']

# (name, model, feature set, augmentation, complexity = number of components)
SPECS = [
    ('LogReg',          'logreg', 'base', 'none',  1),
    ('LogReg+Eng',      'logreg', 'all',  'none',  2),
    ('BPN',             'bpn',    'base', 'none',  3),
    ('BPN+Fuzzy',       'bpnfz',  'base', 'none',  4),
    ('BPN+Gauss',       'bpn',    'base', 'gauss', 4),
    ('BPN+SMOTE',       'bpn',    'base', 'smote', 4),
    ('BPN+Eng',         'bpn',    'all',  'none',  4),
    ('BPN+Eng+Gauss',   'bpn',    'all',  'gauss', 5),
    ('BPN+Eng+SMOTE',   'bpn',    'all',  'smote', 5),
    ('GA-BPN',          'gabpn',  'base', 'none',  5),
    ('GA-Opt',          'gaopt',  'pool', 'none',  6),
    ('GA-Opt+SMOTE',    'gaopt',  'pool', 'smote', 7),
]
NAMES = [s[0] for s in SPECS]
COMPLEXITY = {s[0]: s[4] for s in SPECS}


# ------------------------------------------------------------------ helpers
def stratified_kfold(y, k, rng):
    folds = [[] for _ in range(k)]
    for c in (0, 1):
        for i, j in enumerate(rng.permutation(np.where(y == c)[0])):
            folds[i % k].append(j)
    return [np.array(sorted(f)) for f in folds]


def metrics(p, y, t):
    pred = (p >= t).astype(int)
    tp = np.sum((pred == 1) & (y == 1)); fp = np.sum((pred == 1) & (y == 0))
    fn = np.sum((pred == 0) & (y == 1))
    prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9)
    return np.array([np.mean(pred == y), prec, rec,
                     2 * prec * rec / (prec + rec + 1e-9), auc(p, y)])


def acc_threshold(pv, yv):
    """Threshold maximising VALIDATION accuracy; ties -> closest to 0.5."""
    accs = np.array([np.mean((pv >= t) == yv) for t in THR_GRID])
    return THR_GRID[np.lexsort((np.abs(THR_GRID - 0.5), -accs))[0]]


class LogReg:
    """Plain logistic regression by gradient descent (the simplest baseline)."""
    def __init__(self, X, y, lr=0.2, epochs=800, l2=1e-2):
        self.w, self.b = np.zeros(X.shape[1]), 0.0
        for _ in range(epochs):
            g = sigmoid(X @ self.w + self.b) - y
            self.w -= lr * (X.T @ g / len(y) + l2 * self.w)
            self.b -= lr * g.mean()

    def predict_proba(self, X):
        return sigmoid(X @ self.w + self.b)


def train_bpn(Z, y, Zv, yv, hp, seed):
    net = BPN(n_in=Z.shape[1], n_hid=hp['hid'], lr=hp['lr'], mom=hp['mom'],
              rng=np.random.default_rng(seed))
    net.train(Z, y, Zv, yv, hp['ep'])
    return net


# GA chromosome for feature selection + hyperparameters, genes in [0,1]:
#   15 feature-mask genes (>0.5 = keep), hidden units 4..24, lr 0.01..0.5 (log),
#   momentum 0..0.95, epochs 50..300
def decode_opt(g):
    mask = g[:NP] > 0.5
    if not mask.any():
        mask[np.argmax(g[:NP])] = True
    return dict(cols=np.where(mask)[0], hid=int(round(4 + 20 * g[NP])),
                lr=float(10 ** (-2 + 1.7 * g[NP + 1])), mom=float(0.95 * g[NP + 2]),
                ep=int(round(50 + 250 * g[NP + 3])))


SEED_OPT = np.array([1.0] * NP + [0.4, 1 / 1.7, 0.9 / 0.95, 0.6])   # = default setup


# ------------------------------------------------------------------ one split
def evaluate_specs(Xtr, ytr, Xte, yte, seed):
    """Fit every spec on (Xtr, ytr), score on (Xte, yte). Returns dict."""
    med = np.nanmedian(Xtr, 0)
    fill = lambda A: np.where(np.isnan(A), med, A)
    Xtr, Xte = fill(Xtr), fill(Xte)
    E = features.engineer(Xtr)
    mu, sd = E.mean(0), E.std(0)
    Zs = lambda A: (features.engineer(A) - mu) / sd          # (n, 15) scaled

    rng = np.random.default_rng([seed, 0])
    itr, iva, _ = stratified_split(ytr, rng, (0.8, 0.2))
    Xi, yi, Xv, yv = Xtr[itr], ytr[itr], Xtr[iva], ytr[iva]
    cols_of = {'base': np.arange(8), 'all': np.arange(NP)}

    def rows(aug):                                           # augmentation: inner-train only
        r = np.random.default_rng([seed, 1, {'none': 0, 'gauss': 1, 'smote': 2}[aug]])
        if aug == 'gauss':
            Xs, ys = augment.generate(augment.fit_generator(Xi, yi), 1.0, r)
            return np.vstack([Xi, Xs]), np.concatenate([yi, ys])
        if aug == 'smote':
            return augment.smote(Xi, yi, r)
        return Xi, yi

    # --- GA: feature selection + hyperparameters (once per split) ----------
    Zi = Zs(Xi)
    folds = stratified_kfold(yi, 3, np.random.default_rng([seed, 2]))

    def fit(g):
        hp = decode_opt(g)
        accs, lls = [], []
        for f in folds:
            tr = np.setdiff1d(np.arange(len(yi)), f)
            net = train_bpn(Zi[tr][:, hp['cols']], yi[tr], Zi[f][:, hp['cols']],
                            yi[f], hp, 0)
            p = net.predict_proba(Zi[f][:, hp['cols']]).ravel()
            accs.append(np.mean((p >= 0.5) == yi[f]))
            lls.append(bce(p, yi[f]))
        # accuracy, small parsimony pressure, tiny smooth tie-breaker
        return np.mean(accs) - 0.003 * len(hp['cols']) + 0.02 / (1 + np.mean(lls))

    g_best, _ = fz.ga_maximize(fit, NP + 4, np.random.default_rng([seed, 3]),
                               seed_ind=SEED_OPT, pop=GA_POP, gens=GA_GENS)
    hp_opt = decode_opt(g_best)

    # --- fit all models -----------------------------------------------------
    out, probs = {}, {}

    def fit_predict(kind, cols, aug, hp, tag):
        Xr, yr = rows(aug)
        Z, Zv, Zt = Zs(Xr)[:, cols], Zs(Xv)[:, cols], Zs(Xte)[:, cols]
        if kind == 'logreg':
            m = LogReg(Z, yr)
        elif kind == 'bpn':
            m = train_bpn(Z, yr, Zv, yv, hp, seed * 100 + tag)
        else:                                                # 'gabpn' (existing GA, unchanged)
            r = np.random.default_rng([seed, 4, tag])
            m = BPN(n_in=len(cols), rng=r)
            m.set_weights(run_ga(m, Z, yr, rng=r)[0])
            m.train(Z, yr, Zv, yv, 100)
        return m.predict_proba(Zv).ravel(), m.predict_proba(Zt).ravel()

    for tag, (name, kind, fs, aug, _) in enumerate(SPECS):
        if kind == 'bpnfz':
            continue
        if kind == 'gaopt':
            pv, pt = fit_predict('bpn', hp_opt['cols'], aug, hp_opt, tag)
        else:
            pv, pt = fit_predict(kind, cols_of[fs], aug, DEF_HP, tag)
        probs[name] = (pv, pt)
    # BPN + expert fuzzy layer (alpha picked on validation, rules untuned)
    ex = fz.FuzzySystem()
    pv, pt = probs['BPN']
    sv, st = ex.score(Xv), ex.score(Xte)
    a0 = ALPHA_GRID[int(np.argmin([bce(fz.fuse(pv, sv, a), yv) for a in ALPHA_GRID]))]
    probs['BPN+Fuzzy'] = (fz.fuse(pv, sv, a0), fz.fuse(pt, st, a0))

    for name, (pv, pt) in probs.items():
        t = acc_threshold(pv, yv)
        out[name] = {'0.5': metrics(pt, yte, 0.5), 'acc': metrics(pt, yte, t), 'thr': t}
    out['_gaopt'] = hp_opt
    return out


# ------------------------------------------------------------------ statistics
def t_sf(t, df):
    """P(T > t) for Student's t, by numerical integration (no scipy)."""
    x = np.linspace(t, t + 400, 40001)
    c = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)
    return float(np.trapezoid(c * (1 + x ** 2 / df) ** (-(df + 1) / 2), x))


def nb_test(d, n_test, n_train):
    """Nadeau-Bengio corrected resampled paired t-test (accounts for the
    overlap between CV training sets). Returns (mean diff, p two-sided)."""
    d = np.asarray(d, float)
    k, v = len(d), np.var(d, ddof=1)
    if v < 1e-12:
        return float(d.mean()), 1.0
    t = d.mean() / math.sqrt((1 / k + n_test / n_train) * v)
    return float(d.mean()), 2 * t_sf(abs(t), k - 1)


def wilson(acc, n, z=1.96):
    c = (acc + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(acc * (1 - acc) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


# ------------------------------------------------------------------ main
if __name__ == '__main__':
    X, y = load_xy()
    dev, hold, _ = stratified_split(y, np.random.default_rng(HOLDOUT_SEED), (0.8, 0.2))
    Xd, yd, Xh, yh = X[dev], y[dev], X[hold], y[hold]
    print(f'DEV {len(yd)} rows ({yd.mean():.3f} positive), HOLDOUT {len(yh)} rows '
          f'({yh.mean():.3f} positive). Holdout is evaluated once at the end.')

    R = {n: {'0.5': [], 'acc': [], 'thr': []} for n in NAMES}
    GAOPT = []
    n_te = n_tr = 0
    for rep in range(N_REPEATS):
        folds = stratified_kfold(yd, K_OUTER, np.random.default_rng([rep, 100]))
        for fi, f in enumerate(folds):
            tr = np.setdiff1d(np.arange(len(yd)), f)
            res = evaluate_specs(Xd[tr], yd[tr], Xd[f], yd[f], seed=rep * 10 + fi)
            for n in NAMES:
                for k_ in ('0.5', 'acc', 'thr'):
                    R[n][k_].append(res[n][k_])
            GAOPT.append(res['_gaopt'])
            n_te, n_tr = len(f), len(tr)
            print(f'repeat {rep} fold {fi} done', flush=True)
    A = {n: {k_: np.array(R[n][k_]) for k_ in R[n]} for n in NAMES}
    K = len(A[NAMES[0]]['acc'])

    md = [f'# Accuracy study (auto-generated by accuracy_experiment.py)\n',
          f'Stratified {K_OUTER}-fold CV x {N_REPEATS} repeats on {len(yd)} DEV rows '
          f'({K} outer test folds of {n_te} rows); mean ± std over the {K} folds. '
          f'A final holdout of {len(yh)} rows is reported separately.\n']
    for key, title in (('acc', 'threshold tuned for ACCURACY on validation'),
                       ('0.5', 'threshold 0.5')):
        md += [f'### {title}\n', '| Model | ' + ' | '.join(METRICS) + ' | mean thr |',
               '|---|' + '---|' * (len(METRICS) + 1)]
        for n in NAMES:
            a = A[n][key]
            thr = f'{A[n]["thr"].mean():.2f}' if key == 'acc' else '0.50'
            md.append(f'| {n} | ' + ' | '.join(f'{a[:, i].mean():.3f} ± {a[:, i].std():.3f}'
                                                for i in range(5)) + f' | {thr} |')
        md.append('')

    # contrasts (accuracy)
    md += ['### Paired contrasts on accuracy (Nadeau-Bengio corrected t-test)\n',
           '| A vs B | which accuracy | mean diff (A-B) | p |', '|---|---|---|---|']
    contrasts = [('BPN', 'acc', 'BPN', '0.5', 'tuned vs 0.5 threshold (same model)'),
                 ('BPN+Gauss', 'acc', 'BPN', 'acc', None), ('BPN+SMOTE', 'acc', 'BPN', 'acc', None),
                 ('BPN+SMOTE', 'acc', 'BPN+Gauss', 'acc', None),
                 ('BPN+Eng', 'acc', 'BPN', 'acc', None), ('LogReg+Eng', 'acc', 'LogReg', 'acc', None),
                 ('BPN+Fuzzy', 'acc', 'BPN', 'acc', None), ('GA-BPN', 'acc', 'BPN', 'acc', None),
                 ('GA-Opt', 'acc', 'BPN', 'acc', None), ('GA-Opt', 'acc', 'BPN+Eng', 'acc', None),
                 ('GA-Opt+SMOTE', 'acc', 'GA-Opt', 'acc', None), ('BPN', 'acc', 'LogReg', 'acc', None)]
    ncontrasts = 0
    for a_, ak, b_, bk, note in contrasts:
        d, p = nb_test(A[a_][ak][:, 0] - A[b_][bk][:, 0], n_te, n_tr)
        lab = f'{a_} (tuned thr) vs {b_} (0.5 thr)' if note else f'{a_} vs {b_}'
        md.append(f'| {lab} | accuracy | {d:+.4f} | {p:.3f} |')
        ncontrasts += 1
    md.append(f'\n{ncontrasts} contrasts, uncorrected for multiple comparisons.\n')

    # selection: simplest model not significantly worse than the best
    means = {n: A[n]['acc'][:, 0].mean() for n in NAMES}
    best = max(means, key=means.get)
    md += [f'### Model selection (accuracy at validation-tuned threshold)\n',
           f'Best mean CV accuracy: **{best}** ({means[best]:.4f}). Rule: among models whose '
           f'accuracy is not significantly lower than the best (corrected t-test p > 0.05), '
           f'take the one with the fewest components (ties: higher accuracy).\n',
           '| Model | complexity | mean acc | diff vs best | p vs best | candidate |',
           '|---|---|---|---|---|---|']
    cands = []
    for n in NAMES:
        d, p = nb_test(A[n]['acc'][:, 0] - A[best]['acc'][:, 0], n_te, n_tr) if n != best else (0.0, 1.0)
        ok = p > 0.05
        if ok:
            cands.append(n)
        md.append(f'| {n} | {COMPLEXITY[n]} | {means[n]:.4f} | {d:+.4f} | {p:.3f} | {"yes" if ok else "no"} |')
    chosen = sorted(cands, key=lambda n: (COMPLEXITY[n], -means[n]))[0]
    md.append(f'\n**Chosen (simplest defensible): {chosen}.** Failing to reject equality is '
              f'weak evidence of equivalence with {K} correlated folds; the test is '
              f'low-powered, so this rule favours simple models.\n')

    # holdout, evaluated once
    hres = evaluate_specs(Xd, yd, Xh, yh, seed=999)
    md += ['### Final holdout (evaluated once; model chosen from CV above)\n',
           f'{len(yh)} rows never used in tuning or selection. 95% Wilson interval on accuracy.\n',
           '| Model | thr | Acc | 95% CI | Prec | Rec | F1 | AUC |', '|---|---|---|---|---|---|---|---|']
    for n in dict.fromkeys([chosen, best, 'LogReg', 'BPN']):
        m, t = hres[n]['acc'], hres[n]['thr']
        lo, hi = wilson(m[0], len(yh))
        md.append(f'| {n}{" (chosen)" if n == chosen else ""} | {t:.2f} | {m[0]:.3f} | '
                  f'[{lo:.3f}, {hi:.3f}] | {m[1]:.3f} | {m[2]:.3f} | {m[3]:.3f} | {m[4]:.3f} |')
    md.append('')

    # GA selections
    freq = np.zeros(NP)
    for h in GAOPT:
        freq[h['cols']] += 1
    md += ['### What the GA selected (per outer fold)\n', '| Feature | selected in |', '|---|---|']
    for n_, f_ in zip(features.POOL_NAMES, freq):
        md.append(f'| {n_} | {int(f_)}/{len(GAOPT)} folds |')
    md.append(f'\nHidden units {np.mean([h["hid"] for h in GAOPT]):.1f} ± '
              f'{np.std([h["hid"] for h in GAOPT]):.1f}; lr {np.mean([h["lr"] for h in GAOPT]):.3f}; '
              f'momentum {np.mean([h["mom"] for h in GAOPT]):.2f}; epochs '
              f'{np.mean([h["ep"] for h in GAOPT]):.0f}; features kept '
              f'{np.mean([len(h["cols"]) for h in GAOPT]):.1f} of {NP}.\n')

    text = '\n'.join(md)
    print('\n' + text)
    with open('results_accuracy.md', 'w', encoding='utf-8') as f:
        f.write(text)
    with open('results_accuracy.csv', 'w') as f:
        f.write('model,threshold_rule,' + ','.join(f'{m}_mean,{m}_std' for m in METRICS) + '\n')
        for n in NAMES:
            for k_ in ('0.5', 'acc'):
                a = A[n][k_]
                f.write(f'{n},{k_},' + ','.join(f'{a[:, i].mean():.4f},{a[:, i].std():.4f}'
                                                for i in range(5)) + '\n')

    # figures
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    for i, n in enumerate(NAMES):
        for key, off, fill_ in (('0.5', -0.17, 'white'), ('acc', 0.17, '#0072B2')):
            a = A[n][key][:, 0]
            ax.errorbar(a.mean(), i + off, xerr=a.std(), fmt='o', color='#0072B2', ms=6,
                        mfc=fill_, mew=1.6, capsize=2.5)
    ax.set_yticks(range(len(NAMES)))
    ax.set_yticklabels(NAMES, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(f'Accuracy on outer test folds (mean ± std over {K} folds)')
    ax.set_title('Filled = threshold tuned for accuracy on validation, hollow = threshold 0.5',
                 fontsize=10)
    ax.grid(axis='x', color='#e4e4e4', lw=0.6)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    plt.savefig('fig_accuracy_models.png', dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.barh(range(NP), freq / len(GAOPT), color='#009E73')
    ax.set_yticks(range(NP))
    ax.set_yticklabels(features.POOL_NAMES, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel('Fraction of outer folds in which the GA kept the feature')
    ax.set_title('GA feature selection (engineered features start at Glucose*BMI)', fontsize=10)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    plt.savefig('fig_ga_feature_selection.png', dpi=150)
    plt.close()
    print('\nSaved results_accuracy.md/.csv, fig_accuracy_models.png, fig_ga_feature_selection.png')
