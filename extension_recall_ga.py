"""
Extension for the GA-BPN diabetes project.
Run from the same folder as diabetes_ga_bpn.py:  python extension_recall_ga.py

Adds:
  1. GA fitness based on F1 (a non-differentiable metric backprop cannot optimize)
  2. Decision threshold chosen on the VALIDATION set (never the test set)
  3. Convergence metric: epochs needed to reach validation loss < TARGET
  4. Paired t-tests across seeds
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from diabetes_ga_bpn import BPN, prepare, mse

N_SEEDS = 10
TARGET = 0.45          # validation-loss target for the convergence metric
THRESHOLDS = np.arange(0.20, 0.71, 0.05)


# ------------------------------------------------------------ helpers
def fast_f1(p, y, t=0.5):
    pred = p.ravel() >= t
    tp = np.sum(pred & (y == 1))
    fp = np.sum(pred & (y == 0))
    fn = np.sum(~pred & (y == 1))
    return 2 * tp / (2 * tp + fp + fn + 1e-9)


def fit_mse(net, X, y):                 # original fitness
    return 1.0 / (1.0 + mse(net.predict_proba(X), y.reshape(-1, 1)))


def fit_f1(net, X, y):
    # F1 plus a small MSE term: F1 alone is flat/stepwise early on,
    # the MSE term gives the GA a smooth signal to climb from.
    p = net.predict_proba(X)
    return fast_f1(p, y) + 0.1 / (1.0 + mse(p, y.reshape(-1, 1)))


def run_ga_custom(net, X, y, fit_fn, pop=40, gens=40, pc=0.8, pm=0.1,
                  sigma=0.3, elite=2, k=3, rng=None):
    n_w = net.n_w

    def fitness(w):
        net.set_weights(w)
        return fit_fn(net, X, y)

    def tournament(P, f):
        idx = rng.integers(0, len(P), k)
        return P[idx[np.argmax(f[idx])]]

    P = rng.normal(0, 0.5, (pop, n_w))
    for _ in range(gens):
        f = np.array([fitness(w) for w in P])
        order = np.argsort(-f)
        new = [P[i].copy() for i in order[:elite]]
        while len(new) < pop:
            p1, p2 = tournament(P, f), tournament(P, f)
            if rng.random() < pc:
                child = np.where(rng.random(n_w) < 0.5, p1, p2)
            else:
                child = p1.copy()
            m = rng.random(n_w) < pm
            new.append(child + m * rng.normal(0, sigma, n_w))
        P = np.array(new)
    f = np.array([fitness(w) for w in P])
    return P[np.argmax(f)]


def best_threshold(net, Xva, yva):
    p = net.predict_proba(Xva)
    scores = [fast_f1(p, yva, t) for t in THRESHOLDS]
    return THRESHOLDS[int(np.argmax(scores))]


def evaluate(net, X, y, t):
    pred = (net.predict_proba(X).ravel() >= t).astype(int)
    tp = np.sum((pred == 1) & (y == 1))
    fp = np.sum((pred == 1) & (y == 0))
    fn = np.sum((pred == 0) & (y == 1))
    acc = np.mean(pred == y)
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return np.array([acc, prec, rec, f1])


def epochs_to(hist, target=TARGET):
    idx = np.where(np.array(hist['val']) < target)[0]
    return float(idx[0]) if len(idx) else np.nan


# ------------------------------------------------------------ experiment
def run_seed(seed):
    rng = np.random.default_rng(seed)
    Xtr, Xva, Xte, ytr, yva, yte, _, _ = prepare(seed)

    base = BPN(rng=rng)
    h_base = base.train(Xtr, ytr, Xva, yva, 200)

    ga_mse = BPN(rng=rng)
    ga_mse.set_weights(run_ga_custom(ga_mse, Xtr, ytr, fit_mse, rng=rng))
    h_mse = ga_mse.train(Xtr, ytr, Xva, yva, 100)

    tmp = BPN(rng=rng)
    w_f1 = run_ga_custom(tmp, Xtr, ytr, fit_f1, rng=rng)
    ga_only = BPN(w=w_f1)                    # GA result, no fine-tuning
    ga_f1 = BPN(w=w_f1)
    h_f1 = ga_f1.train(Xtr, ytr, Xva, yva, 100)

    models = {'Plain BPN': base, 'GA-BPN (MSE fitness)': ga_mse,
              'GA only (F1 fitness)': ga_only, 'GA-BPN (F1 fitness)': ga_f1}
    out = {}
    for name, net in models.items():
        t = best_threshold(net, Xva, yva)
        out[name + ' @0.5'] = evaluate(net, Xte, yte, 0.5)
        out[name + ' @tuned'] = evaluate(net, Xte, yte, t)
    conv = {'Plain BPN': epochs_to(h_base), 'GA-BPN (MSE fitness)': epochs_to(h_mse),
            'GA-BPN (F1 fitness)': epochs_to(h_f1)}
    return out, conv


if __name__ == '__main__':
    results, convs = [], []
    for s in range(N_SEEDS):
        r, c = run_seed(s)
        results.append(r)
        convs.append(c)
        print(f'seed {s} done')

    names = list(results[0].keys())
    data = {n: np.array([r[n] for r in results]) for n in names}   # (seeds, 4)

    print(f'\nTest results, mean +/- std over {N_SEEDS} seeds')
    print(f'{"Model":<36}{"Acc":>14}{"Prec":>14}{"Recall":>14}{"F1":>14}')
    for n in names:
        m, sd = data[n].mean(0), data[n].std(0)
        print(f'{n:<36}' + ''.join(f'{m[i]:>8.3f}+/-{sd[i]:.2f}' for i in range(4)))

    print(f'\nConvergence: epochs to reach validation loss < {TARGET}')
    for n in convs[0]:
        v = np.array([c[n] for c in convs])
        print(f'{n:<28} mean {np.nanmean(v):6.1f} epochs, '
              f'reached in {np.sum(~np.isnan(v))}/{N_SEEDS} seeds')

    print('\nPaired t-tests vs Plain BPN (same threshold rule)')
    for thr in ('@0.5', '@tuned'):
        a, b = data['Plain BPN ' + thr], data['GA-BPN (F1 fitness) ' + thr]
        for i, mname in ((0, 'Accuracy'), (2, 'Recall'), (3, 'F1')):
            _, p = ttest_rel(b[:, i], a[:, i])
            print(f'  GA-BPN(F1) vs Plain {thr:<7} {mname:<9} '
                  f'diff {np.mean(b[:, i] - a[:, i]):+.3f}  p = {p:.3f}')

    # plot: recall and F1 for every variant
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for a, i, title in zip(ax, (2, 3), ('Recall', 'F1')):
        means = [data[n][:, i].mean() for n in names]
        stds = [data[n][:, i].std() for n in names]
        a.barh(range(len(names)), means, xerr=stds, color='tab:blue')
        a.set_yticks(range(len(names)))
        a.set_yticklabels(names, fontsize=8)
        a.set(title=f'{title} on test set (mean over {N_SEEDS} seeds)', xlim=(0, 1))
        a.invert_yaxis()
    plt.tight_layout()
    plt.savefig('results_ext.png', dpi=150)
    print('\nSaved results_ext.png')