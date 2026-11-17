"""
Hybrid GA + Neural Network + Fuzzy Logic experiment (numpy only).
Run:  python hybrid_experiment.py        (about a minute or two)

Configurations compared (all tested on the SAME untouched original test rows):
    1. BPN                         plain backprop network, random start
    2. GA + BPN                    GA finds start weights, backprop fine-tunes
    3. BPN + Fuzzy                 BPN + expert fuzzy layer (alpha picked on validation)
    4. GA + BPN + Fuzzy (hybrid)   GA-BPN + fuzzy layer whose rule weights, membership
                                   shifts/stretches and alpha are tuned by a second GA
  + the remaining cells of the 2x3 grid (NN start x fuzzy mode) for the ablation,
  + 'Fuzzy only' (expert rules, no network) as a reference,
  + every row repeated with synthetic training rows added ('real+synth').

Data separation (per seed, 10 seeds):
    stratified split 70/15/15 of the ORIGINAL rows -> train / validation / test
    medians, scaler and the synthetic-data generator use TRAIN rows only
    synthetic rows are appended to the TRAIN part only; val/test are always original
    validation is used for: threshold choice, alpha choice, fuzzy-GA fitness
    test is used once, for reporting.
"""
import itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from diabetes_ga_bpn import BPN, run_ga, load_data, ZERO_COLS, bce
import augment
import fuzzy_layer as fz

N_SEEDS = 10
SYN_RATIO = 1.0                                   # synthetic rows per real row
THRESHOLDS = np.arange(0.20, 0.71, 0.05)
ALPHA_GRID = np.round(np.arange(0.0, 0.81, 0.1), 2)
METRICS = ['Acc', 'Prec', 'Rec', 'F1', 'AUC', 'LogLoss']
MODES = ('real', 'synth')
NN_NAMES = ('BPN', 'GA-BPN')
FZ_NAMES = ('none', 'expert', 'GA')
# Okabe-Ito colour-blind-safe colours, fixed per configuration
C = {'BPN': '#0072B2', 'GA + BPN': '#E69F00', 'BPN + Fuzzy': '#009E73',
     'GA + BPN + Fuzzy': '#D55E00'}


# ------------------------------------------------------------------ data
def load_xy():
    df = load_data()
    X = df[augment.FEATURES].values.astype(float)
    y = df['Outcome'].values.astype(int)
    for j, name in enumerate(augment.FEATURES):
        if name in ZERO_COLS:
            X[X[:, j] == 0, j] = np.nan               # zeros mean "missing"
    return X, y


def stratified_split(y, rng, fr=(0.70, 0.15)):
    tr, va, te = [], [], []
    for c in (0, 1):
        idx = rng.permutation(np.where(y == c)[0])
        a = int(round(fr[0] * len(idx)))
        b = a + int(round(fr[1] * len(idx)))
        tr += list(idx[:a]); va += list(idx[a:b]); te += list(idx[b:])
    return np.array(tr), np.array(va), np.array(te)


def make_data(seed, X, y):
    tr, va, te = stratified_split(y, np.random.default_rng([seed, 0]))
    med = np.nanmedian(X[tr], axis=0)                 # train-only medians
    fill = lambda A: np.where(np.isnan(A), med, A)
    Xtr, Xva, Xte = fill(X[tr]), fill(X[va]), fill(X[te])
    mean, scale = Xtr.mean(0), Xtr.std(0)             # train-only scaler
    gen = augment.fit_generator(Xtr, y[tr])           # train-only generator
    Xsyn, ysyn = augment.generate(gen, SYN_RATIO,
                                  np.random.default_rng([seed, 1]))
    z = lambda A: (A - mean) / scale
    return dict(Xtr=Xtr, Xva=Xva, Xte=Xte, ytr=y[tr], yva=y[va], yte=y[te],
                Xsyn=Xsyn, ysyn=ysyn, mean=mean, scale=scale, med=med,
                Ztr=z(Xtr), Zva=z(Xva), Zte=z(Xte), Zsyn=z(Xsyn),
                realism=augment.realism_report(Xtr, y[tr], Xsyn, ysyn))


# ------------------------------------------------------------------ metrics
def auc(p, y):
    pos, neg = p[y == 1], p[y == 0]
    return float(np.mean(pos[:, None] > neg[None, :])
                 + 0.5 * np.mean(pos[:, None] == neg[None, :]))


def f1_at(p, y, t):
    pred = p >= t
    tp = np.sum(pred & (y == 1)); fp = np.sum(pred & (y == 0))
    fn = np.sum(~pred & (y == 1))
    return 2 * tp / (2 * tp + fp + fn + 1e-9)


def best_threshold(p_val, y_val):
    return THRESHOLDS[int(np.argmax([f1_at(p_val, y_val, t) for t in THRESHOLDS]))]


def evaluate(p, y, t):
    pred = (p >= t).astype(int)
    tp = int(np.sum((pred == 1) & (y == 1))); fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1))); tn = int(np.sum((pred == 0) & (y == 0)))
    prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9)
    m = [(tp + tn) / len(y), prec, rec, 2 * prec * rec / (prec + rec + 1e-9),
         auc(p, y), bce(p, y)]
    return np.array(m), np.array([[tn, fp], [fn, tp]])


# ------------------------------------------------------------------ one seed
def train_nns(Xf, yf, Zva, yva, seed, mode_id):
    """Plain BPN (200 epochs) and GA-BPN (GA, then 100 epochs) on the fit set."""
    rb = np.random.default_rng([seed, 10 + mode_id])
    base = BPN(rng=rb)
    base.train(Xf, yf, Zva, yva, 200)
    rg = np.random.default_rng([seed, 20 + mode_id])
    ga = BPN(rng=rg)
    w, _, _ = run_ga(ga, Xf, yf, rng=rg)              # existing GA, unchanged
    ga.set_weights(w)
    ga.train(Xf, yf, Zva, yva, 100)
    return {'BPN': base, 'GA-BPN': ga}


def fuzzy_variants(p_va, p_te, D, seed, tag):
    """Return {fuzzy mode: (val probs, test probs, info)} for one network."""
    yva = D['yva']
    expert = fz.FuzzySystem()
    s_va, s_te = expert.score(D['Xva']), expert.score(D['Xte'])
    # expert fuzzy: untuned rules, only alpha chosen on validation (no GA)
    losses = [bce(fz.fuse(p_va, s_va, a), yva) for a in ALPHA_GRID]
    a0 = float(ALPHA_GRID[int(np.argmin(losses))])
    out = {'none': (p_va, p_te, {}),
           'expert': (fz.fuse(p_va, s_va, a0), fz.fuse(p_te, s_te, a0),
                      {'alpha': a0})}

    # GA-tuned fuzzy: fitness = 1 / (1 + validation log-loss of the fused score)
    def fit(g):
        sysg, a = fz.decode(g)
        return 1.0 / (1.0 + bce(fz.fuse(p_va, sysg.score(D['Xva']), a), yva))

    rng = np.random.default_rng([seed, 30 + tag])
    g, hist = fz.ga_maximize(fit, fz.N_GENES, rng,
                             seed_ind=fz.default_genes(a0))
    sysg, a = fz.decode(g)
    out['GA'] = (fz.fuse(p_va, sysg.score(D['Xva']), a),
                 fz.fuse(p_te, sysg.score(D['Xte']), a),
                 {'alpha': a, 'genes': g, 'hist': hist, 'sys': sysg})
    return out


class Store:
    def __init__(self):
        self.R, self.CM, self.P, self.T, self.info = {}, {}, {}, {}, {}

    def add(self, key, pv, pt, yva, yte):
        t = best_threshold(pv, yva)
        self.T.setdefault(key, []).append(t)
        self.P.setdefault(key, []).append((pt, yte))
        for tn, thr in (('0.5', 0.5), ('tuned', t)):
            m, cm = evaluate(pt, yte, thr)
            self.R.setdefault(key + (tn,), []).append(m)
            self.CM[key + (tn,)] = self.CM.get(key + (tn,), 0) + cm


def run_seed(seed, X, y, S):
    D = make_data(seed, X, y)
    S.info.setdefault('realism', []).append(D['realism'])
    for mid, mode in enumerate(MODES):
        if mode == 'real':
            Xf, yf = D['Ztr'], D['ytr']
        else:                                          # original + synthetic rows
            Xf = np.vstack([D['Ztr'], D['Zsyn']])
            yf = np.concatenate([D['ytr'], D['ysyn']])
        nets = train_nns(Xf, yf, D['Zva'], D['yva'], seed, mid)
        for tag, nn in enumerate(NN_NAMES):
            net = nets[nn]
            p_va = net.predict_proba(D['Zva']).ravel()
            p_te = net.predict_proba(D['Zte']).ravel()
            for fzn, (pv, pt, info) in fuzzy_variants(p_va, p_te, D, seed,
                                                      10 * mid + tag).items():
                S.add((mode, nn, fzn), pv, pt, D['yva'], D['yte'])
                if info:
                    S.info.setdefault((mode, nn, fzn), []).append(info)
            if mode == 'real' and nn == 'GA-BPN' and seed == 0:
                info = S.info[(mode, nn, 'GA')][-1]
                best_t = S.T[(mode, nn, 'GA')][-1]
                np.savez('hybrid_model.npz', w=net.get_weights(), med=D['med'],
                         mean=D['mean'], scale=D['scale'], genes=info['genes'],
                         alpha=info['alpha'], threshold=best_t)
    # fuzzy-only reference: expert rules, no network, no training
    expert = fz.FuzzySystem()
    sv, st = expert.score(D['Xva']), expert.score(D['Xte'])
    S.add(('real', 'Fuzzy only', 'expert'), sv, st, D['yva'], D['yte'])


# ------------------------------------------------------------------ reporting
MAIN = [('BPN', ('BPN', 'none')), ('GA + BPN', ('GA-BPN', 'none')),
        ('BPN + Fuzzy', ('BPN', 'expert')),
        ('GA + BPN + Fuzzy', ('GA-BPN', 'GA'))]


def arr(S, mode, nn, fzn, thr):
    return np.array(S.R[(mode, nn, fzn, thr)])           # (seeds, 6)


def label(nn, fzn):
    if nn == 'Fuzzy only':
        return 'Fuzzy only (expert rules)'
    base = {'BPN': 'BPN', 'GA-BPN': 'GA + BPN'}[nn]
    return base + {'none': '', 'expert': ' + Fuzzy(expert)',
                   'GA': ' + Fuzzy(GA-tuned)'}[fzn]


def perm_test(d):
    """Exact paired sign-flip permutation test (two-sided), n = #seeds."""
    d = np.asarray(d, float)
    signs = np.array(list(itertools.product([1, -1], repeat=len(d))))
    stat = np.abs((signs * d).mean(1))
    return float(np.mean(stat >= abs(d.mean()) - 1e-12))


def fmt_table(S, rows, thr):
    lines = [f'| Model | ' + ' | '.join(METRICS) + ' |',
             '|---|' + '---|' * len(METRICS)]
    for name, (mode, nn, fzn) in rows:
        a = arr(S, mode, nn, fzn, thr)
        lines.append(f'| {name} | ' + ' | '.join(
            f'{a[:, i].mean():.3f} ± {a[:, i].std():.3f}' for i in range(6)) + ' |')
    return '\n'.join(lines)


def build_report(S):
    md = []
    real_rows = [(n, ('real',) + k) for n, k in MAIN]
    real_rows.append(('Fuzzy only (expert rules)', ('real', 'Fuzzy only', 'expert')))
    syn_rows = [(n + ' [+synth]', ('synth',) + k) for n, k in MAIN]
    for thr, title in (('0.5', 'threshold 0.5'),
                       ('tuned', 'threshold tuned on validation (F1)')):
        md.append(f'### Original training data only, {title}\n'
                  f'(test set, mean ± std over {N_SEEDS} seeds)\n')
        md.append(fmt_table(S, real_rows, thr) + '\n')
        md.append(f'### Original + synthetic training data, {title}\n')
        md.append(fmt_table(S, syn_rows, thr) + '\n')

    md.append('### Ablation grid (mean over seeds, threshold tuned on validation)\n')
    for mode in MODES:
        md.append(f'**{mode}**\n')
        md.append('| Network | Fuzzy: none | Fuzzy: expert + alpha | Fuzzy: GA-tuned |')
        md.append('|---|---|---|---|')
        for nn in NN_NAMES:
            cells = []
            for fzn in FZ_NAMES:
                a = arr(S, mode, nn, fzn, 'tuned')
                cells.append(f'F1 {a[:, 3].mean():.3f} / AUC {a[:, 4].mean():.3f}')
            md.append(f'| {label(nn, "none")} | ' + ' | '.join(cells) + ' |')
        md.append('')

    # paired tests
    md.append('### Paired sign-flip permutation tests (exact, 10 seeds)\n')
    contrasts = []
    ref = lambda m, n, f: ('real' if m == 'real' else 'synth', n, f)
    contrasts += [('GA+BPN vs BPN', ('real', 'GA-BPN', 'none'), ('real', 'BPN', 'none')),
                  ('BPN+Fuzzy vs BPN', ('real', 'BPN', 'expert'), ('real', 'BPN', 'none')),
                  ('Full hybrid vs BPN', ('real', 'GA-BPN', 'GA'), ('real', 'BPN', 'none')),
                  ('Full hybrid vs GA+BPN', ('real', 'GA-BPN', 'GA'), ('real', 'GA-BPN', 'none')),
                  ('Full hybrid vs BPN+Fuzzy', ('real', 'GA-BPN', 'GA'), ('real', 'BPN', 'expert'))]
    for n, k in MAIN:
        contrasts.append((f'{n}: +synthetic vs original only',
                          ('synth',) + k, ('real',) + k))
    tests = [('F1', 'tuned', 3), ('F1', '0.5', 3), ('Recall', 'tuned', 2),
             ('AUC', 'tuned', 4), ('LogLoss', 'tuned', 5)]
    md.append('| Contrast | Metric | mean diff (A-B) | p |')
    md.append('|---|---|---|---|')
    m = 0
    for cname, A, B in contrasts:
        for mname, thr, i in tests:
            d = arr(S, *A, thr)[:, i] - arr(S, *B, thr)[:, i]
            md.append(f'| {cname} | {mname}@{thr} | {d.mean():+.3f} | {perm_test(d):.3f} |')
            m += 1
    md.append(f'\n{m} tests, none corrected for multiple comparisons '
              f'(Bonferroni level would be {0.05 / m:.4f}). Treat p-values as '
              f'indicative only. LogLoss: lower is better.\n')

    # synthetic realism
    rs = S.info['realism']
    md.append('### Synthetic data realism (train rows vs synthetic rows, mean over seeds)\n')
    md.append('| Feature | real mean | synth mean | real std | synth std | KS distance |')
    md.append('|---|---|---|---|---|---|')
    for j, f in enumerate(augment.FEATURES):
        md.append(f'| {f} | {np.mean([r["mean_real"][j] for r in rs]):.2f} | '
                  f'{np.mean([r["mean_syn"][j] for r in rs]):.2f} | '
                  f'{np.mean([r["std_real"][j] for r in rs]):.2f} | '
                  f'{np.mean([r["std_syn"][j] for r in rs]):.2f} | '
                  f'{np.mean([r["ks"][j] for r in rs]):.3f} |')
    md.append(f'\nPositive rate real/synthetic: '
              f'{np.mean([r["pos_rate_real"] for r in rs]):.3f} / '
              f'{np.mean([r["pos_rate_syn"] for r in rs]):.3f}; correlation matrices: '
              f'mean |diff| {np.mean([r["corr_mean_abs_diff"] for r in rs]):.3f}, '
              f'max |diff| {np.mean([r["corr_max_abs_diff"] for r in rs]):.3f}.\n')

    # learned fuzzy parameters
    inf = S.info[('real', 'GA-BPN', 'GA')]
    md.append('### What the fuzzy GA learned (full hybrid, original data)\n')
    md.append(f'alpha (GA-tuned): {np.mean([i["alpha"] for i in inf]):.2f} ± '
              f'{np.std([i["alpha"] for i in inf]):.2f}; '
              f'alpha (expert, validation grid): '
              f'{np.mean([i["alpha"] for i in S.info[("real", "GA-BPN", "expert")]]):.2f}; '
              f'validation-fitness gain of the GA over the untuned system is '
              f'measured only on validation data (test is untouched).\n')
    W = np.array([i['sys'].rule_w for i in inf])
    md.append('| Rule | mean weight ± std |')
    md.append('|---|---|')
    for r in range(fz.N_RULES):
        md.append(f'| {fz.rule_text(r)} | {W[:, r].mean():.2f} ± {W[:, r].std():.2f} |')
    sh = np.array([i['sys'].shift for i in inf])
    st = np.array([i['sys'].stretch for i in inf])
    md.append('\n| Input | mean shift ± std | mean stretch ± std |\n|---|---|---|')
    for k, n in enumerate(fz.INPUT_NAMES):
        md.append(f'| {n} | {sh[:, k].mean():+.2f} ± {sh[:, k].std():.2f} | '
                  f'{st[:, k].mean():.2f} ± {st[:, k].std():.2f} |')

    # linguistic levels
    md.append('\n### Fuzzy risk levels vs observed outcome (test rows pooled over seeds)\n')
    for key, name in ((('real', 'BPN', 'none'), 'BPN (levels from raw probability)'),
                      (('real', 'GA-BPN', 'GA'), 'Full hybrid')):
        p = np.concatenate([a for a, _ in S.P[key]])
        yy = np.concatenate([b for _, b in S.P[key]])
        lab, _ = fz.risk_levels(p)
        md.append(f'**{name}**\n\n| Level | n | observed diabetes rate |\n|---|---|---|')
        for lv in fz.LEVELS:
            s = lab == lv
            md.append(f'| {lv} | {int(s.sum())} | '
                      f'{(yy[s].mean() if s.any() else float("nan")):.2f} |')
        md.append('')
    return '\n'.join(md)


# ------------------------------------------------------------------ figures
def style(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(axis='x', color='#e4e4e4', lw=0.6)
    ax.set_axisbelow(True)


def make_figures(S):
    # 1. comparison: dot + whisker per configuration, filled=original, hollow=+synth
    panels = [('F1 (tuned threshold)', 'tuned', 3), ('Recall (tuned threshold)', 'tuned', 2),
              ('AUC', 'tuned', 4), ('Log-loss (lower is better)', 'tuned', 5)]
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.6), sharey=True)
    for ax, (title, thr, i) in zip(axs, panels):
        y = 0
        for mode in MODES:
            for n, k in MAIN:
                a = arr(S, mode, *k, thr)[:, i]
                ax.errorbar(a.mean(), y, xerr=a.std(), fmt='o', color=C[n], ms=7,
                            mfc=C[n] if mode == 'real' else 'white', mew=1.8, capsize=3)
                y += 1
            y += 0.6
        ax.set_title(title, fontsize=10)
        style(ax)
    ys = [0, 1, 2, 3, 4.6, 5.6, 6.6, 7.6]
    axs[0].set_yticks(ys)
    axs[0].set_yticklabels([n for n, _ in MAIN] + [n + ' [+synth]' for n, _ in MAIN],
                           fontsize=8)
    axs[0].invert_yaxis()
    fig.suptitle(f'Test-set comparison, mean ± std over {N_SEEDS} seeds '
                 f'(filled = original data, hollow = + synthetic)', fontsize=11)
    plt.tight_layout()
    plt.savefig('fig_hybrid_comparison.png', dpi=150); plt.close()

    # 2. confusion matrices (summed over seeds), original data
    fig, axs = plt.subplots(2, 4, figsize=(14, 6.4))
    for r, thr in enumerate(('0.5', 'tuned')):
        for c, (n, k) in enumerate(MAIN):
            cm = S.CM[('real',) + k + (thr,)]
            ax = axs[r, c]
            ax.imshow(cm, cmap='Blues')
            for (i, j), v in np.ndenumerate(cm):
                ax.text(j, i, int(v), ha='center', va='center', fontsize=12,
                        color='white' if v > cm.max() / 2 else 'black')
            ax.set(title=f'{n}\n{"threshold 0.5" if thr == "0.5" else "tuned threshold"}',
                   xticks=[0, 1], yticks=[0, 1], xlabel='Predicted', ylabel='Actual')
            ax.title.set_fontsize(9)
    fig.suptitle(f'Confusion matrices, test rows pooled over {N_SEEDS} seeds (original training data)')
    plt.tight_layout()
    plt.savefig('fig_hybrid_confusion.png', dpi=150); plt.close()

    # 3. ablation heat-maps
    fig, axs = plt.subplots(2, 2, figsize=(9.5, 6.2))
    for r, (mname, i) in enumerate((('F1 (tuned)', 3), ('AUC', 4))):
        for c, mode in enumerate(MODES):
            G = np.array([[arr(S, mode, nn, f, 'tuned')[:, i].mean()
                           for f in FZ_NAMES] for nn in NN_NAMES])
            ax = axs[r, c]
            ax.imshow(G, cmap='Blues', vmin=G.min() - 0.02, vmax=G.max() + 0.02)
            for (a, b), v in np.ndenumerate(G):
                ax.text(b, a, f'{v:.3f}', ha='center', va='center', fontsize=12)
            ax.set_xticks(range(3)); ax.set_xticklabels(['none', 'expert+α', 'GA-tuned'])
            ax.set_yticks(range(2)); ax.set_yticklabels(['BPN', 'GA + BPN'])
            ax.set_title(f'{mname}, {"original data" if mode == "real" else "+ synthetic"}',
                         fontsize=10)
            ax.set_xlabel('Fuzzy layer')
    plt.tight_layout()
    plt.savefig('fig_hybrid_ablation.png', dpi=150); plt.close()

    # 4. synthetic realism (seed 0 correlations, KS averaged over seeds)
    rs = S.info['realism']
    r0 = rs[0]
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.2))
    short = ['Preg', 'Gluc', 'BP', 'Skin', 'Insul', 'BMI', 'DPF', 'Age']
    for ax, M, t, cm_, lim in ((axs[0], r0['corr_real'], 'Correlation: real (seed 0)', 'Blues', (0, 1)),
                               (axs[1], r0['corr_syn'], 'Correlation: synthetic', 'Blues', (0, 1)),
                               (axs[2], r0['corr_syn'] - r0['corr_real'], 'Synthetic - real', 'PuOr', (-0.4, 0.4))):
        im = ax.imshow(M, cmap=cm_, vmin=lim[0], vmax=lim[1])
        ax.set_xticks(range(8)); ax.set_xticklabels(short, rotation=60, fontsize=8)
        ax.set_yticks(range(8)); ax.set_yticklabels(short, fontsize=8)
        ax.set_title(t, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)
    ks = np.mean([r['ks'] for r in rs], axis=0)
    axs[3].barh(range(8), ks, color='#0072B2')
    axs[3].set_yticks(range(8)); axs[3].set_yticklabels(short, fontsize=8)
    axs[3].invert_yaxis(); axs[3].set_title('KS distance real vs synthetic\n(0 = identical)', fontsize=10)
    style(axs[3])
    plt.tight_layout()
    plt.savefig('fig_synthetic_realism.png', dpi=150); plt.close()

    # 5. fuzzy system: membership functions, learned rule weights, GA convergence
    inf = S.info[('real', 'GA-BPN', 'GA')]
    fig = plt.figure(figsize=(16, 7.5))
    gs = GridSpec(2, 4, figure=fig)
    rngs = {'Glucose': (40, 220), 'BMI': (15, 55), 'Age': (20, 85), 'DPF': (0, 1.5)}
    for k, name in enumerate(fz.INPUT_NAMES):
        ax = fig.add_subplot(gs[0, k])
        col, _, sets = fz.INPUTS[name]
        x = np.linspace(*rngs[name], 300)
        for (sn, p), col_ in zip(sets.items(), ('#0072B2', '#E69F00', '#D55E00')):
            ax.plot(x, fz.trap(x, *p), color=col_, lw=2, label=sn)
        ax.set_title(f'{name}: expert membership functions', fontsize=10)
        ax.set_ylim(-0.02, 1.05); ax.legend(fontsize=8, frameon=False); style(ax)
    ax = fig.add_subplot(gs[1, :3])
    W = np.array([i['sys'].rule_w for i in inf])
    ax.bar(range(fz.N_RULES), W.mean(0), yerr=W.std(0), color='#009E73', capsize=2)
    ax.axhline(1, color='#999', lw=0.8, ls='--')
    ax.set_xticks(range(fz.N_RULES)); ax.set_xticklabels([f'R{r + 1}' for r in range(fz.N_RULES)])
    ax.set_ylim(0, 1.15)
    ax.set_title('GA-tuned rule weights (mean ± std over seeds; 1 = untuned expert)', fontsize=10)
    ax.grid(axis='y', color='#e4e4e4', lw=0.6)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax = fig.add_subplot(gs[1, 3])
    for i in inf:
        ax.plot(i['hist'], color='#D55E00', alpha=0.35, lw=1)
    ax.set_title('Fuzzy GA: best fitness per generation\n(one line per seed)', fontsize=10)
    ax.set_xlabel('Generation'); ax.set_ylabel('1/(1+val log-loss)'); style(ax)
    plt.tight_layout()
    plt.savefig('fig_fuzzy_system.png', dpi=150); plt.close()

    # 6. risk levels vs observed outcome
    fig, ax = plt.subplots(figsize=(7, 4.2))
    names = list(fz.LEVELS)
    for off, (key, nm, col) in zip((-0.2, 0.2), ((('real', 'BPN', 'none'), 'BPN probability', '#0072B2'),
                                                 (('real', 'GA-BPN', 'GA'), 'Full hybrid score', '#D55E00'))):
        p = np.concatenate([a for a, _ in S.P[key]])
        yy = np.concatenate([b for _, b in S.P[key]])
        lab, _ = fz.risk_levels(p)
        rate = [yy[lab == n].mean() if (lab == n).any() else 0 for n in names]
        cnt = [int((lab == n).sum()) for n in names]
        ax.bar(np.arange(4) + off, rate, 0.38, color=col, label=nm)
        for x, r_, c_ in zip(np.arange(4) + off, rate, cnt):
            ax.text(x, r_ + 0.015, f'n={c_}', ha='center', fontsize=7)
    ax.set_xticks(range(4)); ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05); ax.set_ylabel('Observed diabetes rate (test rows)')
    ax.set_title('Do the linguistic risk levels separate outcomes?', fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    plt.savefig('fig_risk_levels.png', dpi=150); plt.close()


# ------------------------------------------------------------------ main
if __name__ == '__main__':
    X, y = load_xy()
    print(f'Loaded {len(y)} rows ({int(y.sum())} positive, {int((1 - y).sum())} negative)')
    S = Store()
    for s in range(N_SEEDS):
        run_seed(s, X, y, S)
        print(f'seed {s} done')
    report = build_report(S)
    print('\n' + report)
    with open('results_hybrid.md', 'w', encoding='utf-8') as f:
        f.write(f'# Hybrid GA + NN + Fuzzy results (auto-generated by hybrid_experiment.py)\n\n'
                f'{N_SEEDS} seeds, synthetic ratio {SYN_RATIO}, test rows are original only.\n\n')
        f.write(report)
    with open('results_hybrid.csv', 'w') as f:
        f.write('mode,network,fuzzy,threshold,' + ','.join(f'{m}_mean,{m}_std' for m in METRICS) + '\n')
        for key, v in S.R.items():
            a = np.array(v)
            f.write(','.join(key) + ',' + ','.join(f'{a[:, i].mean():.4f},{a[:, i].std():.4f}'
                                                    for i in range(6)) + '\n')
    make_figures(S)
    print('\nSaved results_hybrid.md/.csv, hybrid_model.npz and fig_*.png')
