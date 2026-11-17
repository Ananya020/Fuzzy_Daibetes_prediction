import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# sklearn is imported inside prepare() / metrics() / run_once() so that the
# BPN and GA classes can be imported without it (e.g. by the numpy-only
# hybrid_experiment.py).

COLS = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
        'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
ZERO_COLS = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
URL = ('https://raw.githubusercontent.com/jbrownlee/Datasets/master/'
       'pima-indians-diabetes.data.csv')


# ---------------------------------------------------------------- data
def load_data():
    if os.path.exists('diabetes.csv'):
        # The Kaggle/Brownlee copies have NO header row. Reading with the
        # default header=0 silently drops the first patient (767 rows instead
        # of 768), so detect a header explicitly instead of assuming one.
        with open('diabetes.csv') as f:
            first = f.readline().split(',')[0].strip()
        has_header = not first.replace('.', '', 1).isdigit()
        return pd.read_csv('diabetes.csv', header=0 if has_header else None,
                           names=COLS)
    try:
        return pd.read_csv(URL, header=None, names=COLS)
    except Exception:
        from sklearn.datasets import fetch_openml
        d = fetch_openml('diabetes', version=1, as_frame=True)
        df = d.data.copy()
        df.columns = COLS[:-1]
        df['Outcome'] = (d.target == 'tested_positive').astype(int)
        return df


def prepare(seed):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    df = load_data()
    X = df.drop(columns='Outcome').copy()
    y = df['Outcome'].values
    X[ZERO_COLS] = X[ZERO_COLS].replace(0, np.nan)      # zeros mean missing
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.30,
                                            stratify=y, random_state=seed)
    Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.50,
                                          stratify=ytmp, random_state=seed)
    med = Xtr.median()                                   # train-only medians
    Xtr, Xva, Xte = [a.fillna(med) for a in (Xtr, Xva, Xte)]
    sc = StandardScaler().fit(Xtr)
    Xtr, Xva, Xte = [sc.transform(a) for a in (Xtr, Xva, Xte)]
    return Xtr, Xva, Xte, ytr, yva, yte, med, sc


# ---------------------------------------------------------------- BPN
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def bce(p, y):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def mse(p, y):
    return float(np.mean((p - y) ** 2))


class BPN:
    """8-H-1 feed-forward network trained with backprop + momentum."""

    def __init__(self, n_in=8, n_hid=12, lr=0.1, mom=0.9, w=None, rng=None):
        self.n_in, self.n_hid, self.lr, self.mom = n_in, n_hid, lr, mom
        self.n_w = n_in * n_hid + n_hid + n_hid + 1
        if w is None:
            rng = rng or np.random.default_rng()
            w = rng.normal(0, 0.5, self.n_w)
        self.set_weights(w)

    def set_weights(self, w):
        a, h = self.n_in, self.n_hid
        self.W1 = w[:a * h].reshape(a, h).copy()
        self.b1 = w[a * h:a * h + h].copy()
        self.W2 = w[a * h + h:a * h + 2 * h].reshape(h, 1).copy()
        self.b2 = w[a * h + 2 * h:].copy()

    def get_weights(self):
        return np.concatenate([self.W1.ravel(), self.b1,
                               self.W2.ravel(), self.b2])

    def _fwd(self, X):
        h = sigmoid(X @ self.W1 + self.b1)
        return h, sigmoid(h @ self.W2 + self.b2)

    def predict_proba(self, X):
        return self._fwd(X)[1]

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int).ravel()

    def train(self, X, y, Xv, yv, epochs=200):
        y, yv = y.reshape(-1, 1), yv.reshape(-1, 1)
        params = [self.W1, self.b1, self.W2, self.b2]
        vel = [np.zeros_like(p) for p in params]
        hist = {'train': [], 'val': []}
        for _ in range(epochs):
            h, o = self._fwd(X)
            d2 = (o - y) / len(X)                  # output delta (sigmoid + BCE)
            d1 = (d2 @ self.W2.T) * h * (1 - h)    # hidden delta (backpropagated)
            grads = [X.T @ d1, d1.sum(0), h.T @ d2, d2.sum(0)]
            for p, v, g in zip(params, vel, grads):
                v *= self.mom
                v -= self.lr * g
                p += v                              # in-place weight update
            hist['train'].append(bce(self.predict_proba(X), y))
            hist['val'].append(bce(self.predict_proba(Xv), yv))
        return hist


# ---------------------------------------------------------------- GA
def run_ga(net, X, y, pop=40, gens=40, pc=0.8, pm=0.1, sigma=0.3,
           elite=2, k=3, rng=None):
    y = y.reshape(-1, 1)
    n_w = net.n_w

    def fitness(w):
        net.set_weights(w)
        return 1.0 / (1.0 + mse(net.predict_proba(X), y))

    def tournament(P, f):
        idx = rng.integers(0, len(P), k)
        return P[idx[np.argmax(f[idx])]]

    P = rng.normal(0, 0.5, (pop, n_w))                  # initial population
    best_hist, avg_hist = [], []
    for _ in range(gens):
        f = np.array([fitness(w) for w in P])
        best_hist.append(f.max())
        avg_hist.append(f.mean())
        order = np.argsort(-f)
        new = [P[i].copy() for i in order[:elite]]       # elitism
        while len(new) < pop:
            p1, p2 = tournament(P, f), tournament(P, f)  # selection
            if rng.random() < pc:                        # uniform crossover
                mask = rng.random(n_w) < 0.5
                child = np.where(mask, p1, p2)
            else:
                child = p1.copy()
            m = rng.random(n_w) < pm                     # gaussian mutation
            child = child + m * rng.normal(0, sigma, n_w)
            new.append(child)
        P = np.array(new)
    f = np.array([fitness(w) for w in P])
    return P[np.argmax(f)], best_hist, avg_hist


# ---------------------------------------------------------------- experiment
def metrics(y, p):
    from sklearn.metrics import (accuracy_score, precision_score,
                                 recall_score, f1_score)
    return {'Accuracy': accuracy_score(y, p),
            'Precision': precision_score(y, p, zero_division=0),
            'Recall': recall_score(y, p),
            'F1': f1_score(y, p)}


def run_once(seed, epochs_base=200, epochs_ft=100):
    from sklearn.metrics import confusion_matrix
    rng = np.random.default_rng(seed)
    Xtr, Xva, Xte, ytr, yva, yte, med, sc = prepare(seed)

    base = BPN(rng=rng)                                  # plain BPN
    h_base = base.train(Xtr, ytr, Xva, yva, epochs_base)

    ga_net = BPN(rng=rng)                                # GA-BPN
    w, best, avg = run_ga(ga_net, Xtr, ytr, rng=rng)
    ga_net.set_weights(w)
    h_ga = ga_net.train(Xtr, ytr, Xva, yva, epochs_ft)

    return {'base': metrics(yte, base.predict(Xte)),
            'ga': metrics(yte, ga_net.predict(Xte)),
            'cm_base': confusion_matrix(yte, base.predict(Xte)),
            'cm_ga': confusion_matrix(yte, ga_net.predict(Xte)),
            'h_base': h_base, 'h_ga': h_ga, 'best': best, 'avg': avg,
            'model': (ga_net, med, sc)}


if __name__ == '__main__':
    N = 10
    runs = [run_once(s) for s in range(N)]

    keys = ['Accuracy', 'Precision', 'Recall', 'F1']
    table = {}
    for name in ('base', 'ga'):
        arr = np.array([[r[name][k] for k in keys] for r in runs])
        table[name] = (arr.mean(0), arr.std(0))
    print(f'\nResults over {N} random splits (test set), mean +/- std')
    print(f'{"Metric":<11}{"Plain BPN":>18}{"GA-BPN":>18}')
    for i, k in enumerate(keys):
        b, g = table['base'], table['ga']
        print(f'{k:<11}{b[0][i]:>10.3f} +/- {b[1][i]:.3f}'
              f'{g[0][i]:>10.3f} +/- {g[1][i]:.3f}')

    r = runs[0]                                          # plots for seed 0
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].plot(r['best'], label='Best')
    ax[0].plot(r['avg'], label='Average')
    ax[0].set(title='GA fitness vs generation', xlabel='Generation',
              ylabel='Fitness = 1/(1+MSE)')
    ax[0].legend()
    ax[1].plot(r['h_base']['val'], label='Plain BPN (random init)')
    ax[1].plot(r['h_ga']['val'], label='GA-BPN (GA init + fine-tune)')
    ax[1].set(title='Validation loss per epoch', xlabel='Epoch',
              ylabel='BCE loss')
    ax[1].legend()
    x = np.arange(len(keys))
    ax[2].bar(x - 0.2, table['base'][0], 0.4, yerr=table['base'][1],
              label='Plain BPN')
    ax[2].bar(x + 0.2, table['ga'][0], 0.4, yerr=table['ga'][1],
              label='GA-BPN')
    ax[2].set_xticks(x)
    ax[2].set_xticklabels(keys)
    ax[2].set(title=f'Test metrics (mean over {N} seeds)', ylim=(0, 1))
    ax[2].legend()
    plt.tight_layout()
    plt.savefig('results.png', dpi=150)

    fig, ax = plt.subplots(1, 2, figsize=(8, 3.6))
    for a, key, t in zip(ax, ('cm_base', 'cm_ga'), ('Plain BPN', 'GA-BPN')):
        a.imshow(r[key], cmap='Blues')
        for (i, j), v in np.ndenumerate(r[key]):
            a.text(j, i, v, ha='center', va='center', fontsize=14)
        a.set(title=t, xlabel='Predicted', ylabel='Actual',
              xticks=[0, 1], yticks=[0, 1])
    plt.tight_layout()
    plt.savefig('confusion.png', dpi=150)

    net, med, sc = r['model']                            # save for the app
    np.savez('model.npz', w=net.get_weights(), med=med.values,
             mean=sc.mean_, scale=sc.scale_)
    print('\nSaved results.png, confusion.png, model.npz')