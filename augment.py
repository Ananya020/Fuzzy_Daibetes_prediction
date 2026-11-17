"""
Synthetic data augmentation for the diabetes project (numpy only).

Idea (easy to explain in a viva):
  * Fit ONE multivariate Gaussian PER CLASS on the (imputed) TRAINING rows.
    The mean vector keeps the typical level of every feature, the covariance
    matrix keeps the relationships between features (e.g. BMI vs SkinThickness).
  * Skewed features (Insulin, DiabetesPedigreeFunction, Age, Pregnancies) are
    modelled on a log1p scale, because a Gaussian is a poor fit for them.
  * Draw new rows from each class Gaussian, convert back, round the integer
    features and clip every feature to the range seen in training, so no
    impossible values (negative age, BMI of 3, ...) appear.
  * Number of synthetic rows per class = ratio x real rows of that class,
    so the class distribution is preserved exactly.

What this is NOT: it adds no new medical information. It only re-samples the
mean and covariance estimated from the real training rows, so any gain must
come from regularisation (a smoother, denser picture of the same distribution),
not from "more data". The experiments test whether it helps at all.

Only TRAINING rows are ever used to fit the generator; validation and test
rows are never touched (no leakage).
"""
import numpy as np

FEATURES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
LOG_IDX = [0, 4, 6, 7]            # Pregnancies, Insulin, DPF, Age -> log1p
INT_IDX = [0, 1, 2, 3, 4, 7]      # integer-valued in the original data


def _to_model_space(X):
    Z = X.astype(float).copy()
    Z[:, LOG_IDX] = np.log1p(Z[:, LOG_IDX])
    return Z


def _from_model_space(Z):
    X = Z.copy()
    X[:, LOG_IDX] = np.expm1(X[:, LOG_IDX])
    return X


def fit_generator(X, y):
    """X: imputed raw training features (n, 8), y: 0/1 labels."""
    gen = {'lo': X.min(0), 'hi': X.max(0), 'classes': {}}
    for c in (0, 1):
        Z = _to_model_space(X[y == c])
        cov = np.cov(Z, rowvar=False) + 1e-6 * np.eye(Z.shape[1])
        gen['classes'][c] = (Z.mean(0), cov, int(np.sum(y == c)))
    return gen


def generate(gen, ratio, rng):
    """Return (X_syn, y_syn) with `ratio` x (real class count) rows per class."""
    Xs, ys = [], []
    for c, (mu, cov, n_real) in gen['classes'].items():
        n_new = int(round(ratio * n_real))
        Z = rng.multivariate_normal(mu, cov, size=n_new)
        X = _from_model_space(Z)
        X[:, INT_IDX] = np.round(X[:, INT_IDX])
        X = np.clip(X, gen['lo'], gen['hi'])
        Xs.append(X)
        ys.append(np.full(n_new, c))
    return np.vstack(Xs), np.concatenate(ys)


# ------------------------------------------------------------ SMOTE
def smote(X, y, rng, k=5):
    """Classic SMOTE, numpy only. Balances the classes 1:1.

    For each new row: pick a random minority row, pick one of its k nearest
    minority neighbours (distance on std-scaled features), and place the new
    row at a random point on the straight line between the two.
    NOTE: unlike the Gaussian generator this CHANGES the class distribution
    (to 50/50), so predicted probabilities shift upwards; a threshold tuned on
    an un-augmented validation set compensates for that."""
    minority = 1 if np.sum(y == 1) < np.sum(y == 0) else 0
    Xm = X[y == minority]
    n_new = int(np.sum(y != minority) - len(Xm))
    if n_new <= 0:
        return X, y
    s = X.std(0) + 1e-12
    D = (((Xm[:, None, :] - Xm[None, :, :]) / s) ** 2).sum(-1)
    np.fill_diagonal(D, np.inf)
    nn = np.argsort(D, axis=1)[:, :k]
    i = rng.integers(0, len(Xm), n_new)
    j = nn[i, rng.integers(0, nn.shape[1], n_new)]
    lam = rng.random((n_new, 1))
    Xn = Xm[i] + lam * (Xm[j] - Xm[i])
    return np.vstack([X, Xn]), np.concatenate([y, np.full(n_new, minority)])


# ------------------------------------------------------------ realism checks
def ks_statistic(a, b):
    """Two-sample Kolmogorov-Smirnov distance (0 = identical distributions)."""
    allv = np.sort(np.concatenate([a, b]))
    ca = np.searchsorted(np.sort(a), allv, side='right') / len(a)
    cb = np.searchsorted(np.sort(b), allv, side='right') / len(b)
    return float(np.max(np.abs(ca - cb)))


def realism_report(Xr, yr, Xs, ys):
    """Compare real and synthetic rows. Returns a dict of plain numbers."""
    rep = {'n_real': len(Xr), 'n_syn': len(Xs),
           'pos_rate_real': float(yr.mean()), 'pos_rate_syn': float(ys.mean()),
           'mean_real': Xr.mean(0), 'mean_syn': Xs.mean(0),
           'std_real': Xr.std(0), 'std_syn': Xs.std(0),
           'ks': np.array([ks_statistic(Xr[:, j], Xs[:, j])
                           for j in range(Xr.shape[1])])}
    cr = np.corrcoef(Xr, rowvar=False)
    cs = np.corrcoef(Xs, rowvar=False)
    rep['corr_real'], rep['corr_syn'] = cr, cs
    rep['corr_max_abs_diff'] = float(np.max(np.abs(cr - cs)))
    rep['corr_mean_abs_diff'] = float(np.mean(np.abs(cr - cs)))
    return rep
