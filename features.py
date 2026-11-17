"""Engineered features for the diabetes data (numpy only).

Input: imputed RAW features in the usual column order
    0 Pregnancies 1 Glucose 2 BloodPressure 3 SkinThickness 4 Insulin
    5 BMI 6 DiabetesPedigreeFunction 7 Age
Output: the 8 original columns followed by 7 engineered columns.
Everything is computed row by row (no statistics from other rows), so there
is nothing to leak; scaling is done afterwards on training rows only.
"""
import numpy as np

BASE_NAMES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
              'Insulin', 'BMI', 'DPF', 'Age']
ENG_NAMES = ['Glucose*BMI', 'Age*BMI', 'Age*Glucose', 'log(DPF)',
             'Glucose^2', 'BMI^2', 'Pregnancies/Age']
POOL_NAMES = BASE_NAMES + ENG_NAMES


def engineer(X):
    g, bmi, age = X[:, 1], X[:, 5], X[:, 7]
    eng = np.column_stack([g * bmi, age * bmi, age * g, np.log(X[:, 6]),
                           g ** 2, bmi ** 2, X[:, 0] / age])
    return np.hstack([X, eng])
