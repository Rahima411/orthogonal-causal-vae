# metrics/disentanglement.py
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer
import warnings
warnings.filterwarnings("ignore")  # Kill the annoying sklearn warning globally


def discretize(arr, num_bins=20):
    """Fast and silent discretization"""
    est = KBinsDiscretizer(n_bins=num_bins, encode='ordinal', strategy='uniform')
    return est.fit_transform(arr.reshape(-1, 1)).ravel().astype(int)


def compute_mig(latents, factors, num_bins=20):
    """Fast MIG — uses uniform bins (much faster than quantile)"""
    latents_disc = np.zeros_like(latents, dtype=int)
    for i in range(latents.shape[1]):
        latents_disc[:, i] = discretize(latents[:, i], num_bins)

    mi = np.zeros((latents.shape[1], factors.shape[1]))
    for i in range(latents.shape[1]):
        for j in range(factors.shape[1]):
            mi[i, j] = mutual_info_score(latents_disc[:, i], factors[:, j].astype(int))

    sorted_mi = np.sort(mi, axis=0)[::-1]
    mig = np.mean(sorted_mi[0] - sorted_mi[1])  # top - second-top
    return float(mig)


def compute_sap(latents, factors):
    """SAP score — very fast with LogisticRegression"""
    scores = []
    for j in range(factors.shape[1]):
        clf = LogisticRegression(max_iter=1000, solver='lbfgs')
        clf.fit(latents, factors[:, j].astype(int))
        score_matrix = np.abs(clf.coef_)
        sorted_scores = np.sort(score_matrix, axis=1)[0][::-1]
        scores.append(sorted_scores[0] - sorted_scores[1])
    return float(np.mean(scores))


def compute_dci(latents, factors):
    """DCI Disentanglement (fast version with LinearRegression)"""
    importance = np.zeros((latents.shape[1], factors.shape[1]))
    for j in range(factors.shape[1]):
        reg = LinearRegression()
        reg.fit(latents, factors[:, j])
        importance[:, j] = np.abs(reg.coef_)
    importance /= importance.sum(axis=0, keepdims=True) + 1e-10
    disentanglement = 1.0 - (-importance * np.log(importance + 1e-10)).sum(axis=0) / np.log(latents.shape[1])
    return float(disentanglement.mean())