import numpy as np
from sklearn.metrics import mutual_info_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import KBinsDiscretizer


# --------------------------------------------------------
# Helper: discretization for continuous variables
# --------------------------------------------------------
def discretize(arr, num_bins=20):
    est = KBinsDiscretizer(n_bins=num_bins, encode="ordinal", strategy="quantile")
    return est.fit_transform(arr.reshape(-1, 1)).astype(int).reshape(-1)


# --------------------------------------------------------
# Mutual Information Gap (MIG)
# Chen et al., 2018
# --------------------------------------------------------
def compute_mig(latents, factors, bins=20):
    """
    latents: (N, L)
    factors: (N, K)  *must be discrete*
    """

    N, L = latents.shape
    K = factors.shape[1]

    # Discretize latents into uniform-size bins
    latents_disc = np.zeros_like(latents, dtype=int)
    for i in range(L):
        latents_disc[:, i] = discretize(latents[:, i], num_bins=bins)

    # Compute MI matrix
    mi = np.zeros((L, K))
    entropy = np.zeros(K)

    for k in range(K):
        fk = factors[:, k]

        # Ensure factor is discrete
        fk_disc = discretize(fk) if len(np.unique(fk)) > 20 else fk

        entropy[k] = mutual_info_score(fk_disc, fk_disc) + 1e-9

        for l in range(L):
            mi[l, k] = mutual_info_score(latents_disc[:, l], fk_disc)

    # MIG per factor: (top1 - top2) / entropy
    mig_per_factor = []
    for k in range(K):
        top_two = np.sort(mi[:, k])[-2:]
        gap = (top_two[-1] - top_two[-2]) / entropy[k]
        mig_per_factor.append(gap)

    return float(np.mean(mig_per_factor))


# --------------------------------------------------------
# SAP (Separated Attribute Predictability)
# Kumar et al., 2018
# --------------------------------------------------------
def compute_sap(latents, factors):
    """
    latents: (N, L)
    factors: (N, K)
    """

    N, L = latents.shape
    K = factors.shape[1]

    scores = []

    for k in range(K):
        y = factors[:, k]

        # Choose classifier or regressor
        if len(np.unique(y)) < 20:  # discrete
            clf = GradientBoostingClassifier()
        else:
            clf = GradientBoostingRegressor()

        clf.fit(latents, y)
        imp = clf.feature_importances_

        # SAP score: difference between most important and second-most
        top2 = np.sort(imp)[-2:]
        scores.append(top2[-1] - top2[-2])

    return float(np.mean(scores))


# --------------------------------------------------------
# DCI Disentanglement (Eastwood & Williams 2018)
# --------------------------------------------------------
def compute_dci(latents, factors):
    """
    latents: (N, L)
    factors: (N, K)
    """

    N, L = latents.shape
    K = factors.shape[1]

    importance = np.zeros((L, K))

    # Predict each factor from latents
    for k in range(K):
        y = factors[:, k]

        if len(np.unique(y)) < 20:
            clf = GradientBoostingClassifier()
        else:
            clf = GradientBoostingRegressor()

        clf.fit(latents, y)
        importance[:, k] = clf.feature_importances_

    # Normalize per factor
    importance_norm = importance / (importance.sum(axis=0, keepdims=True) + 1e-9)

    # DCI disentanglement metric
    entropy = -(importance_norm * np.log(importance_norm + 1e-9)).sum(axis=0)
    disentanglement = 1 - entropy / np.log(L)

    return float(disentanglement.mean())
