#from flamby.datasets.fed_heart_disease import FedHeartDisease, HeartDiseaseRaw

# To load the first center as a pytorch dataset
#center0 = FedHeartDisease(center=0, train=True)
# To load the second center as a pytorch dataset
#center1 = FedHeartDisease(center=1, train=True)
# To sample batches from each of the local datasets use the traditional pytorch API
#from torch.utils.data import DataLoader as dl

#X, y = iter(dl(center0, batch_size=16, shuffle=True, num_workers=0)).next()

import pandas as pd
import random

columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
           "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]

centers = {
    "cleveland": "flamby/datasets/fed_heart_disease/dataset_creation_scripts/heart_disease_dataset/processed.cleveland.data",
    "hungarian": "flamby/datasets/fed_heart_disease/dataset_creation_scripts/heart_disease_dataset/processed.hungarian.data",
    "switzerland": "flamby/datasets/fed_heart_disease/dataset_creation_scripts/heart_disease_dataset/processed.switzerland.data",
    "va": "flamby/datasets/fed_heart_disease/dataset_creation_scripts/heart_disease_dataset/processed.va.data",
}

dfs = {}
for name, path in centers.items():
    df = pd.read_csv(path, header=None, names=columns, na_values="?")
    df = df.dropna()
    dfs[name] = df
    print(name, df.shape)

import numpy as np
import matplotlib.pyplot as plt

# Combine all centers into one array (or plot separately if you want per-center comparison)
all_df = pd.concat(dfs.values(), ignore_index=True)

ageO = np.array(all_df["age"])
cholO = np.array(all_df["chol"])

#


def fit_multilinear(df, feature_cols, target_col):
    """
    Fit y = X @ beta (with intercept) using ordinary least squares.
    
    Returns:
        beta: coefficient array, beta[0] = intercept, beta[1:] = feature coefficients
        feature_names: list matching beta[1:] order
    """
    X = df[feature_cols].to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=float)

    # add intercept column
    X_design = np.hstack([np.ones((X.shape[0], 1)), X])

    # normal equations: beta = (X^T X)^-1 X^T y
    beta, residuals, rank, sv = np.linalg.lstsq(X_design, y, rcond=None)

    return beta, ["intercept"] + feature_cols

feature_cols = ["age"]
target_col = "chol"

beta, names = fit_multilinear(all_df, feature_cols, target_col)

for name, coef in zip(names, beta):
    print(f"{name:12s}: {coef: .4f}")

#

import numpy as np

def split_into_quantiles(df, sort_col, cols, n_quantiles):
    """
    Sort df by sort_col, rank globally, and split into n_quantiles
    equal-sized chunks.

    Returns a list of n_quantiles numpy arrays, each with columns
    matching `cols` order (e.g. [rank, age, chol]).
    """
    sorted_df = df.sort_values(sort_col, ascending=True).reset_index(drop=True)
    sorted_df["rank"] = np.arange(1, len(sorted_df) + 1)

    chunks = np.array_split(sorted_df, n_quantiles)

    arrays = [chunk[cols].to_numpy() for chunk in chunks]
    return arrays


# --- usage ---
q_arrays = split_into_quantiles(
    all_df,
    sort_col="chol",
    cols=["rank", "age", "chol"],
    n_quantiles=1,   # change this to 5, 10, whatever you need
)

for i, arr in enumerate(q_arrays, start=1):
    print(f"Group {i}: shape={arr.shape}, chol range=({arr[:,2].min():.0f}, {arr[:,2].max():.0f})")

#

def add_residuals(q_arrays, beta, age_col_idx=1, chol_col_idx=2):
    """
    For each quantile array [rank, age, chol], compute predicted chol
    from the linear fit, then append residual (flat) and residual (%)
    as two new columns.

    Returns a new list of arrays with columns:
    [rank, age, chol, residual, residual_pct]
    """
    intercept, slope = beta[0], beta[1]

    updated = []
    for arr in q_arrays:
        age = arr[:, age_col_idx]
        chol = arr[:, chol_col_idx]

        predicted = intercept + slope * age
        residual = chol - predicted
        residual_pct = (residual / chol) * 100  # residual as % of actual chol

        new_arr = np.column_stack([arr, residual, residual_pct])
        updated.append(new_arr)

    return updated


# --- usage ---
q_arrays_with_resid = add_residuals(q_arrays, beta)

for i, arr in enumerate(q_arrays_with_resid, start=1):
    print(f"Group {i}: shape={arr.shape}")
    print(arr[:3])  # peek at first 3 rows: [rank, age, chol, residual, residual_pct]

#

def create_q_encrypt(q_arrays_with_resid, beta):
    n = len(q_arrays_with_resid)
    intercept, slope = beta[0], beta[1]
    q_encrypt = [None] * n

    def build_pair_row(P, R):
        # P = original point, R = matched partner (source of "new" residual pct)
        P_rank, P_age, P_chol, _, P_resid_pct = P
        R_rank, R_resid_pct = R[0], R[4]
        pred_P = intercept + slope * P_age
        new_chol_P = round(pred_P * (100 + (random.choice([1, -1]) * R_resid_pct)) / 100)
        return [P_rank, P_age, P_chol, R_rank, P_resid_pct, R_resid_pct, new_chol_P]

    def self_pair(points, build_pair_row):
        """Randomly pair points within the same group, no replacement.
        Used only for the odd-n_quantiles middle group, since that group
        has no other group to borrow a pair from for a triple."""
        n_pts = len(points)
        idx = np.random.permutation(n_pts)
        rows = [None] * n_pts

        if n_pts == 1:
            # single leftover point, nobody to pair with in this group
            a = idx[0]
            rows[a] = build_pair_row(points[a], points[a])  # falls back to self-match
            return rows

        is_odd = n_pts % 2 == 1
        pair_idx = idx[:-3] if is_odd else idx  # reserve last 3 of idx for the triple

        k = 0
        while k + 1 < len(pair_idx):
            a, b = pair_idx[k], pair_idx[k + 1]
            rows[a] = build_pair_row(points[a], points[b])
            rows[b] = build_pair_row(points[b], points[a])
            k += 2
        if is_odd: # odd number group
            a, b, c = idx[-3], idx[-2], idx[-1]
            rows[a] = build_pair_row(points[a], points[b])  # A <- B's residual
            rows[b] = build_pair_row(points[b], points[c])  # B <- C's residual
            rows[c] = build_pair_row(points[c], points[a])  # C <- A's residual
        return rows

    i, j = 0, n - 1
    while j > i:
        qi, qj = q_arrays_with_resid[i], q_arrays_with_resid[j]
        m = min(len(qi), len(qj))

        idx_i = list(np.random.choice(len(qi), size=m, replace=False))
        idx_j = list(np.random.choice(len(qj), size=m, replace=False))

        rows_i, rows_j = [None] * len(qi), [None] * len(qj)
        pairs = list(zip(idx_i, idx_j))  # matched (a in qi, b in qj) pairs, tracked so a leftover can borrow one

        for a, b in pairs:
            rows_i[a] = build_pair_row(qi[a], qj[b])
            rows_j[b] = build_pair_row(qj[b], qi[a])

        # leftovers (larger group) get folded into a triple instead of self-paired within their own set
        leftover_i = [k for k in range(len(qi)) if rows_i[k] is None]
        leftover_j = [k for k in range(len(qj)) if rows_j[k] is None]

        # bigger group's leftovers: fold each into a triple by borrowing
        # one already-matched pair (2 points from qi, 1 from qj)
        if leftover_i:
            for L_idx in leftover_i:
                if not pairs:
                    rows_i[L_idx] = build_pair_row(qi[L_idx], qi[L_idx])  # rare fallback
                    continue
                a_idx, b_idx = pairs.pop()
                A, B, L = qi[a_idx], qj[b_idx], qi[L_idx]
                rows_i[L_idx] = build_pair_row(L, A)
                rows_i[a_idx] = build_pair_row(A, B)
                rows_j[b_idx] = build_pair_row(B, L)

        elif leftover_j:
            for L_idx in leftover_j:
                if not pairs:
                    rows_j[L_idx] = build_pair_row(qj[L_idx], qj[L_idx])  # rare fallback
                    continue
                a_idx, b_idx = pairs.pop()
                A, B, L = qi[a_idx], qj[b_idx], qj[L_idx]
                rows_j[L_idx] = build_pair_row(L, B)
                rows_j[b_idx] = build_pair_row(B, A)
                rows_i[a_idx] = build_pair_row(A, L)

        q_encrypt[i], q_encrypt[j] = np.array(rows_i), np.array(rows_j)
        i += 1
        j -= 1

    if i == j:  # odd n -> middle group left over, self-paired entirely
        mid = q_arrays_with_resid[i]
        q_encrypt[i] = np.array(self_pair(list(mid), build_pair_row))

    return q_encrypt

q_encrypt = create_q_encrypt(q_arrays_with_resid, beta)

for k, arr in enumerate(q_encrypt):
    print(f"q_encrypt[{k}]: shape={arr.shape}")
    print(arr[:3])  # [old_order, age, old_chol, new_order, old_resid_pct, new_resid_pct, new_chol]

#

def check_matches(q_encrypt, ageO, cholO):
    """
    For each new (encrypted) point, checks if there exists an old point
    in the original dataset with the same age AND the same cholesterol
    value when both are rounded to the nearest integer.

    Marks (flags) any new point that matches an old point this way —
    meaning the "encrypted" point is indistinguishable from a real
    original data point.

    Returns:
        marked_points: np.ndarray, all points with an extra column
                        (1 = marked/matched, 0 = not matched) appended.
                        Columns: [old_order, age, old_chol, new_order,
                        old_resid_pct, new_resid_pct, new_chol, marked]
        marked_count: int, number of points marked
        total_count: int, total number of points
    """
    all_encrypted = np.vstack(q_encrypt)

    ages = all_encrypted[:, 1]
    new_chols = all_encrypted[:, 6]

    old_chol_rounded = np.round(cholO)

    marked = np.zeros(len(all_encrypted), dtype=int)

    for idx in range(len(all_encrypted)):
        age = ages[idx]
        new_chol_rounded = round(new_chols[idx])

        # find all old points sharing this age
        same_age_mask = ageO == age
        candidates = old_chol_rounded[same_age_mask]

        if new_chol_rounded in candidates:
            marked[idx] = 1

    marked_points = np.column_stack([all_encrypted, marked])

    marked_count = int(np.sum(marked))
    total_count = len(all_encrypted)

    return marked_points, marked_count, total_count


marked_points, marked_count, total_count = check_matches(q_encrypt, ageO, cholO)
print(f"{marked_count} of {total_count} points match an old point on age + rounded cholesterol")

# combine all q_encrypt groups into one array
all_encrypted = np.vstack(q_encrypt)

# columns: [old_order, age, old_chol, new_order, old_resid_pct, new_resid_pct, new_chol]
ageN = all_encrypted[:, 1]
cholN = all_encrypted[:, 6]

#

df_new = pd.DataFrame({"age": ageN, "chol": cholN})
beta_new, names_new = fit_multilinear(df_new, feature_cols, target_col)

print("Encrypted fit:")
for name, coef in zip(names_new, beta_new):
    print(f"  {name:12s}: {coef: .4f}")

age_range = np.linspace(
    min(ageO.min(), ageN.min()),
    max(ageO.max(), ageN.max()),
    100
)

# PLOTTING

plt.figure(figsize=(8, 6))
plt.scatter(ageO, cholO, alpha=0.6, edgecolor="k", color="blue")
plt.scatter(ageN, cholN, alpha=0.6, edgecolor="k", color="green")

line_old = beta[0] + beta[1] * age_range
line_new = beta_new[0] + beta_new[1] * age_range

plt.plot(age_range, line_old, color="tab:blue", linewidth=2, label="Original fit")
plt.plot(age_range, line_new, color="tab:orange", linewidth=2, label="Encrypted fit")

plt.xlabel("Age")
plt.ylabel("Cholesterol (mg/dl)")
plt.title("Age vs Cholesterol")
plt.grid(True, alpha=0.3)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

def build_discriminator_dataset(ageO, cholO, ageN, cholN, beta):
    """
    Builds a labeled dataset for the discriminator:
    label 0 = real/original point, label 1 = encrypted/swapped point.
    Adds a residual feature: chol - predicted(age), using the
    ORIGINAL regression fit (beta) as the reference line for both
    real and encrypted points.
    """
    intercept, slope = beta[0], beta[1]

    predicted_O = intercept + slope * ageO
    residual_O = cholO - predicted_O

    predicted_N = intercept + slope * ageN
    residual_N = cholN - predicted_N

    real_df = pd.DataFrame({
        "age": ageO,
        "chol": cholO,
        "residual": residual_O,
        "label": 0
    })
    fake_df = pd.DataFrame({
        "age": ageN,
        "chol": cholN,
        "residual": residual_N,
        "label": 1
    })

    combined = pd.concat([real_df, fake_df], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    return combined


def train_discriminator(combined_df, feature_cols=["age", "chol", "residual"]):
    X = combined_df[feature_cols].to_numpy(dtype=float)
    y = combined_df["label"].to_numpy(dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"Discriminator accuracy: {acc:.4f}")
    print(f"Discriminator AUC:      {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["real", "encrypted"]))

    return clf, acc, auc

def train_rf_discriminator(combined_df, feature_cols=["age", "chol", "residual"], n_estimators=100):
    """
    Trains a Random Forest discriminator and reports how well it
    distinguishes real vs encrypted points.

    Returns:
        clf: trained RandomForestClassifier
        acc: test accuracy
        auc: test AUC
    """
    X = combined_df[feature_cols].to_numpy(dtype=float)
    y = combined_df["label"].to_numpy(dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"Random Forest accuracy: {acc:.4f}")
    print(f"Random Forest AUC:      {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["real", "encrypted"]))

    print("\nFeature importances:")
    for name, importance in zip(feature_cols, clf.feature_importances_):
        print(f"  {name:12s}: {importance:.4f}")

    return clf, acc, auc


# --- usage ---
combined = build_discriminator_dataset(ageO, cholO, ageN, cholN, beta)
clf, acc, auc = train_discriminator(combined, feature_cols=["age", "chol", "residual"])
rf_clf, rf_acc, rf_auc = train_rf_discriminator(combined, feature_cols=["age", "chol", "residual"])

plt.show()