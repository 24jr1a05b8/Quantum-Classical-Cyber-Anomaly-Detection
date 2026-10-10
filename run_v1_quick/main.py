
"""Quantum-Classical Cyber Anomaly Detection using CICIDS2017."""

import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.metrics import average_precision_score

from qiskit.circuit.library import zz_feature_map
from qiskit.quantum_info import Statevector


# ------------------------- CONFIG -------------------------

QUICK = "--quick" in sys.argv

SEEDS = [0] if QUICK else [0, 1, 2]
KS = (4, 6) if QUICK else (4, 6, 8)

N_TRAIN = 300 if QUICK else 1000
N_VAL = 500 if QUICK else 1000

PER_TEST, PER_TUNE = (20, 8) if QUICK else (60, 20)

NU = 0.05
REPS = 2

GRID = {
    "classical": [0.05, 0.1, 0.5, 1, 2],
    "quantum": [0.25, 0.5, 1, 2],
}

CHART_K = KS[1]

FILES = {
    "mon": "Monday-WorkingHours.pcap_ISCX.csv",
    "wed": "Wednesday-workingHours.pcap_ISCX.csv",
    "fri": "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
}

FEATS = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Packet Length Mean",
    "Packet Length Std",
    "Flow IAT Mean",
    "Flow IAT Std",
]


def log(*args):
    print(time.strftime("%H:%M:%S"), *args, flush=True)


# ------------------------- DATA LOADING -------------------------

def load(path):
    wanted = set(FEATS + ["Label"])

    df = pd.read_csv(
        path,
        usecols=lambda col: col.strip() in wanted,
        skipinitialspace=True,
    )

    df.columns = df.columns.str.strip()

    df = df[FEATS + ["Label"]].copy()

    for col in FEATS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[FEATS] = df[FEATS].replace([np.inf, -np.inf], np.nan)
    df = df.dropna().drop_duplicates()
    df["Label"] = df["Label"].astype(str).str.strip()

    return df


log("Loading data ...")

mon = load(FILES["mon"])
pool = pd.concat(
    [load(FILES["wed"]), load(FILES["fri"])],
    ignore_index=True,
)

ben = pool[pool["Label"] == "BENIGN"].sample(
    frac=1, random_state=0
).reset_index(drop=True)

att = pool[pool["Label"] != "BENIGN"]

if ben.empty or att.empty:
    raise ValueError("No benign or attack samples found. Check the CSV files.")

# Split each attack type into test and tuning samples.
att_te, att_tu = [], []

for attack_label, group in att.groupby("Label"):
    group = group.sample(frac=1, random_state=0)

    n_test = min(PER_TEST, len(group))
    n_tune = min(PER_TUNE, max(0, len(group) - n_test))

    if n_test:
        att_te.append(group.iloc[:n_test])

    if n_tune:
        att_tu.append(group.iloc[n_test:n_test + n_tune])

if not att_te or not att_tu:
    raise ValueError(
        "Not enough attack samples for test and tuning. "
        "Reduce PER_TEST or PER_TUNE."
    )

att_te = pd.concat(att_te, ignore_index=True)
att_tu = pd.concat(att_tu, ignore_index=True)

# Maintain approximately 10% attack prevalence in each evaluation set.
n_te_b = min(9 * len(att_te), len(ben))
n_tu_b = min(3 * len(att_tu), max(0, len(ben) - n_te_b))

test = pd.concat(
    [ben.iloc[:n_te_b], att_te], ignore_index=True
).sample(frac=1, random_state=10).reset_index(drop=True)

tune = pd.concat(
    [ben.iloc[n_te_b:n_te_b + n_tu_b], att_tu],
    ignore_index=True,
).sample(frac=1, random_state=11).reset_index(drop=True)

y = (test["Label"] != "BENIGN").astype(int).to_numpy()
labels = test["Label"].to_numpy()

ytu = (tune["Label"] != "BENIGN").astype(int).to_numpy()

log(
    f"Monday benign: {len(mon)} | "
    f"test: {len(test)} ({int(y.sum())} attacks) | "
    f"tune: {len(tune)} ({int(ytu.sum())} attacks)"
)

log(
    "Attack types in test:",
    dict(pd.Series(labels[y == 1]).value_counts()),
)


# ------------------------- QUANTUM FEATURE MAP -------------------------

def make_fm(k):
    return zz_feature_map(feature_dimension=k, reps=REPS)


def qstates(X, k):
    """Encode each row and return its quantum statevector."""
    circuit = make_fm(k)
    parameters = list(circuit.parameters)
    states = []

    for row in X:
        values = {
            parameter: float(value)
            for parameter, value in zip(parameters, row)
        }
        bound = circuit.assign_parameters(values)
        states.append(Statevector(bound).data)

    return np.asarray(states)


def quantum_kernel(A, B):
    """Fidelity kernel: |<psi(A)|psi(B)>| squared."""
    return np.abs(A.conj() @ B.T) ** 2


# ------------------------- MODELS -------------------------

def fit(name, X_train, gamma):
    if name == "classical":
        model = OneClassSVM(
            kernel="rbf",
            gamma=gamma,
            nu=NU,
        )
        model.fit(X_train)

        # Higher scores should mean more anomalous.
        return lambda X: -model.decision_function(X).ravel()

    # Quantum kernel model
    train_states = qstates(X_train, X_train.shape[1])
    K_train = quantum_kernel(train_states, train_states)

    model = OneClassSVM(
        kernel="precomputed",
        nu=NU,
    )
    model.fit(K_train)

    def score(X):
        test_states = qstates(X, X_train.shape[1])
        K_test = quantum_kernel(test_states, train_states)
        return -model.decision_function(K_test).ravel()

    return score


# ------------------------- EVALUATION -------------------------

def evaluate(validation_scores, test_scores, fpr):
    """Set threshold using benign validation data."""
    benign_scores = validation_scores[:N_VAL]

    threshold = np.quantile(benign_scores, 1 - fpr)
    predicted = test_scores >= threshold

    attack_mask = y == 1
    benign_mask = y == 0

    recall = (
        predicted[attack_mask].mean()
        if attack_mask.any()
        else 0.0
    )

    precision = (
        attack_mask[predicted].sum() / predicted.sum()
        if predicted.sum()
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    per_attack = {}

    for attack in np.unique(labels[attack_mask]):
        mask = labels == attack
        per_attack[attack] = float(predicted[mask].mean())

    # Return measured benign false-positive rate on the test set too.
    measured_fpr = (
        float(predicted[benign_mask].mean())
        if benign_mask.any()
        else 0.0
    )

    return recall, f1, per_attack, measured_fpr


# ------------------------- EXPERIMENT -------------------------

rows = []
per_rows = []

for seed in SEEDS:
    m = mon.sample(frac=1, random_state=seed).reset_index(drop=True)

    train = m.iloc[:N_TRAIN]
    val = m.iloc[N_TRAIN:N_TRAIN + N_VAL]

    if len(train) < N_TRAIN or len(val) < N_VAL:
        raise ValueError("Not enough Monday samples for train/validation.")

    def raw_features(df):
        return np.log1p(df[FEATS].clip(lower=0).to_numpy())

    scaler = StandardScaler().fit(raw_features(train))

    def standardize(df):
        return scaler.transform(raw_features(df))

    Z_train = standardize(train)

    corr = np.abs(np.corrcoef(Z_train.T))
    corr = np.nan_to_num(corr, nan=0.0)

    keep = []

    for i in range(len(FEATS)):
        if all(corr[i, j] <= 0.95 for j in keep):
            keep.append(i)

    log(
        f"Seed {seed}: kept {len(keep)} of "
        f"{len(FEATS)} features after correlation filtering"
    )

    for requested_k in KS:
        k = min(requested_k, len(keep))

        pca = PCA(n_components=k).fit(Z_train[:, keep])

        def project(df):
            return pca.transform(standardize(df)[:, keep])

        minmax = MinMaxScaler(feature_range=(0, np.pi))
        minmax.fit(project(train))

        def transform(df):
            return np.clip(minmax.transform(project(df)), 0, np.pi)

        X_train = transform(train)
        X_val = transform(val)
        X_test = transform(test)
        X_tune = transform(tune)

        # Use only the known benign validation subset to set thresholds.
        val_benign_scores = {}

        for name in ("classical", "quantum"):
            start = time.time()
            best_ap = -1
            best_score = None
            best_gamma = None

            for gamma in GRID[name]:
                score = fit(name, X_train, gamma)
                tuning_scores = score(X_tune)

                ap = average_precision_score(ytu, tuning_scores)

                if ap > best_ap:
                    best_ap = ap
                    best_score = score
                    best_gamma = gamma

            val_scores = best_score(X_val)
            test_scores = best_score(X_test)

            # Validation is all benign, as Monday is benign-only.
            r1, _, per1, fpr1 = evaluate(val_scores, test_scores, 0.01)
            r5, f5, per5, fpr5 = evaluate(val_scores, test_scores, 0.05)

            rows.append({
                "seed": seed,
                "k": k,
                "model": name,
                "gamma": best_gamma,
                "Recall_1": r1,
                "Test_FPR_1": fpr1,
                "Recall_5": r5,
                "Test_FPR_5": fpr5,
                "F1_5": f5,
                "PRAUC": average_precision_score(y, test_scores),
            })

            for attack, recall in per5.items():
                per_rows.append({
                    "seed": seed,
                    "k": k,
                    "model": name,
                    "attack": attack,
                    "recall_5": recall,
                })

            log(
                f"Seed {seed} k={k} {name:9s} "
                f"gamma={best_gamma} "
                f"recall@1%={r1:.3f} "
                f"recall@5%={r5:.3f} "
                f"({time.time() - start:.0f}s)"
            )


# ------------------------- RESULTS -------------------------

res = pd.DataFrame(rows)
res.to_csv("results.csv", index=False)

per = pd.DataFrame(per_rows)
per.to_csv("per_attack_recall.csv", index=False)

summary = res.groupby(["k", "model"])[
    ["Recall_1", "Recall_5", "F1_5", "PRAUC", "Test_FPR_1", "Test_FPR_5"]
].agg(["mean", "std"]).round(3)

summary.to_csv("summary.csv")

print("\n=== SUMMARY (mean, std over seeds) ===")
print(summary.to_string())

chart_k = min(CHART_K, per["k"].max())

pa = (
    per[per.k == chart_k]
    .groupby(["attack", "model"])["recall_5"]
    .mean()
    .unstack()
    .round(3)
)

print(f"\n=== PER-ATTACK RECALL at 5% target FPR (k={chart_k}) ===")
print(pa.to_string())


# ------------------------- CHARTS -------------------------

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

for ax, col, title in zip(
    axes,
    ["Recall_1", "Recall_5"],
    ["Recall at 1% target FPR", "Recall at 5% target FPR"],
):
    grouped = res.groupby(["k", "model"])[col].agg(["mean", "std"])
    means = grouped["mean"].unstack()
    errors = grouped["std"].unstack().fillna(0)

    means.plot.bar(ax=ax, yerr=errors, capsize=3, rot=0)
    ax.set_title(title)
    ax.set_xlabel("PCA components (= qubits)")
    ax.set_ylim(0, 1)

axes[0].set_ylabel("Attack recall")
plt.tight_layout()
plt.savefig("chart_recall.png", dpi=150)
plt.close()

pa.plot.bar(figsize=(8, 4), rot=20)
plt.ylabel("Recall at 5% target FPR")
plt.title(f"Recall per attack type (k={chart_k})")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("chart_per_attack.png", dpi=150)
plt.close()


# ------------------------- CIRCUIT DIAGRAM -------------------------

try:
    make_fm(4).decompose().draw("mpl", filename="circuit.png")
except Exception as exc:
    log("Circuit image unavailable:", exc)
    try:
        open("circuit.txt", "w", encoding="utf-8").write(
            str(make_fm(4).draw("text"))
        )
    except Exception as inner_exc:
        log("Circuit text unavailable:", inner_exc)

log(
    "Done. Files: results.csv, summary.csv, "
    "per_attack_recall.csv, chart_recall.png, "
    "chart_per_attack.png, circuit.png"
)