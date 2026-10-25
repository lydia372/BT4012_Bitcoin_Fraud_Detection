"""
Run all experiments (Phases A–D) and generate submission CSVs.
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_score,
    recall_score, f1_score,
)
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA
import xgboost as xgb
import lightgbm as lgb

DATA_DIR = Path('/Users/yichuan/Desktop/BT4012/Bitcoin')

# =====================================================================
# Section 6: Data loading and setup
# =====================================================================
print("=" * 60)
print("SECTION 6: Loading data...")
print("=" * 60)

train = pd.read_csv(DATA_DIR / 'train.csv')
test  = pd.read_csv(DATA_DIR / 'test.csv')
edges = pd.read_csv(DATA_DIR / 'txs_edgelist.csv', usecols=['txId1', 'txId2'])

feature_cols = [c for c in train.columns if c.startswith('feat_')]
print(f'train={len(train):,}  test={len(test):,}  edges={len(edges):,}  features={len(feature_cols)}')

labelled = train[train['label'].notna()].copy()
labelled['label'] = labelled['label'].astype(int)
print(f'labelled={len(labelled):,}  pos_rate={labelled.label.mean():.3f}')

train_part = labelled[labelled['time_step'] <= 28].copy()
valid_part = labelled[labelled['time_step'] >= 29].copy()
print(f'train_part={len(train_part):,} (pos={train_part.label.mean():.3f})  '
      f'valid_part={len(valid_part):,} (pos={valid_part.label.mean():.3f})')

pos_count = int(train_part['label'].sum())
neg_count = len(train_part) - pos_count
scale_pos = neg_count / max(pos_count, 1)

def make_sample_weights(labels):
    counts = labels.value_counts()
    total  = len(labels)
    w = {int(c): total / (2 * n) for c, n in counts.items()}
    return labels.map(w).to_numpy()

def evaluate(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        'pr_auc':    average_precision_score(y_true, y_prob),
        'roc_auc':   roc_auc_score(y_true, y_prob),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall':    recall_score(y_true, y_pred, zero_division=0),
        'f1':        f1_score(y_true, y_pred, zero_division=0),
    }

sample_sub = pd.read_csv(DATA_DIR / 'sample_submission.csv')

def save_submission(prob, name):
    sub = sample_sub.copy()
    sub['target'] = prob
    assert sub['target'].between(0, 1).all(), f"Probabilities out of [0,1] for {name}"
    path = DATA_DIR / f'submission_{name}.csv'
    sub.to_csv(path, index=False)
    print(f'  Saved {path.name}: mean={prob.mean():.4f}')
    return path

all_results = []

# =====================================================================
# Section 7: Phase A — Tabular baselines
# =====================================================================
print("\n" + "=" * 60)
print("SECTION 7: Phase A — Tabular baselines (feat_* only)")
print("=" * 60)

classifiers_A = {
    'A1_LogReg': Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(class_weight='balanced', max_iter=1000, C=0.1, random_state=42)),
    ]),
    'A2_HistGBT': HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.08, max_leaf_nodes=31,
        l2_regularization=1.0, random_state=42,
    ),
    'A3_XGBoost': xgb.XGBClassifier(
        n_estimators=200, learning_rate=0.08, max_depth=6,
        scale_pos_weight=scale_pos, eval_metric='aucpr',
        tree_method='hist', random_state=42, verbosity=0,
    ),
    'A4_LightGBM': lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.08, num_leaves=31,
        scale_pos_weight=scale_pos, random_state=42, verbose=-1,
    ),
}

for name, clf in classifiers_A.items():
    fit_kw = {}
    if 'HistGBT' in name:
        fit_kw['sample_weight'] = make_sample_weights(train_part['label'])
    clf.fit(train_part[feature_cols], train_part['label'], **fit_kw)
    prob = clf.predict_proba(valid_part[feature_cols])[:, 1]
    row = {'experiment': name, 'phase': 'A', **evaluate(valid_part['label'], prob)}
    all_results.append(row)
    print(f"{name}: PR-AUC={row['pr_auc']:.4f}  ROC-AUC={row['roc_auc']:.4f}")

# =====================================================================
# Section 8: Phase B — Graph feature engineering
# =====================================================================
print("\n" + "=" * 60)
print("SECTION 8: Phase B — Degree + Neighbor features")
print("=" * 60)

# B1: Degree features
in_deg  = edges['txId2'].value_counts().rename('in_degree')
out_deg = edges['txId1'].value_counts().rename('out_degree')
degree_df = pd.concat([in_deg, out_deg], axis=1).fillna(0)
degree_df['total_degree'] = degree_df['in_degree'] + degree_df['out_degree']
for c in list(degree_df.columns):
    degree_df[f'log_{c}'] = np.log1p(degree_df[c])
degree_df.index = degree_df.index.astype(str)

def attach_degree(df):
    mapped = degree_df.reindex(df['txId'].astype(str)).fillna(0)
    mapped.index = df.index
    return pd.concat([df, mapped], axis=1)

train = attach_degree(train)
test  = attach_degree(test)
degree_cols = list(degree_df.columns)
print(f'Degree columns ({len(degree_cols)}): {degree_cols}')

# B2: Neighbor feature aggregation
top_k = 30
sel_X = train_part[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
sel_scores, _ = f_classif(sel_X, train_part['label'])
top_k_feats = [feature_cols[i] for i in np.argsort(np.nan_to_num(sel_scores, nan=-np.inf))[-top_k:]]
print(f'Top-{top_k} features for neighbor aggregation selected')

node_lookup = pd.concat([
    train[['txId'] + top_k_feats],
    test[['txId'] + top_k_feats],
], ignore_index=True)
node_lookup['txId'] = node_lookup['txId'].astype(str)
node_lookup = node_lookup.drop_duplicates('txId').set_index('txId')

edges_str = edges.copy()
edges_str['txId1'] = edges_str['txId1'].astype(str)
edges_str['txId2'] = edges_str['txId2'].astype(str)

print("Computing predecessor aggregation...")
src_vals = edges_str[['txId1', 'txId2']].join(node_lookup, on='txId1')
pred_agg = src_vals.groupby('txId2')[top_k_feats].agg(['mean', 'std', 'min', 'max'])
pred_agg.columns = [f'pred_{s}_{f}' for f, s in pred_agg.columns]
pred_agg = pred_agg.fillna(0)

print("Computing successor aggregation...")
tgt_vals = edges_str[['txId1', 'txId2']].join(node_lookup, on='txId2')
succ_agg = tgt_vals.groupby('txId1')[top_k_feats].agg(['mean', 'std', 'min', 'max'])
succ_agg.columns = [f'succ_{s}_{f}' for f, s in succ_agg.columns]
succ_agg = succ_agg.fillna(0)

neighbor_df = pred_agg.join(succ_agg, how='outer').fillna(0)
neighbor_cols = list(neighbor_df.columns)
print(f'Neighbor columns: {len(neighbor_cols)}')

def attach_neighbor(df):
    mapped = neighbor_df.reindex(df['txId'].astype(str)).fillna(0)
    mapped.index = df.index
    return pd.concat([df, mapped], axis=1)

train = attach_neighbor(train)
test  = attach_neighbor(test)

# Refresh views
labelled   = train.loc[labelled.index]
train_part = train.loc[train_part.index]
valid_part = train.loc[valid_part.index]
print(f'train shape now: {train.shape}')

# Phase B ablation
feature_groups = {
    'B1_feat_only':            feature_cols,
    'B2_feat+degree':          feature_cols + degree_cols,
    'B3_feat+neighbor':        feature_cols + neighbor_cols,
    'B4_feat+degree+neighbor': feature_cols + degree_cols + neighbor_cols,
}

best_B_prauc = 0
best_B_name  = None

for name, cols in feature_groups.items():
    clf = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.08, num_leaves=31,
        scale_pos_weight=scale_pos, random_state=42, verbose=-1,
    )
    clf.fit(train_part[cols], train_part['label'])
    prob = clf.predict_proba(valid_part[cols])[:, 1]
    row  = {'experiment': name, 'phase': 'B', **evaluate(valid_part['label'], prob)}
    all_results.append(row)
    if row['pr_auc'] > best_B_prauc:
        best_B_prauc = row['pr_auc']
        best_B_name  = name
    print(f"{name}: PR-AUC={row['pr_auc']:.4f}  ROC-AUC={row['roc_auc']:.4f}")

best_cols = feature_groups[best_B_name]
print(f'\nBest Phase B feature set: {best_B_name} ({len(best_cols)} features)')

# =====================================================================
# Section 9: Multi-model comparison and ensemble
# =====================================================================
print("\n" + "=" * 60)
print("SECTION 9: Multi-model on best feature set + ensemble")
print("=" * 60)

classifiers_multi = {
    'LogReg':   Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(class_weight='balanced', max_iter=1000, C=0.1, random_state=42)),
    ]),
    'HistGBT':  HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.08, max_leaf_nodes=31,
        l2_regularization=1.0, random_state=42,
    ),
    'XGBoost':  xgb.XGBClassifier(
        n_estimators=200, learning_rate=0.08, max_depth=6,
        scale_pos_weight=scale_pos, eval_metric='aucpr',
        tree_method='hist', random_state=42, verbosity=0,
    ),
    'LightGBM': lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.08, num_leaves=31,
        scale_pos_weight=scale_pos, random_state=42, verbose=-1,
    ),
}

trained_multi = {}
valid_probs   = {}

for name, clf in classifiers_multi.items():
    fit_kw = {}
    if name == 'HistGBT':
        fit_kw['sample_weight'] = make_sample_weights(train_part['label'])
    clf.fit(train_part[best_cols], train_part['label'], **fit_kw)
    prob = clf.predict_proba(valid_part[best_cols])[:, 1]
    row  = {'experiment': f'Multi_{name}', 'phase': 'A+B', **evaluate(valid_part['label'], prob)}
    all_results.append(row)
    trained_multi[name] = clf
    valid_probs[name]   = prob
    print(f"{name}: PR-AUC={row['pr_auc']:.4f}")

tree_names = ['HistGBT', 'XGBoost', 'LightGBM']
ens_prob = np.mean([valid_probs[n] for n in tree_names], axis=0)
ens_row  = {'experiment': 'Ensemble_avg', 'phase': 'A+B', **evaluate(valid_part['label'], ens_prob)}
all_results.append(ens_row)
print(f"Ensemble_avg: PR-AUC={ens_row['pr_auc']:.4f}")

# =====================================================================
# Section 10: Optuna tuning
# =====================================================================
print("\n" + "=" * 60)
print("SECTION 10: Optuna LightGBM tuning (60 trials)")
print("=" * 60)

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    params = {
        'n_estimators':      trial.suggest_int('n_estimators', 100, 500),
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'num_leaves':        trial.suggest_int('num_leaves', 15, 127),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'reg_alpha':         trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda':        trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'subsample':         trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'scale_pos_weight':  scale_pos,
        'random_state': 42, 'verbose': -1,
    }
    clf = lgb.LGBMClassifier(**params)
    clf.fit(train_part[best_cols], train_part['label'])
    prob = clf.predict_proba(valid_part[best_cols])[:, 1]
    return average_precision_score(valid_part['label'], prob)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=60)

print(f'Best PR-AUC: {study.best_value:.6f}')
print('Best params:')
for k, v in study.best_params.items():
    print(f'  {k}: {v}')

tuned_params = {**study.best_params, 'scale_pos_weight': scale_pos, 'random_state': 42, 'verbose': -1}
tuned_lgbm = lgb.LGBMClassifier(**tuned_params)
tuned_lgbm.fit(train_part[best_cols], train_part['label'])
tuned_prob = tuned_lgbm.predict_proba(valid_part[best_cols])[:, 1]
tuned_eval = {'experiment': 'Tuned_LightGBM', 'phase': 'A+B', **evaluate(valid_part['label'], tuned_prob)}
all_results.append(tuned_eval)
print(f"\nTuned LightGBM: PR-AUC={tuned_eval['pr_auc']:.4f}  ROC-AUC={tuned_eval['roc_auc']:.4f}")

# =====================================================================
# Section 11: Rolling time-based CV
# =====================================================================
print("\n" + "=" * 60)
print("SECTION 11: Rolling time-based CV (3 folds)")
print("=" * 60)

time_folds = [
    {'train': (1, 20), 'valid': (21, 25)},
    {'train': (1, 25), 'valid': (26, 30)},
    {'train': (1, 30), 'valid': (31, 35)},
]

fold_results = []
for i, fold in enumerate(time_folds):
    tlo, thi = fold['train']
    vlo, vhi = fold['valid']
    f_train = train[
        (train['time_step'] >= tlo) & (train['time_step'] <= thi) & train['label'].notna()
    ].copy()
    f_valid = train[
        (train['time_step'] >= vlo) & (train['time_step'] <= vhi) & train['label'].notna()
    ].copy()
    if len(f_valid) == 0:
        print(f'Fold {i}: no validation data, skipping')
        continue
    f_train['label'] = f_train['label'].astype(int)
    f_valid['label'] = f_valid['label'].astype(int)
    sp = (f_train['label'] == 0).sum() / max((f_train['label'] == 1).sum(), 1)

    clf = lgb.LGBMClassifier(**{**tuned_params, 'scale_pos_weight': sp})
    clf.fit(f_train[best_cols], f_train['label'])
    prob = clf.predict_proba(f_valid[best_cols])[:, 1]
    ev = evaluate(f_valid['label'], prob)
    fold_results.append({
        'fold': i, 'train_time': f'{tlo}-{thi}', 'valid_time': f'{vlo}-{vhi}',
        'train_n': len(f_train), 'valid_n': len(f_valid),
        **ev,
    })
    print(f"Fold {i} ({tlo}-{thi}/{vlo}-{vhi}): PR-AUC={ev['pr_auc']:.4f}  ROC-AUC={ev['roc_auc']:.4f}")

fold_df = pd.DataFrame(fold_results)
print(f'\nMean PR-AUC: {fold_df["pr_auc"].mean():.4f} +/- {fold_df["pr_auc"].std():.4f}')
print(f'Mean ROC-AUC: {fold_df["roc_auc"].mean():.4f} +/- {fold_df["roc_auc"].std():.4f}')

# =====================================================================
# Section 12: Phase C — PCA + Node2Vec
# =====================================================================
print("\n" + "=" * 60)
print("SECTION 12: Phase C — PCA + Node2Vec")
print("=" * 60)

# C1: PCA
print("Computing PCA...")
all_feat = pd.concat([
    train[feature_cols], test[feature_cols],
], ignore_index=True).fillna(0)

scaler_pca = StandardScaler()
all_scaled  = scaler_pca.fit_transform(all_feat)

n_comp = 50
pca = PCA(n_components=n_comp, random_state=42)
pca_all = pca.fit_transform(all_scaled)
print(f'PCA: {n_comp} components explain {pca.explained_variance_ratio_.sum():.3f} of variance')

pca_train = pca_all[:len(train)]
pca_test  = pca_all[len(train):]
pca_cols  = [f'pca_{i}' for i in range(n_comp)]

for i, c in enumerate(pca_cols):
    train[c] = pca_train[:, i]
    test[c]  = pca_test[:, i]

# C2: Node2Vec
print("Computing Node2Vec (this may take a few minutes)...")
import networkx as nx
import random

G = nx.from_pandas_edgelist(edges, source='txId1', target='txId2', create_using=nx.DiGraph())
print(f'Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges')

try:
    from node2vec import Node2Vec
    n2v = Node2Vec(G, dimensions=64, walk_length=20, num_walks=10, p=1.0, q=0.5, workers=1, seed=42)
    n2v_model = n2v.fit(window=5, min_count=1, batch_words=4)
    print("node2vec package used successfully")
except Exception as e:
    print(f"node2vec package failed ({e}), using manual implementation...")
    from gensim.models import Word2Vec

    def node2vec_walk(graph, start, walk_length, p=1.0, q=0.5):
        walk = [start]
        for _ in range(walk_length - 1):
            cur = walk[-1]
            neighbors = list(graph.neighbors(cur)) + list(graph.predecessors(cur))
            if not neighbors:
                break
            if len(walk) == 1:
                walk.append(random.choice(neighbors))
            else:
                prev = walk[-2]
                weights = []
                for n in neighbors:
                    if n == prev:
                        weights.append(1.0 / p)
                    elif graph.has_edge(n, prev) or graph.has_edge(prev, n):
                        weights.append(1.0)
                    else:
                        weights.append(1.0 / q)
                total = sum(weights)
                weights = [w / total for w in weights]
                walk.append(random.choices(neighbors, weights=weights, k=1)[0])
        return [str(n) for n in walk]

    random.seed(42)
    nodes = list(G.nodes())
    walks = []
    for _ in range(10):
        random.shuffle(nodes)
        for node in nodes:
            walks.append(node2vec_walk(G, node, walk_length=20, p=1.0, q=0.5))
    print(f"Generated {len(walks)} walks")

    n2v_model = Word2Vec(walks, vector_size=64, window=5, min_count=1,
                         workers=1, seed=42, epochs=1)
    print("Manual Node2Vec training complete")

n2v_dim  = 64
n2v_cols = [f'n2v_{i}' for i in range(n2v_dim)]

def get_n2v(txids):
    embs = []
    for t in txids.astype(str):
        embs.append(n2v_model.wv[t] if t in n2v_model.wv else np.zeros(n2v_dim))
    return np.array(embs)

train_n2v = get_n2v(train['txId'])
test_n2v  = get_n2v(test['txId'])
cov = np.mean(np.any(train_n2v != 0, axis=1))
print(f'Node2Vec coverage: train={cov:.3f}')

for i, c in enumerate(n2v_cols):
    train[c] = train_n2v[:, i]
    test[c]  = test_n2v[:, i]

# Refresh views
labelled   = train.loc[labelled.index]
train_part = train.loc[train_part.index]
valid_part = train.loc[valid_part.index]
print(f'train shape: {train.shape}')

# Phase C ablation
C_specs = {
    'C1_pca_only':     pca_cols,
    'C2_feat+pca':     feature_cols + pca_cols,
    'C3_best+pca':     best_cols + pca_cols,
    'C4_n2v_only':     n2v_cols,
    'C5_feat+n2v':     feature_cols + n2v_cols,
    'C6_best+n2v':     best_cols + n2v_cols,
    'C7_best+pca+n2v': best_cols + pca_cols + n2v_cols,
}

for name, cols in C_specs.items():
    clf = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.08, num_leaves=31,
        scale_pos_weight=scale_pos, random_state=42, verbose=-1,
    )
    clf.fit(train_part[cols], train_part['label'])
    prob = clf.predict_proba(valid_part[cols])[:, 1]
    row  = {'experiment': name, 'phase': 'C', **evaluate(valid_part['label'], prob)}
    all_results.append(row)
    print(f"{name}: PR-AUC={row['pr_auc']:.4f}")

# =====================================================================
# Section 13: Phase D — Semi-supervised
# =====================================================================
print("\n" + "=" * 60)
print("SECTION 13: Phase D — Pseudo-labeling + Label propagation")
print("=" * 60)

# D1: Pseudo-labeling
teacher = lgb.LGBMClassifier(**{**tuned_params, 'scale_pos_weight': scale_pos})
teacher.fit(train_part[best_cols], train_part['label'])

unlabelled_mask = train['label'].isna()
unlabelled_idx  = train[unlabelled_mask].index
unlabelled_prob = teacher.predict_proba(train.loc[unlabelled_idx, best_cols])[:, 1]

teacher_prob = teacher.predict_proba(valid_part[best_cols])[:, 1]
teacher_eval = evaluate(valid_part['label'], teacher_prob)
all_results.append({'experiment': 'D0_teacher_baseline', 'phase': 'D', **teacher_eval})
print(f"Teacher baseline: PR-AUC={teacher_eval['pr_auc']:.4f}")

for threshold in [0.90, 0.95]:
    high_pos = unlabelled_idx[unlabelled_prob >= threshold]
    high_neg = unlabelled_idx[unlabelled_prob <= (1 - threshold)]
    pseudo = pd.concat([
        pd.Series(1.0, index=high_pos),
        pd.Series(0.0, index=high_neg),
    ])
    exp_idx    = train_part.index.append(pseudo.index)
    exp_labels = pd.concat([train_part['label'], pseudo])

    print(f't={threshold}: +{len(high_pos)} pos, +{len(high_neg)} neg  '
          f'(total {len(exp_idx):,} vs {len(train_part):,})')

    student = lgb.LGBMClassifier(**{**tuned_params, 'scale_pos_weight': scale_pos})
    student.fit(train.loc[exp_idx, best_cols], exp_labels)
    prob = student.predict_proba(valid_part[best_cols])[:, 1]
    row  = {'experiment': f'D1_pseudo_t={threshold}', 'phase': 'D', **evaluate(valid_part['label'], prob)}
    all_results.append(row)
    print(f"  PR-AUC={row['pr_auc']:.4f}")

# D2: Label propagation
print("\nLabel propagation via graph diffusion...")
from scipy.sparse import csr_matrix

all_txids    = pd.concat([train['txId'], test['txId']], ignore_index=True).unique()
txid_to_idx  = {str(t): i for i, t in enumerate(all_txids)}
n_nodes      = len(all_txids)

src_idx = edges['txId1'].astype(str).map(txid_to_idx)
tgt_idx = edges['txId2'].astype(str).map(txid_to_idx)
valid_mask = src_idx.notna() & tgt_idx.notna()
src_arr = src_idx[valid_mask].astype(int).values
tgt_arr = tgt_idx[valid_mask].astype(int).values
rows_sp = np.concatenate([src_arr, tgt_arr])
cols_sp = np.concatenate([tgt_arr, src_arr])
adj = csr_matrix((np.ones(len(rows_sp)), (rows_sp, cols_sp)), shape=(n_nodes, n_nodes))
print(f'Adjacency: {n_nodes} nodes, {adj.nnz} entries')

all_lab = np.full(n_nodes, -1.0)
lab_idx = labelled['txId'].astype(str).map(txid_to_idx)
valid_lab = lab_idx.notna()
all_lab[lab_idx[valid_lab].astype(int).values] = labelled.loc[valid_lab.values, 'label'].astype(int).values

known_mask = all_lab >= 0
soft = np.full(n_nodes, 0.5)
soft[all_lab == 0] = 0.0
soft[all_lab == 1] = 1.0

alpha, n_iter = 0.3, 20
for _ in range(n_iter):
    propagated = adj.dot(soft)
    deg = np.array(adj.sum(axis=1)).flatten()
    deg[deg == 0] = 1
    propagated /= deg
    soft = np.where(known_mask, soft, alpha * propagated + (1 - alpha) * soft)

print(f'After {n_iter} iterations:')
print(f'  Unknown mean={soft[~known_mask].mean():.4f}, std={soft[~known_mask].std():.4f}')

train['label_prop'] = train['txId'].astype(str).map(txid_to_idx).map(lambda i: soft[int(i)] if pd.notna(i) else 0.5).values
test['label_prop']  = test['txId'].astype(str).map(txid_to_idx).map(lambda i: soft[int(i)] if pd.notna(i) else 0.5).values

# Refresh views
labelled   = train.loc[labelled.index]
train_part = train.loc[train_part.index]
valid_part = train.loc[valid_part.index]

lp_specs = {
    'D2_best_cols':      best_cols,
    'D2_best+labelprop': best_cols + ['label_prop'],
}
for name, cols in lp_specs.items():
    clf = lgb.LGBMClassifier(**{**tuned_params, 'scale_pos_weight': scale_pos})
    clf.fit(train_part[cols], train_part['label'])
    prob = clf.predict_proba(valid_part[cols])[:, 1]
    row  = {'experiment': name, 'phase': 'D', **evaluate(valid_part['label'], prob)}
    all_results.append(row)
    print(f"{name}: PR-AUC={row['pr_auc']:.4f}")

# =====================================================================
# Section 14: Master summary
# =====================================================================
print("\n" + "=" * 60)
print("SECTION 14: Master summary")
print("=" * 60)

all_exp = pd.DataFrame(all_results).sort_values('pr_auc', ascending=False).reset_index(drop=True)
print(all_exp.to_string())

best_exp = all_exp.iloc[0]
print(f"\nBest: {best_exp['experiment']}  PR-AUC={best_exp['pr_auc']:.6f}")

# =====================================================================
# Section 15: Final submissions
# =====================================================================
print("\n" + "=" * 60)
print("SECTION 15: Generating final submissions")
print("=" * 60)

full_sp = (labelled['label'] == 0).sum() / max((labelled['label'] == 1).sum(), 1)
full_params = {**tuned_params, 'scale_pos_weight': full_sp}

# 1) Tuned LightGBM
m1 = lgb.LGBMClassifier(**full_params)
m1.fit(labelled[best_cols], labelled['label'])
p1 = m1.predict_proba(test[best_cols])[:, 1]
print('1) Tuned LightGBM')
save_submission(p1, 'tuned_lgbm')

# 2) Ensemble: tuned LGBM + XGB + HistGBT
m2_xgb = xgb.XGBClassifier(
    n_estimators=200, learning_rate=0.08, max_depth=6,
    scale_pos_weight=full_sp, eval_metric='aucpr',
    tree_method='hist', random_state=42, verbosity=0,
)
m2_xgb.fit(labelled[best_cols], labelled['label'])
p2_xgb = m2_xgb.predict_proba(test[best_cols])[:, 1]

m2_hgbt = HistGradientBoostingClassifier(
    max_iter=200, learning_rate=0.08, max_leaf_nodes=31,
    l2_regularization=1.0, random_state=42,
)
m2_hgbt.fit(labelled[best_cols], labelled['label'],
            sample_weight=make_sample_weights(labelled['label']))
p2_hgbt = m2_hgbt.predict_proba(test[best_cols])[:, 1]

p2 = (p1 + p2_xgb + p2_hgbt) / 3
print('2) Ensemble (LGBM+XGB+HistGBT)')
save_submission(p2, 'ensemble')

# 3) Kitchen sink: best_cols + PCA + N2V + label_prop
kitchen_cols = best_cols + pca_cols + n2v_cols + ['label_prop']
m3 = lgb.LGBMClassifier(**full_params)
m3.fit(labelled[kitchen_cols], labelled['label'])
p3 = m3.predict_proba(test[kitchen_cols])[:, 1]
print('3) Kitchen sink')
save_submission(p3, 'kitchen_sink')

# 4) Kitchen sink ensemble
m4_xgb = xgb.XGBClassifier(
    n_estimators=200, learning_rate=0.08, max_depth=6,
    scale_pos_weight=full_sp, eval_metric='aucpr',
    tree_method='hist', random_state=42, verbosity=0,
)
m4_xgb.fit(labelled[kitchen_cols], labelled['label'])
p4_xgb = m4_xgb.predict_proba(test[kitchen_cols])[:, 1]

m4_hgbt = HistGradientBoostingClassifier(
    max_iter=200, learning_rate=0.08, max_leaf_nodes=31,
    l2_regularization=1.0, random_state=42,
)
m4_hgbt.fit(labelled[kitchen_cols], labelled['label'],
            sample_weight=make_sample_weights(labelled['label']))
p4_hgbt = m4_hgbt.predict_proba(test[kitchen_cols])[:, 1]

p4 = (p3 + p4_xgb + p4_hgbt) / 3
print('4) Kitchen sink ensemble')
save_submission(p4, 'kitchen_sink_ensemble')

print("\n" + "=" * 60)
print("ALL DONE. Submissions generated:")
print("=" * 60)
import glob
for f in sorted(glob.glob(str(DATA_DIR / 'submission_*.csv'))):
    print(f'  {Path(f).name}')
