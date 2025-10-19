from typing import Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import log_loss
from catboost import CatBoostClassifier, Pool

from config import CFG
from features import feature_engineer
from calibration import softmax, find_temperature


def train_and_predict(train: pd.DataFrame, test: pd.DataFrame, sample: Optional[pd.DataFrame]) -> pd.DataFrame:
    target_col = "OutcomeType"
    if target_col not in train.columns:
        raise KeyError("`OutcomeType` not found in train.csv — please ensure you are using the Kaggle dataset.")

    y_raw = train[target_col].astype(str)

    if sample is not None:
        label_order = list(sample.columns[1:])
    else:
        label_order = sorted(y_raw.unique().tolist())

    label2id = {lbl: i for i, lbl in enumerate(label_order)}
    id2label = {i: lbl for lbl, i in label2id.items()}

    unseen = [lbl for lbl in y_raw.unique() if lbl not in label2id]
    if unseen:
        start = len(label2id)
        for k, lbl in enumerate(unseen):
            label2id[lbl] = start + k
            id2label[start + k] = lbl
            label_order.append(lbl)

    y = y_raw.map(label2id).values
    n_classes = len(label_order)

    train_feats, cat_cols, num_cols, train_id_col = feature_engineer(train, is_train=True)
    test_feats, _, _, test_id_col = feature_engineer(test, is_train=False)

    all_cols = sorted(set(train_feats.columns) | set(test_feats.columns))
    train_feats = train_feats.reindex(columns=all_cols)
    test_feats = test_feats.reindex(columns=all_cols)

    # Drop identifier columns on both sides
    drop_ids = [c for c in ["ID", "AnimalID"] if c in train_feats.columns or c in test_feats.columns]
    X = train_feats.drop(columns=[c for c in drop_ids if c in train_feats.columns]).copy()
    X_test = test_feats.drop(columns=[c for c in drop_ids if c in test_feats.columns]).copy()

    # Categorical feature indices relative to X columns
    cat_features = [X.columns.get_loc(c) for c in X.columns if c in cat_cols]

    # Rare-category bucketing on combined train+test (feature-only, no target)
    for c in cat_cols:
        if c in X.columns:
            s = pd.concat([X[c].astype("object"), X_test[c].astype("object")], axis=0)
            vc = s.value_counts(dropna=False)
            rare_vals = set(vc[vc < CFG.rare_threshold].index)
            if rare_vals:
                X[c] = X[c].apply(lambda v: "Other" if v in rare_vals else v)
                X_test[c] = X_test[c].apply(lambda v: "Other" if v in rare_vals else v)

    # Numeric fill
    for c in num_cols:
        if c in X.columns:
            med = pd.concat([X[c], X_test[c]], axis=0).median()
            X[c] = X[c].fillna(med)
            X_test[c] = X_test[c].fillna(med)

    # Categorical fill and coercion to string
    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].astype("object").fillna("").astype(str)
            X_test[c] = X_test[c].astype("object").fillna("").astype(str)

    if CFG.cv_strategy == "time":
        dt = pd.to_datetime(train.get("DateTime"), errors="coerce")
        groups = (dt.dt.year.fillna(0).astype(int) * 100 + dt.dt.month.fillna(0).astype(int)).values
        splitter = StratifiedGroupKFold(n_splits=CFG.n_splits)
    else:
        groups = None
        splitter = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.random_state)
    oof_logits_accum = np.zeros((len(X), n_classes), dtype=float)
    test_logits_seeds = []

    class_counts = np.bincount(y, minlength=n_classes).astype(float)
    manual_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
    manual_weights = (manual_weights / manual_weights.mean()).tolist()

    for seed in CFG.seeds:
        test_logits_folds = []
        # Build a fresh split iterator for each seed (sklearn splitters return one-shot generators)
        if CFG.cv_strategy == "time":
            split_iter = splitter.split(X, y, groups=groups)
        else:
            split_iter = splitter.split(X, y)
        for fold, (trn_idx, val_idx) in enumerate(split_iter, 1):
            X_tr, X_val = X.iloc[trn_idx], X.iloc[val_idx]
            y_tr, y_val = y[trn_idx], y[val_idx]

            train_pool = Pool(X_tr, y_tr, cat_features=cat_features)
            valid_pool = Pool(X_val, y_val, cat_features=cat_features)

            params = dict(
                loss_function="MultiClass",
                eval_metric="MultiClass",
                iterations=CFG.iterations,
                learning_rate=CFG.learning_rate,
                depth=CFG.depth,
                l2_leaf_reg=CFG.l2_leaf_reg,
                random_seed=seed + fold,
                early_stopping_rounds=CFG.early_stopping_rounds,
                verbose=CFG.verbose_eval,
                bootstrap_type="Bayesian",
                bagging_temperature=CFG.bagging_temperature,
                random_strength=CFG.random_strength,
                one_hot_max_size=CFG.one_hot_max_size,
            )
            if CFG.auto_class_weights:
                params["auto_class_weights"] = CFG.auto_class_weights
            else:
                params["class_weights"] = manual_weights

            model = CatBoostClassifier(**params)
            model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

            val_logits = model.predict(valid_pool, prediction_type="RawFormulaVal")

            T = 1.0
            if CFG.use_temperature_scaling:
                T = find_temperature(
                    logits_val=val_logits,
                    y_val=y_val,
                    tmin=CFG.temp_grid[0],
                    tmax=CFG.temp_grid[1],
                    steps=CFG.temp_grid[2],
                )

            oof_logits_accum[val_idx] += val_logits / T

            test_pool = Pool(X_test, cat_features=cat_features)
            fold_test_logits = model.predict(test_pool, prediction_type="RawFormulaVal")
            test_logits_folds.append(fold_test_logits / T)

            val_probs = softmax(val_logits / T, axis=1)
            fold_loss = log_loss(y_val, val_probs, labels=np.arange(n_classes))
            print(f"[Seed {seed}] [Fold {fold}] log loss (calibrated): {fold_loss:.6f} | Temperature: {T:.3f}")

        test_logits_seed = np.mean(np.stack(test_logits_folds, axis=0), axis=0)
        test_logits_seeds.append(test_logits_seed)

    oof_logits_mean = oof_logits_accum / max(len(CFG.seeds), 1)
    if CFG.use_temperature_scaling:
        T_oof = find_temperature(oof_logits_mean, y, *CFG.temp_grid)
        oof_loss = log_loss(y, softmax(oof_logits_mean / T_oof, axis=1), labels=np.arange(n_classes))
        print(f"[OOF] log loss (global calibrated): {oof_loss:.6f} | Temperature: {T_oof:.3f}")
    else:
        oof_loss = log_loss(y, softmax(oof_logits_mean, axis=1), labels=np.arange(n_classes))
        print(f"[OOF] log loss: {oof_loss:.6f}")

    test_logits = np.mean(np.stack(test_logits_seeds, axis=0), axis=0)
    test_probs = softmax(test_logits, axis=1)

    key = "ID" if "ID" in test.columns else ("AnimalID" if "AnimalID" in test.columns else None)
    if key is None:
        key = test_feats.columns[-1]
    submission = pd.DataFrame(test[key].values, columns=["ID"])
    for i, cls in enumerate(label_order):
        submission[cls] = test_probs[:, i]
    return submission
