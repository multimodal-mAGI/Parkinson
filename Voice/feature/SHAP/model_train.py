"""
model_train.py

RandomForest 기반 K-Fold 교차 검증 학습 모듈
- Stratified K-Fold로 데이터 분할
- Tree-wise ROC-AUC 추적 (warm_start)
- Fold별 Accuracy, ROC-AUC 기록
- Fold별 트리 수 대비 ROC-AUC 시각화 저장
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier


def train_kfold(X, y, n_splits: int = 5, n_estimators: int = 200, random_state: int = 42):
    """
    Stratified K-Fold로 RandomForest 학습 및 성능 기록

    Parameters
    ----------
    X : pd.DataFrame
        입력 특징 데이터
    y : pd.Series
        레이블
    n_splits : int
        K-Fold 분할 수 (기본값: 5)
    n_estimators : int
        RandomForest 트리 수 (기본값: 200)
    random_state : int
        시드 (기본값: 42)

    Returns
    -------
    model : RandomForestClassifier
        마지막 Fold 학습된 모델
    metrics_df : pd.DataFrame
        Fold별 Accuracy와 ROC-AUC 기록
    """

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        print(f"[TRAIN] Fold {fold}/{n_splits}")

        # 데이터 분할
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # RandomForest 모델 초기화 (warm_start=True)
        model = RandomForestClassifier(
            n_estimators=1,
            warm_start=True,
            random_state=random_state
        )

        tree_roc_auc = []

        # 트리별 학습 및 ROC-AUC 추적
        for i in range(1, n_estimators + 1):
            model.n_estimators = i
            model.fit(X_train, y_train)

            y_proba = model.predict_proba(X_test)[:, 1]
            roc = roc_auc_score(y_test, y_proba)
            tree_roc_auc.append(roc)

            if i % 20 == 0 or i == 1:
                print(f"  Tree {i}/{n_estimators} - ROC AUC: {roc:.4f}")

        # Fold별 최종 성능 기록
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        roc_final = tree_roc_auc[-1]
        fold_metrics.append({'fold': fold, 'accuracy': acc, 'roc_auc': roc_final})
        print(f"  Fold {fold} final - Accuracy: {acc:.4f}, ROC AUC: {roc_final:.4f}")

        # Fold별 트리 수 대비 ROC-AUC 시각화 저장
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, n_estimators + 1), tree_roc_auc, label=f"Fold {fold} ROC-AUC")
        plt.xlabel("Number of Trees")
        plt.ylabel("ROC-AUC")
        plt.title(f"Fold {fold} Tree-wise ROC-AUC")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"rf_fold{fold}_treewise_roc_auc.png", dpi=300)
        plt.close()

    # 전체 Fold metrics 저장
    metrics_df = pd.DataFrame(fold_metrics)
    metrics_df.to_csv("kfold_metrics.csv", index=False)
    print("[TRAIN] K-Fold 학습 완료, metrics 저장됨 (kfold_metrics.csv)")

    return model, metrics_df
