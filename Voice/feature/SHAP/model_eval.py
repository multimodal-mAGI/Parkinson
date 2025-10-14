"""
model_eval.py

모델 평가 모듈
- 분류 성능 리포트 저장 (Classification Report)
- 혼동 행렬 시각화 및 저장
- ROC Curve 시각화 및 저장
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    RocCurveDisplay,
    ConfusionMatrixDisplay
)


def evaluate_model(model, X_test, y_test):
    """
    학습된 모델 평가 및 시각화

    Parameters
    ----------
    model : estimator
        학습된 분류 모델 (predict, predict_proba 가능)
    X_test : pd.DataFrame
        테스트 입력 데이터
    y_test : pd.Series
        테스트 레이블

    Returns
    -------
    None
        평가 결과 파일로 저장 및 시각화
    """

    # 1️⃣ 예측
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # 2️⃣ Classification Report 저장
    report = classification_report(y_test, y_pred, output_dict=True)
    pd.DataFrame(report).to_csv("classification_report.csv", index=True)
    print("[EVAL] Classification report saved (classification_report.csv)")

    # 3️⃣ Confusion Matrix 시각화 및 저장
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.show()
    print("[EVAL] Confusion matrix saved (confusion_matrix.png)")

    # 4️⃣ ROC Curve 시각화 및 저장
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title("ROC Curve")
    plt.savefig("roc_curve.png", dpi=300)
    plt.show()
    print("[EVAL] ROC curve saved (roc_curve.png)")
