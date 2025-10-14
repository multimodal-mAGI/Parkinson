"""
shap_analysis.py

SHAP 기반 모델 해석 모듈
- Tree 기반 모델 (RandomForest) 전역/국소적 해석
- Summary, Dependence, Waterfall, Force Plot 시각화
- SHAP 값 CSV 저장
- Global Feature Importance 계산
- SHAP Clustering Force Plot 생성
"""

import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram


def shap_analysis(model, X_test, feature_names):
    """
    SHAP Explainer 생성 및 전역/국소 해석 수행

    Parameters
    ----------
    model : estimator
        학습된 RandomForest 모델
    X_test : pd.DataFrame
        테스트용 입력 데이터
    feature_names : list
        특징 이름 리스트

    Returns
    -------
    explainer : shap.TreeExplainer
        SHAP Explainer 객체
    shap_values_class1 : np.ndarray
        클래스 1 SHAP 값
    expected_value_class1_scalar : float
        클래스 1 기대값 (base value)
    """

    print("=== [1] SHAP Explainer 객체 생성 및 SHAP 값 계산 ===")
    explainer = shap.TreeExplainer(model)
    shap_values_list = explainer.shap_values(X_test)

    # 클래스 1 SHAP 값 추출
    if shap_values_list.ndim == 3 and shap_values_list.shape[2] == 2:
        shap_values_class1 = shap_values_list[:, :, 1]
    else:
        shap_values_class1 = shap_values_list

    # 클래스 1 기대값
    if isinstance(explainer.expected_value, (list, np.ndarray)):
        expected_value_class1_scalar = float(explainer.expected_value[1])
    else:
        expected_value_class1_scalar = float(explainer.expected_value)

    # ----------------------------
    # 전역 해석 (Global Interpretation)
    # ----------------------------
    print("\n=== [2] SHAP 전역적 해석 ===")
    # Summary Plot
    shap.summary_plot(shap_values_class1, X_test, feature_names=feature_names, show=False)
    plt.savefig("shap_summary_plot.png", bbox_inches='tight')
    plt.show()

    # SHAP 값 CSV 저장
    shap_df = pd.DataFrame(shap_values_class1, columns=X_test.columns)
    shap_df.to_csv("shap_values.csv", index=False)
    print("[SHAP] shap_values.csv 저장 완료")

    # Dependence Plot (첫 번째 피처)
    first_feature = X_test.columns[0]
    print(f"[SHAP] Dependence Plot ({first_feature})")
    shap.dependence_plot(first_feature, shap_values_class1, X_test, show=False)
    plt.savefig("shap_dependence_plot.png", bbox_inches='tight')
    plt.show()

    # Global Feature Importance
    global_importance = np.mean(np.abs(shap_values_class1), axis=0)
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': global_importance
    }).sort_values(by='importance', ascending=False)
    print("[SHAP] Global Feature Importance:")
    print(feature_importance_df)

    shap.summary_plot(shap_values_class1, X_test, plot_type="bar", show=False)
    plt.savefig("shap_summary_plot_bar.png", bbox_inches='tight')
    plt.show()

    # ----------------------------
    # 국소 해석 (Local Interpretation)
    # ----------------------------
    print("\n=== [3] 국소적 해석 (샘플 0 기준) ===")
    sample_index = 0
    sample_data = X_test.iloc[[sample_index]]
    sample_shap_values = shap_values_class1[sample_index]

    # Waterfall Plot
    shap.plots.waterfall(shap.Explanation(
        values=sample_shap_values,
        base_values=expected_value_class1_scalar,
        data=sample_data.values[0],
        feature_names=feature_names
    ), show=False)
    plt.savefig("shap_waterfall_plot.png", bbox_inches='tight')
    plt.show()

    # Force Plot (HTML)
    force_plot_html = shap.plots.force(
        base_value=expected_value_class1_scalar,
        shap_values=sample_shap_values,
        features=sample_data,
        matplotlib=False,
        show=False
    )
    shap.save_html("force_plot.html", force_plot_html)
    print("[SHAP] Force Plot 저장 완료 (force_plot.html)")

    # ----------------------------
    # SHAP Clustering Force Plot (Interactive)
    # ----------------------------
    clustering_linkage = linkage(shap_values_class1, method='ward')
    dendrogram_order = dendrogram(clustering_linkage, no_plot=True)['leaves']
    sorted_shap_values = shap_values_class1[dendrogram_order]

    clustered_force_html = shap.force_plot(
        expected_value_class1_scalar,
        sorted_shap_values,
        X_test.iloc[dendrogram_order]
    )
    shap.save_html("clustered_shap_force_plot.html", clustered_force_html)
    print("[SHAP] Clustering Force Plot 저장 완료 (clustered_shap_force_plot.html)")

    print("=== SHAP 분석 완료 ===")
    return explainer, shap_values_class1, expected_value_class1_scalar


def shap_analysis_node(model, X_test, feature_names, top_trees: int = 5):
    """
    SHAP Explainer + 트리 노드 정보 확인 및 전역/국소적 해석 수행

    Parameters
    ----------
    model : estimator
        학습된 RandomForest 모델
    X_test : pd.DataFrame
        테스트용 입력 데이터
    feature_names : list
        특징 이름 리스트
    top_trees : int
        출력할 상위 트리 수 (기본값: 5)

    Returns
    -------
    explainer : shap.TreeExplainer
    shap_values_class1 : np.ndarray
    expected_value_class1_scalar : float
    """

    # ----------------------------
    # RandomForest 트리 정보 확인
    # ----------------------------
    print("=== [1] Random Forest 상위 트리 정보 확인 ===")
    for i, tree in enumerate(model.estimators_[:top_trees]):
        print(f"Tree {i+1}/{top_trees} - depth: {tree.get_depth()}, nodes: {tree.tree_.node_count}")
    print(f"상위 {top_trees}개 트리 정보 출력 완료\n")

    # SHAP 분석 수행
    return shap_analysis(model, X_test, feature_names)
