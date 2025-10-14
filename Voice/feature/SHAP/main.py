from data_preprocess import load_and_preprocess
from model_train import train_kfold
from model_eval import evaluate_model
from shap_analysis import shap_analysis, shap_analysis_node

# 1. 데이터
X, y = load_and_preprocess("voice_features.csv")

# 2. 학습 (k-fold)
model, metrics_df = train_kfold(X, y, n_splits=5)

# 3. 평가
evaluate_model(model, X, y)

# 4. SHAP 분석
explainer, shap_values_class1, expected_value_class1 = shap_analysis_node(model, X, X.columns.tolist())
