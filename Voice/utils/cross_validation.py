import numpy as np
from sklearn.model_selection import StratifiedKFold
from ensemble import StackingEnsemble


def cross_validation_ensemble(audio_paths, labels, cv_folds=5):
    """교차 검증을 통한 앙상블 성능 평가"""
    print("=== Cross Validation ===")
    
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_results = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(audio_paths, labels)):
        print(f"\nFold {fold + 1}/{cv_folds}")
        
        train_paths = audio_paths[train_idx]
        test_paths = audio_paths[test_idx]
        train_labels = labels[train_idx]
        test_labels = labels[test_idx]
        
        ensemble = StackingEnsemble(device='cuda')
        
        base_predictions = ensemble.train_base_models(train_paths, train_labels, epochs=20)
        
        ensemble.train_meta_learners(base_predictions, train_labels)
        
        from evaluation import ModelEvaluator
        evaluator = ModelEvaluator()
        results = evaluator.evaluate_models(ensemble, test_paths, test_labels, save_plots=False)
        cv_results.append(results)
    
    print("\n=== Cross Validation Summary ===")
    
    base_avg = {}
    for model_name in ['cnn', 'rnn', 'transformer', 'hybrid']:
        base_avg[model_name] = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc']:
            scores = [result['base_metrics'][model_name][metric] for result in cv_results]
            base_avg[model_name][metric] = {
                'mean': np.mean(scores),
                'std': np.std(scores)
            }
    
    meta_avg = {}
    for model_name in ['xgb', 'rf', 'gb']:
        meta_avg[model_name] = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc']:
            scores = [result['meta_metrics'][model_name][metric] for result in cv_results]
            meta_avg[model_name][metric] = {
                'mean': np.mean(scores),
                'std': np.std(scores)
            }
    
    return base_avg, meta_avg