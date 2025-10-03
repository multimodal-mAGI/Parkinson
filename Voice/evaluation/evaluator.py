import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from .visualizer import ModelVisualizer


class ModelEvaluator:
    """모델 평가 클래스"""
    
    def __init__(self):
        self.visualizer = ModelVisualizer()
    
    def calculate_metrics(self, y_true, y_pred, y_proba):
        """성능 메트릭 계산"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'auc': roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) == 2 else 0
        }
    
    def evaluate_models(self, ensemble, test_audio_paths, test_labels, save_plots=True, plots_dir=None):
        """전체 모델 성능 평가 + 시각화 및 저장"""
        print("\n=== Model Evaluation ===")
        
        final_predictions, base_predictions = ensemble.predict(test_audio_paths)
        
        print("\n--- Base Models Performance ---")
        base_metrics = {}
        
        for name, predictions in base_predictions.items():
            pred_classes = np.argmax(predictions, axis=1)
            pred_proba = predictions[:, 1]
            
            metrics = self.calculate_metrics(test_labels, pred_classes, pred_proba)
            base_metrics[name] = metrics
            
            print(f"\n{name.upper()} Model:")
            for metric, value in metrics.items():
                print(f"  {metric.capitalize()}: {value:.4f}")
        
        print("\n--- Meta Learners Performance ---")
        meta_metrics = {}
        
        for name, results in final_predictions.items():
            pred_classes = results['predictions']
            pred_proba = results['probabilities'][:, 1]
            
            metrics = self.calculate_metrics(test_labels, pred_classes, pred_proba)
            meta_metrics[name] = metrics
            
            print(f"\n{name.upper()} Meta Learner:")
            for metric, value in metrics.items():
                print(f"  {metric.capitalize()}: {value:.4f}")
        
        best_base = max(base_metrics.items(), key=lambda x: x[1]['auc'])
        best_meta = max(meta_metrics.items(), key=lambda x: x[1]['auc'])
        
        print(f"\n--- Best Models ---")
        print(f"Best Base Model: {best_base[0].upper()} (AUC: {best_base[1]['auc']:.4f})")
        print(f"Best Meta Learner: {best_meta[0].upper()} (AUC: {best_meta[1]['auc']:.4f})")
        
        plot_paths = {}
        if save_plots:
            if plots_dir is None:
                plots_dir = "./evaluation_plots"
            
            os.makedirs(plots_dir, exist_ok=True)
            print(f"\n=== Generating and Saving Plots to {plots_dir} ===")
            
            plot_paths = self._generate_all_plots(
                base_predictions, final_predictions, base_metrics, meta_metrics,
                test_labels, plots_dir, ensemble.training_history
            )
            
            print(f"All plots and metrics saved to: {plots_dir}")
            print("Generated files:")
            for name, path in plot_paths.items():
                print(f"  - {name}: {os.path.basename(path)}")
        
        return {
            'base_metrics': base_metrics,
            'meta_metrics': meta_metrics,
            'best_base': best_base,
            'best_meta': best_meta,
            'plot_paths': plot_paths if save_plots else {},
            'base_predictions': base_predictions,
            'meta_predictions': final_predictions
        }
    
    def _generate_all_plots(self, base_predictions, final_predictions, base_metrics, meta_metrics, 
                           test_labels, plots_dir, training_history):
        """모든 플롯 생성"""
        plot_paths = {}
        
        print("Generating confusion matrices...")
        confusion_matrices = {}
        
        # 베이스 모델 confusion matrices
        for name, predictions in base_predictions.items():
            pred_classes = np.argmax(predictions, axis=1)
            cm_path, cm_data = self.visualizer.plot_confusion_matrix(
                test_labels, pred_classes, f'{name}_base', plots_dir
            )
            plot_paths[f'confusion_matrix_{name}_base'] = cm_path
            confusion_matrices[f'{name}_base'] = cm_data
        
        # 메타 러너 confusion matrices
        for name, results in final_predictions.items():
            pred_classes = results['predictions']
            cm_path, cm_data = self.visualizer.plot_confusion_matrix(
                test_labels, pred_classes, f'{name}_meta', plots_dir
            )
            plot_paths[f'confusion_matrix_{name}_meta'] = cm_path
            confusion_matrices[f'{name}_meta'] = cm_data
        
        print("Generating ROC curves...")
        roc_base_path, roc_base_data = self.visualizer.plot_roc_curves(
            base_predictions, test_labels, plots_dir, 'base'
        )
        roc_meta_path, roc_meta_data = self.visualizer.plot_roc_curves(
            final_predictions, test_labels, plots_dir, 'meta'
        )
        plot_paths['roc_curves_base'] = roc_base_path
        plot_paths['roc_curves_meta'] = roc_meta_path
        
        print("Generating precision-recall curves...")
        pr_base_path, pr_base_data = self.visualizer.plot_precision_recall_curves(
            base_predictions, test_labels, plots_dir, 'base'
        )
        pr_meta_path, pr_meta_data = self.visualizer.plot_precision_recall_curves(
            final_predictions, test_labels, plots_dir, 'meta'
        )
        plot_paths['pr_curves_base'] = pr_base_path
        plot_paths['pr_curves_meta'] = pr_meta_path
        
        print("Generating model comparison chart...")
        comparison_path, comparison_df = self.visualizer.plot_model_comparison(
            base_metrics, meta_metrics, plots_dir
        )
        plot_paths['model_comparison'] = comparison_path
        
        print("Generating learning curves...")
        learning_path, learning_data = self.visualizer.plot_learning_curves(
            training_history, plots_dir
        )
        if learning_path:
            plot_paths['learning_curves'] = learning_path
        
        print("Saving detailed metrics...")
        csv_path, reports_path, metrics_df = self._save_detailed_metrics(
            base_metrics, meta_metrics, base_predictions, final_predictions, 
            test_labels, plots_dir
        )
        plot_paths['detailed_metrics_csv'] = csv_path
        plot_paths['classification_reports'] = reports_path
        
        return plot_paths
    
    def _save_detailed_metrics(self, base_metrics, meta_metrics, base_predictions, 
                             meta_predictions, test_labels, save_dir):
        """상세 메트릭을 CSV 및 JSON으로 저장"""
        
        all_metrics = []
        
        for model_name, metrics in base_metrics.items():
            row = {'Model': model_name.upper(), 'Type': 'Base Model'}
            row.update(metrics)
            all_metrics.append(row)
        
        for model_name, metrics in meta_metrics.items():
            row = {'Model': model_name.upper(), 'Type': 'Meta Learner'}
            row.update(metrics)
            all_metrics.append(row)
        
        df_metrics = pd.DataFrame(all_metrics)
        
        csv_path = os.path.join(save_dir, 'detailed_metrics.csv')
        df_metrics.to_csv(csv_path, index=False)
        
        reports = {}
        
        for model_name, predictions in base_predictions.items():
            y_pred = np.argmax(predictions, axis=1)
            report = classification_report(test_labels, y_pred, 
                                         target_names=['Healthy Control', 'Parkinson Disease'],
                                         output_dict=True)
            reports[f'{model_name}_base'] = report
        
        for model_name, results in meta_predictions.items():
            y_pred = results['predictions']
            report = classification_report(test_labels, y_pred,
                                         target_names=['Healthy Control', 'Parkinson Disease'], 
                                         output_dict=True)
            reports[f'{model_name}_meta'] = report
        
        reports_path = os.path.join(save_dir, 'classification_reports.json')
        with open(reports_path, 'w') as f:
            json.dump(reports, f, indent=2, ensure_ascii=False)
        
        return csv_path, reports_path, df_metrics