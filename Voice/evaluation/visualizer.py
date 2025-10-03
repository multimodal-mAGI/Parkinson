import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve


class ModelVisualizer:
    """모델 성능 시각화 클래스"""
    
    def __init__(self):
        pass
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name, save_dir):
        """Confusion Matrix 생성 및 저장"""
        plt.figure(figsize=(8, 6))
        
        cm = confusion_matrix(y_true, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Healthy Control', 'Parkinson Disease'],
                   yticklabels=['Healthy Control', 'Parkinson Disease'])
        
        plt.title(f'Confusion Matrix - {model_name.upper()}', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f'confusion_matrix_{model_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path, cm
    
    def plot_roc_curves(self, predictions_data, labels, save_dir, model_type='base'):
        """ROC Curves 생성 및 저장"""
        plt.figure(figsize=(10, 8))
        
        roc_data = {}
        
        for model_name, predictions in predictions_data.items():
            if model_type == 'base':
                y_proba = predictions[:, 1]
            else:
                y_proba = predictions['probabilities'][:, 1]
            
            fpr, tpr, _ = roc_curve(labels, y_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, linewidth=2, label=f'{model_name.upper()} (AUC = {roc_auc:.3f})')
            
            roc_data[model_name] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.8, label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curves - {model_type.title()} Models', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f'roc_curves_{model_type}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path, roc_data
    
    def plot_precision_recall_curves(self, predictions_data, labels, save_dir, model_type='base'):
        """Precision-Recall Curves 생성 및 저장"""
        plt.figure(figsize=(10, 8))
        
        pr_data = {}
        
        for model_name, predictions in predictions_data.items():
            if model_type == 'base':
                y_proba = predictions[:, 1]
            else:
                y_proba = predictions['probabilities'][:, 1]
            
            precision, recall, _ = precision_recall_curve(labels, y_proba)
            pr_auc = auc(recall, precision)
            
            plt.plot(recall, precision, linewidth=2, label=f'{model_name.upper()} (AUC = {pr_auc:.3f})')
            
            pr_data[model_name] = {'precision': precision, 'recall': recall, 'auc': pr_auc}
        
        positive_ratio = sum(labels) / len(labels)
        plt.axhline(y=positive_ratio, color='k', linestyle='--', linewidth=1, alpha=0.8, 
                   label=f'Random Classifier ({positive_ratio:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curves - {model_type.title()} Models', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f'precision_recall_curves_{model_type}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path, pr_data
    
    def plot_model_comparison(self, base_metrics, meta_metrics, save_dir):
        """모델 성능 비교 차트 생성 및 저장"""
        models = []
        accuracy_scores = []
        f1_scores = []
        auc_scores = []
        model_types = []
        
        for model_name, metrics in base_metrics.items():
            models.append(model_name.upper())
            accuracy_scores.append(metrics['accuracy'])
            f1_scores.append(metrics['f1_score'])
            auc_scores.append(metrics['auc'])
            model_types.append('Base Model')
        
        for model_name, metrics in meta_metrics.items():
            models.append(model_name.upper())
            accuracy_scores.append(metrics['accuracy'])
            f1_scores.append(metrics['f1_score'])
            auc_scores.append(metrics['auc'])
            model_types.append('Meta Learner')
        
        df = pd.DataFrame({
            'Model': models,
            'Accuracy': accuracy_scores,
            'F1_Score': f1_scores,
            'AUC': auc_scores,
            'Type': model_types
        })
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        metrics_to_plot = ['Accuracy', 'F1_Score', 'AUC']
        colors = ['lightblue', 'lightgreen']
        
        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i]
            
            for j, (model_type, color) in enumerate(zip(['Base Model', 'Meta Learner'], colors)):
                mask = df['Type'] == model_type
                if mask.any():
                    ax.bar(df[mask]['Model'], df[mask][metric], 
                          color=color, alpha=0.8, label=model_type)
            
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison', fontsize=12, fontweight='bold')
            ax.set_ylabel(metric.replace("_", " ").title(), fontsize=10)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            for idx, v in enumerate(df[metric]):
                ax.text(idx, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, 'model_performance_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path, df
    
    def plot_learning_curves(self, training_history, save_dir):
        """학습 곡선 생성 및 저장"""
        if not training_history:
            print("Warning: No training history available for learning curves")
            return None, None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (model_name, history) in enumerate(training_history.items()):
            if i >= 4:
                break
                
            ax = axes[i]
            
            if 'losses' in history and len(history['losses']) > 0:
                epochs = range(1, len(history['losses']) + 1)
                ax.plot(epochs, history['losses'], 'b-', linewidth=2, label='Training Loss')
                
                ax.set_title(f'Learning Curve - {model_name.upper()}', fontsize=12, fontweight='bold')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'No data for {model_name.upper()}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Learning Curve - {model_name.upper()}')
        
        for i in range(len(training_history), 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, 'learning_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path, training_history