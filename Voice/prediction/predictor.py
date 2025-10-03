import torch
import torch.nn.functional as F
import numpy as np
from preprocessing import prepare_batch_data


class EnsemblePredictor:
    """앙상블 예측 클래스"""
    
    def __init__(self, base_models, meta_learners, imputer, device='cuda'):
        self.base_models = base_models
        self.meta_learners = meta_learners
        self.imputer = imputer
        self.device = device
    
    def predict(self, processed_data, batch_size=16):
        """앙상블 예측 - NaN 처리 개선"""
        base_predictions = self._get_base_predictions(processed_data, batch_size)
        final_predictions = self._get_meta_predictions(base_predictions)
        
        return final_predictions, base_predictions
    
    def _get_base_predictions(self, processed_data, batch_size):
        """베이스 모델들의 예측 생성"""
        base_predictions = {}
        dataset_size = len(processed_data['cnn'])
        
        for name, model in self.base_models.items():
            print(f"Predicting with {name.upper()} model...")
            model.eval()
            all_predictions = []
            
            torch.cuda.empty_cache()
            
            with torch.no_grad():
                current_batch_size = 2 if name == 'transformer' else batch_size
                
                for i in range(0, dataset_size, current_batch_size):
                    end_idx = min(i + current_batch_size, dataset_size)
                    
                    batch_data = prepare_batch_data(processed_data, name, 
                                                  slice(i, end_idx), self.device)
                    
                    outputs = model(batch_data)
                    batch_predictions = F.softmax(outputs, dim=1).cpu().numpy()
                    
                    batch_predictions = self._check_and_handle_nan(batch_predictions, name)
                    all_predictions.append(batch_predictions)
                    
                    del batch_data, outputs
                    torch.cuda.empty_cache()
            
            base_predictions[name] = np.vstack(all_predictions)
            base_predictions[name] = self._check_and_handle_nan(base_predictions[name], name)
        
        return base_predictions
    
    def _get_meta_predictions(self, base_predictions):
        """메타 러너들의 예측 생성"""
        stacked_features = np.hstack([pred for pred in base_predictions.values()])
        
        if np.isnan(stacked_features).any():
            print("Warning: NaN detected in prediction features. Applying imputation...")
            stacked_features = self.imputer.transform(stacked_features)
        
        final_predictions = {}
        for name, meta_model in self.meta_learners.items():
            try:
                pred = meta_model.predict(stacked_features)
                pred_proba = meta_model.predict_proba(stacked_features)
                final_predictions[name] = {
                    'predictions': pred,
                    'probabilities': pred_proba
                }
            except Exception as e:
                print(f"Error in {name} prediction: {e}")
                n_samples = len(stacked_features)
                final_predictions[name] = {
                    'predictions': np.random.randint(0, 2, n_samples),
                    'probabilities': np.random.rand(n_samples, 2)
                }
        
        return final_predictions
    
    def _check_and_handle_nan(self, data, model_name):
        """NaN 값 체크 및 처리"""
        if np.isnan(data).any():
            print(f"Warning: NaN detected in {model_name} predictions")
            data = np.nan_to_num(data, nan=0.0)
        return data
    
    def predict_single_sample(self, processed_sample_data):
        """단일 샘플 예측"""
        # 배치 차원 추가
        for key in processed_sample_data:
            if processed_sample_data[key].ndim == 1:
                processed_sample_data[key] = processed_sample_data[key][np.newaxis, :]
            elif processed_sample_data[key].ndim == 2:
                processed_sample_data[key] = processed_sample_data[key][np.newaxis, :, :]
        
        return self.predict(processed_sample_data, batch_size=1)