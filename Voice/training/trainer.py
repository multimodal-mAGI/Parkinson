import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from preprocessing import prepare_batch_data


class BaseModelTrainer:
    """베이스 모델 훈련 클래스"""
    
    def __init__(self, device='cuda'):
        self.device = device
        
    def train_base_models(self, base_models, processed_data, train_labels, val_paths=None, val_labels=None, epochs=30, batch_size=16):
        """베이스 모델들 훈련 - Early Stopping 포함"""
        print("=== Base Models Training ===")

        # Validation 데이터 전처리
        val_processed = None
        if val_paths is not None and val_labels is not None:
            print("Validation 데이터 전처리 중...")
            from preprocessing import AudioPreprocessor
            preprocessor = AudioPreprocessor()
            val_processed = preprocessor.load_and_preprocess_audio(val_paths)

        base_predictions = {}
        base_metrics = {}
        training_history = {}

        for name, model in base_models.items():
            print(f"\nTraining {name.upper()} model...")

            torch.cuda.empty_cache()

            lr = 0.0001 if name == 'transformer' else 0.001
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
            criterion = nn.CrossEntropyLoss()

            max_grad_norm = 1.0

            # Early stopping 변수
            best_val_loss = float('inf')
            patience = 40
            patience_counter = 0
            best_model_state = None

            training_history[name] = {'train_losses': [], 'val_losses': []}

            model.train()
            dataset_size = len(train_labels)
            all_predictions = []

            for epoch in range(epochs):
                # Training phase
                epoch_loss = 0
                num_batches = 0

                current_batch_size = 2 if name == 'transformer' else batch_size

                for i in range(0, dataset_size, current_batch_size):
                    end_idx = min(i + current_batch_size, dataset_size)

                    batch_data = prepare_batch_data(processed_data, name,
                                                  slice(i, end_idx), self.device)
                    batch_labels = torch.LongTensor(train_labels[i:end_idx]).to(self.device)

                    optimizer.zero_grad()
                    outputs = model(batch_data)

                    if torch.isnan(outputs).any():
                        print(f"Warning: NaN detected in {name} outputs")
                        continue

                    loss = criterion(outputs, batch_labels)

                    if torch.isnan(loss):
                        print(f"Warning: NaN loss detected in {name}")
                        continue

                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                    optimizer.step()

                    epoch_loss += loss.item()
                    num_batches += 1

                    del batch_data, batch_labels, outputs, loss
                    torch.cuda.empty_cache()

                if num_batches > 0:
                    avg_train_loss = epoch_loss / num_batches
                    training_history[name]['train_losses'].append(avg_train_loss)

                    # Validation phase
                    if val_processed is not None:
                        val_loss = self._validate_model(model, val_processed, val_labels, name, criterion, current_batch_size)
                        training_history[name]['val_losses'].append(val_loss)

                        if (epoch + 1) % 10 == 0:
                            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

                        # Early stopping 체크
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            patience_counter = 0
                            best_model_state = model.state_dict().copy()
                        else:
                            patience_counter += 1

                        if patience_counter >= patience:
                            print(f"Early stopping at epoch {epoch+1}")
                            model.load_state_dict(best_model_state)
                            break
                    else:
                        if (epoch + 1) % 10 == 0:
                            print(f"Epoch [{epoch+1}/{epochs}], Avg Loss: {avg_train_loss:.4f}")
            
            # 예측 생성
            model.eval()
            all_predictions = []
            
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
            
            print(f"{name.upper()} model training completed!")
        
        return base_predictions, training_history
    
    def _validate_model(self, model, val_processed, val_labels, name, criterion, batch_size):
        """Validation 데이터로 모델 평가"""
        model.eval()
        val_loss = 0
        num_batches = 0

        with torch.no_grad():
            dataset_size = len(val_labels)
            for i in range(0, dataset_size, batch_size):
                end_idx = min(i + batch_size, dataset_size)

                batch_data = prepare_batch_data(val_processed, name,
                                              slice(i, end_idx), self.device)
                batch_labels = torch.LongTensor(val_labels[i:end_idx]).to(self.device)

                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)

                val_loss += loss.item()
                num_batches += 1

                del batch_data, batch_labels, outputs, loss
                torch.cuda.empty_cache()

        model.train()
        return val_loss / num_batches if num_batches > 0 else 0

    def _check_and_handle_nan(self, data, model_name):
        """NaN 값 체크 및 처리"""
        if np.isnan(data).any():
            print(f"Warning: NaN detected in {model_name} predictions")
            data = np.nan_to_num(data, nan=0.0)
        return data


class MetaLearnerTrainer:
    """메타 러너 훈련 클래스"""
    
    def __init__(self):
        self.meta_learners = {
            'xgb': XGBClassifier(n_estimators=100, random_state=42),
            'rf': RandomForestClassifier(n_estimators=100, random_state=42),
            'gb': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        self.imputer = SimpleImputer(strategy='mean')
    
    def train_meta_learners(self, base_predictions, train_labels):
        """메타 러너들 훈련 - NaN 처리 개선"""
        print("\n=== Meta Learners Training ===")
        
        stacked_features = np.hstack([pred for pred in base_predictions.values()])
        
        print(f"Stacked features shape: {stacked_features.shape}")
        print(f"NaN count in stacked features: {np.isnan(stacked_features).sum()}")
        
        if np.isnan(stacked_features).any():
            print("Warning: NaN detected in stacked features. Applying imputation...")
            stacked_features = self.imputer.fit_transform(stacked_features)
            print(f"After imputation - NaN count: {np.isnan(stacked_features).sum()}")
        
        meta_metrics = {}
        
        for name, meta_model in self.meta_learners.items():
            print(f"\nTraining {name.upper()} meta learner...")
            
            try:
                meta_model.fit(stacked_features, train_labels)
                
                meta_pred = meta_model.predict(stacked_features)
                meta_pred_proba = meta_model.predict_proba(stacked_features)
                
                meta_metrics[name] = self._calculate_metrics(
                    train_labels, meta_pred, meta_pred_proba[:, 1]
                )
                
                print(f"{name.upper()} meta learner training completed!")
                
            except Exception as e:
                print(f"Error training {name} meta learner: {e}")
                meta_metrics[name] = {
                    'accuracy': 0.5, 'precision': 0.5, 'recall': 0.5, 
                    'f1_score': 0.5, 'auc': 0.5
                }
        
        return meta_metrics, self.imputer
    
    def _calculate_metrics(self, y_true, y_pred, y_proba):
        """성능 메트릭 계산"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'auc': roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) == 2 else 0
        }