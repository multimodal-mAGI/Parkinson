import os
import pickle
import json
import torch
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

from models import CNNModel, RNNModel, TransformerModel, HybridModel
from preprocessing import AudioPreprocessor
from training import BaseModelTrainer, MetaLearnerTrainer
from prediction import EnsemblePredictor


class StackingEnsemble:
    """수정된 스태킹 앙상블 모델 - 모듈화된 버전"""
    
    def __init__(self, device='cuda'):
        self.device = device
        
        self.base_models = {
            'cnn': CNNModel().to(device),
            'rnn': RNNModel().to(device),
            'transformer': TransformerModel().to(device),
            'hybrid': HybridModel().to(device)
        }
        
        self.meta_learners = {
            'xgb': XGBClassifier(n_estimators=100, random_state=42),
            'rf': RandomForestClassifier(n_estimators=100, random_state=42),
            'gb': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        self.imputer = SimpleImputer(strategy='mean')
        self.is_trained = False
        self.training_history = {}
        self.base_metrics = {}
        self.meta_metrics = {}
        
        # 모듈 인스턴스들
        self.preprocessor = AudioPreprocessor()
        self.base_trainer = BaseModelTrainer(device)
        self.meta_trainer = MetaLearnerTrainer()
        self.predictor = None
    
    def train_base_models(self, audio_paths, train_labels, val_paths=None, val_labels=None, epochs=30, batch_size=16):
        """베이스 모델들 훈련"""
        processed_data = self.preprocessor.load_and_preprocess_audio(audio_paths)

        base_predictions, self.training_history = self.base_trainer.train_base_models(
            self.base_models, processed_data, train_labels,
            val_paths=val_paths, val_labels=val_labels,
            epochs=epochs, batch_size=batch_size
        )

        return base_predictions
    
    def train_meta_learners(self, base_predictions, train_labels):
        """메타 러너들 훈련"""
        self.meta_trainer.meta_learners = self.meta_learners
        self.meta_metrics, self.imputer = self.meta_trainer.train_meta_learners(
            base_predictions, train_labels
        )
        
        # 예측기 초기화
        self.predictor = EnsemblePredictor(
            self.base_models, self.meta_learners, self.imputer, self.device
        )
        
        self.is_trained = True
    
    def predict(self, audio_paths, batch_size=16):
        """앙상블 예측"""
        if not self.is_trained:
            raise ValueError("모델이 훈련되지 않았습니다. 먼저 train을 호출하세요.")
        
        processed_data = self.preprocessor.load_and_preprocess_audio(audio_paths)
        return self.predictor.predict(processed_data, batch_size)
    
    def save_models(self, save_dir="./ensemble_models"):
        """훈련된 모델들을 저장"""
        if not self.is_trained:
            print("Warning: 훈련되지 않은 모델을 저장하려고 합니다.")
        
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"모델들을 {save_dir}에 저장 중...")
        
        # 베이스 모델 저장
        for name, model in self.base_models.items():
            model_path = os.path.join(save_dir, f"{name}_model.pth")
            torch.save(model.state_dict(), model_path)
            print(f"  {name} 모델 저장 완료")
        
        # 메타 러너 저장
        for name, meta_model in self.meta_learners.items():
            meta_path = os.path.join(save_dir, f"{name}_meta.pkl")
            with open(meta_path, 'wb') as f:
                pickle.dump(meta_model, f)
            print(f"  {name} 메타러너 저장 완료")
        
        # Imputer 저장
        imputer_path = os.path.join(save_dir, "imputer.pkl")
        with open(imputer_path, 'wb') as f:
            pickle.dump(self.imputer, f)
        print("  Imputer 저장 완료")
        
        # 메타데이터 저장
        metadata = {
            'is_trained': self.is_trained,
            'device': str(self.device),
            'save_time': datetime.now().isoformat(),
            'base_models': list(self.base_models.keys()),
            'meta_learners': list(self.meta_learners.keys()),
            'training_history': self.training_history,
            'base_metrics': self.base_metrics,
            'meta_metrics': self.meta_metrics
        }
        
        metadata_path = os.path.join(save_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print("  메타데이터 저장 완료")
        
        print(f"모든 모델이 {save_dir}에 저장되었습니다!")
        return save_dir
    
    def load_models(self, save_dir="./ensemble_models"):
        """저장된 모델들을 로드"""
        if not os.path.exists(save_dir):
            raise FileNotFoundError(f"저장 디렉토리 {save_dir}가 존재하지 않습니다.")
        
        print(f"{save_dir}에서 모델들을 로드 중...")
        
        # 메타데이터 로드
        metadata_path = os.path.join(save_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"  저장 시간: {metadata.get('save_time', 'Unknown')}")
            print(f"  저장된 장치: {metadata.get('device', 'Unknown')}")
        
        # 베이스 모델 로드
        for name, model in self.base_models.items():
            model_path = os.path.join(save_dir, f"{name}_model.pth")
            if os.path.exists(model_path):
                try:
                    model.load_state_dict(torch.load(model_path, map_location=self.device))
                    model.to(self.device)
                    print(f"  {name} 모델 로드 완료")
                except Exception as e:
                    print(f"  {name} 모델 로드 실패: {e}")
            else:
                print(f"  Warning: {name} 모델 파일이 없습니다.")
        
        # 메타 러너 로드
        for name in self.meta_learners.keys():
            meta_path = os.path.join(save_dir, f"{name}_meta.pkl")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, 'rb') as f:
                        self.meta_learners[name] = pickle.load(f)
                    print(f"  {name} 메타러너 로드 완료")
                except Exception as e:
                    print(f"  {name} 메타러너 로드 실패: {e}")
            else:
                print(f"  Warning: {name} 메타러너 파일이 없습니다.")
        
        # Imputer 로드
        imputer_path = os.path.join(save_dir, "imputer.pkl")
        if os.path.exists(imputer_path):
            try:
                with open(imputer_path, 'rb') as f:
                    self.imputer = pickle.load(f)
                print("  Imputer 로드 완료")
            except Exception as e:
                print(f"  Imputer 로드 실패: {e}")
        
        # 메타데이터에서 기타 정보 복원
        if os.path.exists(metadata_path):
            if 'base_metrics' in metadata:
                self.base_metrics = metadata['base_metrics']
            if 'meta_metrics' in metadata:
                self.meta_metrics = metadata['meta_metrics']
            if 'is_trained' in metadata:
                self.is_trained = metadata['is_trained']
            if 'training_history' in metadata:
                self.training_history = metadata['training_history']
        
        # 예측기 초기화
        if self.is_trained:
            self.predictor = EnsemblePredictor(
                self.base_models, self.meta_learners, self.imputer, self.device
            )
        
        print("모든 모델 로드 완료!")
        return True