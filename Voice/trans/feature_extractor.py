"""
CNN과 MFCC 특징 추출 모듈
- CNN 모델에서 중간 레이어 특징 추출
- MFCC 특징 추출
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# 상위 디렉토리 모듈 import
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from cnn.model import CNNModel
from cnn.preprocessing import CNNAudioPreprocessor
from feature.parselmouth.mfcc_features import extract_mfcc_features
import parselmouth


class CNNFeatureExtractor:
    """CNN 모델에서 중간 레이어 특징 추출"""

    def __init__(self, model_path, device='cuda'):
        """
        Args:
            model_path: 학습된 CNN 모델 경로
            device: 'cuda' 또는 'cpu'
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.preprocessor = CNNAudioPreprocessor()

        # 모델 로드
        self.model = CNNModel(num_classes=2).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # ResNet의 avgpool 이전까지의 특징 추출
        # ResNet-50의 경우 avgpool 이전 특징 맵 크기: [batch, 2048, 7, 7]
        self.feature_extractor = nn.Sequential(
            self.model.resnet.conv1,
            self.model.resnet.bn1,
            self.model.resnet.relu,
            self.model.resnet.maxpool,
            self.model.resnet.layer1,
            self.model.resnet.layer2,
            self.model.resnet.layer3,
            self.model.resnet.layer4,
            self.model.resnet.avgpool
        )

        print(f"CNN Feature Extractor initialized on {self.device}")

    def extract(self, audio_paths):
        """
        오디오 파일들로부터 CNN 특징 추출

        Args:
            audio_paths: 오디오 파일 경로 리스트

        Returns:
            features: [batch, 2048] 형태의 특징 벡터
        """
        # 전처리
        processed_data = self.preprocessor.load_and_preprocess_audio(audio_paths)

        features_list = []

        with torch.no_grad():
            for i in range(len(processed_data)):
                # [1, 1, H, W] 형태로 변환
                data = torch.FloatTensor(processed_data[i:i+1]).unsqueeze(1).to(self.device)

                # 특징 추출: [1, 2048, 1, 1]
                features = self.feature_extractor(data)

                # [1, 2048] 형태로 flatten
                features = features.view(features.size(0), -1)

                features_list.append(features.cpu().numpy())

        # [batch, 2048] 형태로 결합
        return np.vstack(features_list)


class MFCCFeatureExtractor:
    """MFCC 특징 추출 (메타데이터)"""

    def __init__(self, n_mfcc=12):
        """
        Args:
            n_mfcc: MFCC 계수 개수 (기본 12)
        """
        self.n_mfcc = n_mfcc
        print(f"MFCC Feature Extractor initialized (n_mfcc={n_mfcc})")

    def extract(self, audio_paths):
        """
        오디오 파일들로부터 MFCC 특징 추출

        Args:
            audio_paths: 오디오 파일 경로 리스트

        Returns:
            features: [batch, n_mfcc] 형태의 MFCC 특징 벡터
        """
        features_list = []

        for audio_path in audio_paths:
            # Parselmouth로 오디오 로드
            snd = parselmouth.Sound(audio_path)

            # MFCC 추출
            mfcc_dict = extract_mfcc_features(snd)

            # 딕셔너리를 벡터로 변환 (mfcc_1, mfcc_2, ..., mfcc_12)
            mfcc_vector = []
            for i in range(1, self.n_mfcc + 1):
                value = mfcc_dict.get(f'mfcc_{i}', 0.0)
                # NaN 처리
                if np.isnan(value):
                    value = 0.0
                mfcc_vector.append(value)

            features_list.append(mfcc_vector)

        # [batch, n_mfcc] 형태로 변환
        return np.array(features_list, dtype=np.float32)


class CombinedFeatureExtractor:
    """CNN과 MFCC 특징을 함께 추출하는 통합 클래스"""

    def __init__(self, cnn_model_path, n_mfcc=12, device='cuda'):
        """
        Args:
            cnn_model_path: 학습된 CNN 모델 경로
            n_mfcc: MFCC 계수 개수
            device: 'cuda' 또는 'cpu'
        """
        self.cnn_extractor = CNNFeatureExtractor(cnn_model_path, device)
        self.mfcc_extractor = MFCCFeatureExtractor(n_mfcc)

    def extract(self, audio_paths):
        """
        오디오 파일들로부터 CNN과 MFCC 특징을 모두 추출

        Args:
            audio_paths: 오디오 파일 경로 리스트

        Returns:
            cnn_features: [batch, 2048] CNN 특징
            mfcc_features: [batch, n_mfcc] MFCC 특징
        """
        print(f"Extracting features from {len(audio_paths)} audio files...")

        cnn_features = self.cnn_extractor.extract(audio_paths)
        mfcc_features = self.mfcc_extractor.extract(audio_paths)

        print(f"CNN features shape: {cnn_features.shape}")
        print(f"MFCC features shape: {mfcc_features.shape}")

        return cnn_features, mfcc_features


if __name__ == "__main__":
    # 테스트 코드
    import glob

    # 샘플 데이터 경로
    test_audio_path = "../data/EN/healthy"
    audio_files = glob.glob(os.path.join(test_audio_path, "*.wav"))[:5]

    if not audio_files:
        print("테스트 오디오 파일을 찾을 수 없습니다.")
    else:
        # CNN 모델 경로
        cnn_model_path = "../cnn/cnn_model.pth"

        if not os.path.exists(cnn_model_path):
            print(f"CNN 모델을 찾을 수 없습니다: {cnn_model_path}")
        else:
            # 특징 추출 테스트
            extractor = CombinedFeatureExtractor(
                cnn_model_path=cnn_model_path,
                n_mfcc=12,
                device='cuda'
            )

            cnn_feats, mfcc_feats = extractor.extract(audio_files)

            print("\n=== Feature Extraction Test Results ===")
            print(f"Number of samples: {len(audio_files)}")
            print(f"CNN features: {cnn_feats.shape}")
            print(f"MFCC features: {mfcc_feats.shape}")
            print(f"\nCNN feature sample (first 5 values): {cnn_feats[0, :5]}")
            print(f"MFCC feature sample: {mfcc_feats[0]}")
