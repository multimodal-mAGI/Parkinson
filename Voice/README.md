# 파킨슨 병 음성 분류 앙상블 모델 (Refactored)

이 프로젝트는 음성 데이터를 사용하여 파킨슨 병을 분류하는 앙상블 머신러닝 모델

## 프로젝트 구조

```
Voice/
├── data/                       # 데이터 디렉토리
│   └── EN/                    # 영어 데이터셋
│       ├── healthy/           # 건강한 사람 음성 파일
│       ├── parkinson/         # 파킨슨 환자 음성 파일
│       ├── HC_a/              # 건강한 사람 원본 데이터
│       └── PD_a/              # 파킨슨 환자 원본 데이터
├── models/                     # 모델 정의
│   ├── __init__.py
│   └── base_models.py         # CNN, RNN, Transformer, Hybrid 모델
├── preprocessing/              # 전처리 모듈
│   ├── __init__.py
│   └── audio_processor.py     # 오디오 전처리 클래스
├── training/                   # 훈련 모듈
│   ├── __init__.py
│   └── trainer.py             # 베이스 모델 및 메타 러너 훈련
├── prediction/                 # 예측 모듈
│   ├── __init__.py
│   └── predictor.py           # 앙상블 예측 클래스
├── evaluation/                 # 평가 및 시각화
│   ├── __init__.py
│   ├── evaluator.py           # 모델 평가 클래스
│   └── visualizer.py          # 시각화 클래스
├── ensemble/                   # 앙상블 메인 클래스
│   ├── __init__.py
│   └── ensemble_model.py      # 스태킹 앙상블 모델
├── utils/                      # 유틸리티 함수들
│   ├── __init__.py
│   ├── cross_validation.py    # 교차 검증
│   └── data_loader.py         # 데이터 로딩
├── cnn/                        # CNN 단일 모델 (독립 실행)
│   └── (CNN 단독 실행 파일)
├── train_result/               # 훈련 결과 저장 디렉토리
│   ├── (저장된 모델 파일)
│   ├── (시각화 결과)
│   └── (평가 메트릭)
├── main.py                     # 메인 실행 파일 (앙상블)
├── requirements.txt            # 의존성 패키지
└── README.md                   # 프로젝트 설명
```

## 주요 구성 요소

### 1. 베이스 모델들
- **CNN Model**: ResNet-50 기반 컨볼루션 신경망
- **RNN Model**: LSTM + Attention 메커니즘
- **Transformer Model**: Wav2Vec2 기반 트랜스포머
- **Hybrid Model**: CNN-LSTM 하이브리드 아키텍처

### 2. 메타 러너들
- **XGBoost**: 그래디언트 부스팅 알고리즘
- **Random Forest**: 랜덤 포레스트 앙상블
- **Gradient Boosting**: 그래디언트 부스팅 분류기

### 3. 주요 기능
- 오디오 전처리 (멜 스펙트로그램, MFCC 추출)
- 베이스 모델 훈련 및 예측
- 스태킹 앙상블 학습
- **Early Stopping**: Validation 기반 조기 종료 (patience=40)
- **데이터 분할**: Train 70% / Validation 10% / Test 20%
- 성능 평가 및 시각화
- 모델 저장/로드
- 교차 검증

## 설치 및 실행

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. 데이터 준비
- 건강한 사람의 음성 파일을 `healthy` 폴더에 저장
- 파킨슨 환자의 음성 파일을 `parkinson` 폴더에 저장
- 지원 형식: WAV, MP3

### 3. 데이터 경로 설정
`main.py` 파일에서 데이터 경로를 수정
```python
healthy_path = r"path/to/your/healthy/data"
parkinson_path = r"path/to/your/parkinson/data"
```

### 4. 실행
```bash
python main.py
```

## 사용법

### 기본 실행 흐름
1. **데이터 로드**: 지정된 경로에서 오디오 파일 수집
2. **데이터 분할**: Train/Validation/Test 세트 분할 (0.7/0.1/0.2)
3. **모델 훈련**: 베이스 모델들과 메타 러너들 훈련 (Early Stopping 적용)
4. **성능 평가**: 테스트 데이터에서 성능 평가 및 시각화
5. **예측 데모**: 새로운 데이터에 대한 예측 예시
6. **교차 검증**: 선택적으로 교차 검증 수행

### 모듈별 사용법

#### 1. 앙상블 모델 사용
```python
from ensemble import StackingEnsemble

# 모델 초기화
ensemble = StackingEnsemble(device='cuda')

# 훈련 (Early Stopping 포함)
base_predictions = ensemble.train_base_models(
    train_paths, train_labels,
    val_paths=val_paths, val_labels=val_labels,
    epochs=200, batch_size=16
)
ensemble.train_meta_learners(base_predictions, train_labels)

# 예측
predictions, base_preds = ensemble.predict(test_audio_paths)
```

#### 2. 개별 모듈 사용
```python
from models import CNNModel, RNNModel
from preprocessing import AudioPreprocessor
from training import BaseModelTrainer
from evaluation import ModelEvaluator

# 전처리
preprocessor = AudioPreprocessor()
processed_data = preprocessor.load_and_preprocess_audio(audio_paths)
val_processed = preprocessor.load_and_preprocess_audio(val_paths)

# 훈련 (Early Stopping 포함)
trainer = BaseModelTrainer(device='cuda')
base_models = {'cnn': CNNModel(), 'rnn': RNNModel()}
predictions, history = trainer.train_base_models(
    base_models, processed_data, train_labels,
    val_paths=val_paths, val_labels=val_labels,
    epochs=200, batch_size=16
)

# 평가
evaluator = ModelEvaluator()
results = evaluator.evaluate_models(ensemble, test_paths, test_labels)
```

## 출력 결과

### 1. 성능 메트릭
- Accuracy, Precision, Recall, F1-Score, AUC
- 베이스 모델별 및 메타 러너별 성능 비교

### 2. 시각화 파일
- Confusion Matrix (각 모델별)
- ROC Curves (베이스 모델, 메타 러너)
- Precision-Recall Curves
- 모델 성능 비교 차트
- 학습 곡선
- 상세 메트릭 CSV 파일

### 3. 모델 파일
- 훈련된 베이스 모델들 (.pth)
- 메타 러너들 (.pkl)
- 전처리 파이프라인 (imputer.pkl)
- 메타데이터 (metadata.json)

## 라이센스 및 출처

이 프로젝트에서 사용된 모든 모델과 라이브러리는 적절한 오픈소스 라이센스를 따릅니다

- **PyTorch/TorchVision**: BSD 3-Clause License
- **Transformers (Wav2Vec2)**: Apache 2.0 License
- **Scikit-learn**: BSD 3-Clause License
- **XGBoost**: Apache 2.0 License

