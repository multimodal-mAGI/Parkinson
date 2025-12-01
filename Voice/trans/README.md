# Transformer Cross-Attention 기반 파킨슨병 예측 모델

Transformer Cross-Attention을 활용하여 CNN 특징과 MFCC 음성 특징을 융합하고 파킨슨병을 예측하는 딥러닝 모델입니다.

##  개요

이 모델은 다음과 같은 구조로 작동합니다:

1. **특징 추출**
   - CNN 모델(ResNet-50)에서 음성 스펙트로그램의 고수준 특징 추출 (2048차원)
   - MFCC(Mel-Frequency Cepstral Coefficients) 추출 (12차원)

2. **Cross-Attention 융합**
   - CNN 특징과 MFCC 특징 간 Cross-Attention으로 상호작용 학습
   - 두 modality의 정보를 효과적으로 결합

3. **분류**
   - 융합된 특징으로 파킨슨병(PD) vs 건강한 사람(HC) 이진 분류

##  모델 아키텍처

### TransformerCrossAttentionModel

```
Input: CNN Features (2048-dim) + MFCC Features (12-dim)
  ↓
Feature Projection → d_model (256-dim)
  ↓
Self-Attention for each modality
  ↓
Cross-Attention (CNN ↔ MFCC)
  ↓
Feature Fusion
  ↓
Classification Head
  ↓
Output: [HC, PD] probabilities
```

### 주요 특징
- **Multi-head Attention**: 8 heads
- **Transformer Layers**: 4 layers
- **Feed-forward Dimension**: 1024
- **Dropout**: 0.3
- **사전학습**: PyTorch의 Transformer 아키텍처 사용 (BERT와 유사)

##  파일 구조

```
trans/
├── feature_extractor.py    # CNN 및 MFCC 특징 추출
├── model.py                 # Transformer 모델 정의
├── train.py                 # 모델 학습 스크립트
├── predict.py               # 모델 예측 스크립트
├── requirements.txt         # 필요한 패키지
└── README.md               # 이 파일
```

##  사용 방법

### 1. 환경 설정

```bash
pip install -r requirements.txt
```

### 2. 데이터 준비

- 건강한 사람 음성 데이터: `../data/EN/healthy/`
- 파킨슨 환자 음성 데이터: `../data/EN/parkinson/`
- 지원 형식: `.wav`, `.mp3`

### 3. CNN 모델 학습 (선행 작업)

먼저 CNN 모델을 학습해야 합니다:

```bash
cd ../cnn
python main.py  # MODE='train'으로 설정
```

학습된 모델이 `../cnn/cnn_model.pth`에 저장됩니다.

### 4. Transformer 모델 학습

```bash
python train.py
```

**학습 파라미터** (`train.py` 상단에서 수정 가능):
- `EPOCHS`: 100 (기본값)
- `BATCH_SIZE`: 8
- `LEARNING_RATE`: 0.0001
- `PATIENCE`: 20 (Early Stopping)
- `MODEL_TYPE`: 'cross_attention' (또는 'simple')

**출력**:
- `transformer_model.pth`: 학습된 모델
- `transformer_training_curve_*.png`: 학습 곡선 그래프

### 5. 예측 수행

```bash
python predict.py
```

**예측 데이터 경로 설정** (`predict.py` 상단에서 수정):

```python
# 폴더 전체
PREDICT_DATA_PATH = "../data/testdata_KO/healthy"

# 파일 리스트
PREDICT_DATA_PATH = ["audio1.wav", "audio2.wav"]

# 단일 파일
PREDICT_DATA_PATH = "../data/audio.wav"
```

**출력**:
- `transformer_prediction_*.csv`: 예측 결과 (CSV)
- `transformer_prediction_*.txt`: 예측 결과 (TXT)

##  예측 결과 예시

```
[1/10] sample_voice_001.wav
  예측: PD
  신뢰도: 92.3%
  확률: HC=7.7%, PD=92.3%

[2/10] sample_voice_002.wav
  예측: HC
  신뢰도: 85.6%
  확률: HC=85.6%, PD=14.4%
```

##  고급 설정

### 모델 하이퍼파라미터 조정

`train.py`에서 다음 파라미터를 수정할 수 있습니다:

```python
D_MODEL = 256              # Transformer 차원
NHEAD = 8                  # Attention head 개수
NUM_LAYERS = 4             # Transformer layer 개수
DIM_FEEDFORWARD = 1024     # Feed-forward 차원
DROPOUT = 0.3              # Dropout 비율
```

### 모델 타입 선택

```python
MODEL_TYPE = 'cross_attention'  # Cross-Attention 모델
# MODEL_TYPE = 'simple'         # 단순 Transformer 모델 (비교용)
```

##  성능 평가 지표

학습 완료 후 다음 지표가 출력됩니다:

- **Accuracy**: 정확도
- **Precision**: 정밀도
- **Recall**: 재현율
- **F1 Score**: F1 점수
- **AUC**: ROC AUC 점수
- **Confusion Matrix**: 혼동 행렬

##  특징 추출 테스트

특징 추출만 테스트하려면:

```bash
python feature_extractor.py
```

##  모델 선택 가이드

### Cross-Attention 모델 (권장)
- CNN과 MFCC 특징 간 상호작용 학습
- 더 복잡하지만 성능이 더 우수할 가능성
- 파라미터 수: 약 2~3M

### Simple 모델 (비교용)
- CNN과 MFCC 특징을 단순 concat
- 더 간단하고 빠른 학습
- 파라미터 수: 약 1~2M

##  주요 차별점

### 기존 CNN 모델 대비
1. **Multi-modal 학습**: CNN 특징 + MFCC 메타데이터
2. **Attention 메커니즘**: 특징 간 관계 학습
3. **더 풍부한 표현**: 음향학적 특징과 딥러닝 특징 융합

### 일반 Transformer 대비
1. **Cross-Attention**: 서로 다른 modality 간 상호작용
2. **도메인 특화**: 파킨슨병 음성 특징에 최적화
3. **효율적 구조**: 작은 데이터셋에서도 학습 가능

##  참고사항

### 필수 선행 작업
- CNN 모델 학습 완료 (`../cnn/cnn_model.pth` 필요)
- MFCC 추출 모듈 준비 (`../feature/parselmouth/` 필요)

### GPU 사용
- CUDA 사용 가능 시 자동으로 GPU 사용
- CPU로만 학습 시 시간이 오래 걸릴 수 있음

### Early Stopping
- Validation loss가 PATIENCE(20) epoch 동안 개선되지 않으면 학습 중단
- 최적의 모델 자동 저장

##  문제 해결

### "CNN 모델을 찾을 수 없습니다"
```bash
cd ../cnn
python main.py  # MODE='train'으로 설정하여 CNN 모델 먼저 학습
```

### "CUDA out of memory"
- `BATCH_SIZE`를 줄이세요 (예: 8 → 4)
- `D_MODEL`을 줄이세요 (예: 256 → 128)

### "유효한 오디오 파일이 없습니다"
- 데이터 경로가 올바른지 확인
- `.wav` 또는 `.mp3` 파일이 있는지 확인

##  참고 문헌

- Transformer: "Attention Is All You Need" (Vaswani et al., 2017)
- BERT: "Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2018)
- ResNet: "Deep Residual Learning" (He et al., 2016)

