# CNN 전용 모델 (노이즈 제거 강화)

ResNet-50 기반 CNN 모델에 강화된 전처리를 적용한 독립 실행 버전입니다.

## 특징

### 1. 강화된 노이즈 제거 전처리
- **Wiener 필터**: 가우시안 노이즈 제거 (noise_power=0.01)
  - 노이즈 추정 기반 최적 필터링
  - SNR 향상

- **Hamming 윈도우**: 스펙트럼 누설 방지
  - 신호 양 끝의 불연속성 제거
  - 주파수 분석 정확도 향상

- **Wavelet Transform**: 다해상도 노이즈 제거
  - 4단계 다해상도 분해 (db4)
  - MAD (Median Absolute Deviation) 기반 노이즈 추정
  - Bayes Shrink 임계값 계산
  - 소프트 임계값으로 세부 계수 처리

### 2. CNN 모델
- ImageNet 사전학습된 ResNet-50 파인튜닝
- Grayscale Mel Spectrogram 입력 (64x64)
- 2-class 분류 (HC vs PD)

## 디렉터리 구조
```
Voice/cnn/
├── main.py              # 메인 실행 파일
├── preprocessing.py     # 강화된 전처리
├── model.py            # CNN 모델 정의
├── README.md           # 이 파일
└── cnn_model.pth       # 학습된 모델 (훈련 후 생성)
```

## 설치

### 필요한 패키지
```bash
pip install torch torchvision librosa numpy scipy pywt scikit-learn
```

## 사용 방법

### 1. 설정 수정
`main.py` 파일 상단의 설정을 수정하세요:

```python
# 학습용 데이터 경로
HEALTHY_PATH = "../data/testdata_KO/healthy"
PARKINSON_PATH = "../data/testdata_KO/parkinson"

# 예측용 데이터 경로
PREDICT_DATA_PATH = "../data/testdata_KO/healthy"

# 모델 저장 경로
MODEL_SAVE_PATH = "./cnn_model.pth"

# 훈련 파라미터
EPOCHS = 30
BATCH_SIZE = 16
LEARNING_RATE = 0.001

# 실행 모드: 'train' 또는 'predict'
MODE = 'train'
```

### 2. 학습 실행
```bash
cd Voice/cnn
python main.py
```

설정에서 `MODE = 'train'`으로 설정하고 실행하면:
1. 오디오 데이터 로드
2. 강화된 전처리 수행 (Wiener + Hamming + Wavelet)
3. CNN 모델 훈련
4. 성능 평가 (Accuracy, Precision, Recall, F1, AUC)
5. 모델 저장 (`cnn_model.pth`)

### 3. 예측 실행
```bash
cd Voice/cnn
python main.py
```

설정에서 `MODE = 'predict'`로 설정하고 실행하면:
1. 저장된 모델 로드
2. 예측할 오디오 파일 전처리
3. 예측 수행
4. 결과 출력 및 저장
   - `cnn_prediction_YYYYMMDD_HHMMSS.csv`
   - `cnn_prediction_YYYYMMDD_HHMMSS.txt`

## 출력 예시

### 학습 결과
```
=== 성능 평가 ===
Accuracy: 0.9250
Precision: 0.9233
Recall: 0.9250
F1 Score: 0.9241
AUC: 0.9680
Confusion Matrix:
[[45  3]
 [ 3 49]]
```

### 예측 결과
```
[1/100] audio001.wav
  예측: PD
  신뢰도: 92.5%
  확률: HC=7.5%, PD=92.5%

=== 예측 통계 ===
총 파일 수: 100개
HC 예측: 45개 (45.0%)
PD 예측: 55개 (55.0%)
```

## 전처리 파이프라인

```
원본 오디오
    ↓
[1] Wiener 필터 (noise_power=0.01)
    ↓
[2] Hamming 윈도우
    ↓
[3] Wavelet Transform 노이즈 제거
    - 4단계 분해
    - MAD 노이즈 추정
    - Bayes Shrink 임계값
    - 소프트 임계값 적용
    ↓
[4] 길이 정규화 (32000 샘플)
    ↓
[5] Mel Spectrogram 추출 (64x64)
    ↓
CNN 모델 입력
```

## 주의사항

1. **메모리**: 배치 크기는 GPU 메모리에 따라 조정하세요
2. **경로**: 상대 경로는 `Voice/cnn/` 기준입니다
3. **전처리 시간**: Wavelet Transform으로 인해 전처리 시간이 다소 소요됩니다

## 파일 설명

- **preprocessing.py**: 3단계 노이즈 제거 전처리 구현
- **model.py**: ResNet-50 기반 CNN 모델
- **main.py**: 학습/예측 메인 로직

## 기존 앙상블 모델과의 차이

| 항목 | 앙상블 (Voice/main.py) | CNN 단독 (Voice/cnn/main.py) |
|-----|----------------------|---------------------------|
| 모델 | CNN+RNN+Transformer+Hybrid+Meta | CNN만 |
| 전처리 | 기본 Mel Spectrogram | Wiener+Hamming+Wavelet |
| 복잡도 | 높음 (8개 모델) | 낮음 (1개 모델) |
| 속도 | 느림 | 빠름 |
| 정확도 | 높음 (앙상블 효과) | 중간 (단일 모델) |
