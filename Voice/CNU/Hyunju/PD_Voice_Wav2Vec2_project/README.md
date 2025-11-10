
# 프로젝트 구조

```bash
PD_Voice_Wav2Vec2_project/
├── data/
│   ├── vowel_data_preprocessed/             # 학습용 데이터 (지속모음 /pa/, /ta/ 포함)
│   ├── vowel_data_preprocessed_img/         # 학습용 데이터 (MFCC 시각화)
│   ├── italian_voice_pdhc_split_img/        # 외부 테스트용 데이터 (MFCC 시각화)
│   └── italian_voice_pdhc_split/            # 외부 테스트용 데이터
│
├── model/
│   ├── wav2vec2_finetuning.ipynb            # Wav2Vec2 파인튜닝 코드
│   ├── parkinson_voice_classification_pipline.ipynb  # 멀티모달 LLM 최종 파이프라인
│
├── preprocess/
│   ├── voice2mfcc_visual.ipynb              # MFCC 시각화 코드
│   └── preprocess_audio.py                  # 오디오 전처리 코드
│
├── requirements.txt                         # 실행 환경
└── README.md                                # 프로젝트 설명 문서
```

# 주요 구성 및 경로 설정

### 1. 모델 파인튜닝 코드

**경로:**  
'E:\PD_Voice_Wav2Vec2_project\model\finetuning_weightdecay.ipynb'

- **내용:** Wav2Vec2 모델을 파킨슨병 음성 데이터로 파인튜닝.  
- **특징:** Weight Decay 적용, 지속모음 및 발음(/pa/, /ta/) 기반 데이터 사용.

---

### 2. 학습용 데이터

**데이터셋 구성:**
- **IPVS Dataset**: 지속모음 /a/, /e/, /i/, /o/, /u/ + /pa/, /ta/  
- **Voice Samples Dataset**: 지속모음 /a/

**경로:**  
'E:\PD_Voice_Wav2Vec2_project\data\vowel_data_preprocessed'

> 모든 학습 데이터는 **raw 음성 파일** 형태로 사용.

---

### 3. 파인튜닝 완료 후 가중치 파일

**경로:**  
'E:\PD_Voice_Wav2Vec2_project\model\wav2vec2\wav2vec2-finetuned-pd-preprocess-weightdecay\checkpoint-1690'

- PyTorch 모델 체크포인트 (.bin / .pt 형식)
- 후속 테스트 및 PD 분류 파이프라인 입력으로 사용

---

### 4. 테스트용 데이터 (평가용)

**구성:**  
- IPVS 데이터셋 중 학습에 사용하지 않은 발화(지속모음 및 /pa/, /ta/ 제외)

**경로:**  
'E:\PD_Voice_Wav2Vec2_project\data\italian_voice_pdhc_split'

---

### 5. MFCC 시각화 코드

**경로:**  
'E:\PD_Voice_Wav2Vec2_project\preprocess\voice2mfcc_visual.ipynb'

- 음성 신호의 주파수 특성을 MFCC로 변환 후 시각화  


---

### 6. 최종 아키텍처 코드

**경로:**  
'E:\PD_Voice_Wav2Vec2_project\model\parkinson_voice_classification_pipline.ipynb'

- Wav2Vec2 기반 음성 데이터 임베딩 추출
- ResNet18 기반 MFCC 시각화 임베딩 추출
- 음성/이미지 임베딩 추출 -> cross attention 수행
- BERT 모델에 벡터값 입력
- (PD/HC) 이진 분류 수행


# 설치 및 실행


### 1. 의존성 설치



