# Voice Feature Extraction

# 음성 특징 추출

이 저장소는 주로 `parselmouth` 라이브러리를 사용하여 오디오 파일에서 다양한 음성 특징을 추출하는 Python 스크립트를 포함합니다.

## 프로젝트 구조

```
project_root/
│
├── parselmouth/                # audio_features 패키지 전체를 넣은 폴더
│   ├── __init__.py
│   ├── base_utils.py           # 공용 유틸 (파일 로드, NaN 처리 등)
│   ├── duration_energy.py      # Duration, Energy
│   ├── pitch_features.py       # Pitch, Voiced ratio
│   ├── jitter_shimmer.py       # Jitter, Shimmer
│   ├── formant_features.py     # Formants & Bandwidths
│   ├── harmonicity_features.py # HNR
│   ├── intensity_features.py   # Intensity
│   ├── mfcc_features.py        # MFCC
│   ├── spectrum_features.py    # Spectrum, Spectrogram
│   ├── extractor.py            # 통합 관리 (extract_all_features)
│   └── exam.py                 # 사용 예제
│
└── SHAP/                       # SHAP 관련 코드용 폴더
    ├── __init__.py
    ├── main.py                 # SHAP 사용 예제, 학습 → 평가 → SHAP 분석 통합
    ├── data_preprocess.py      # CSV 로드, 범주형 변환, 레이블 분리(X, y)
    ├── model_train.py          # RandomForest 모델 K-Fold 학습 및 트리별 ROC-AUC 시각화
    ├── model_eval.py           # 모델 평가: Classification Report, Confusion Matrix, ROC Curve
    └── shap_analysis.py        # SHAP 분석 모듈
                                  - 전역 해석: Summary, Dependence Plot, Global Feature Importance
                                  - 국소 해석: Waterfall Plot, Force Plot
                                  - 클러스터링 Force Plot (HTML)
```

## License

프로젝트에서 사용한 라이브러리 및 라이선스:

| 라이브러리     | 설명                                    | 라이선스 |
|----------------|---------------------------------------|----------|
| Parselmouth    | Praat Python 인터페이스                 | GPL v3+ |
| NumPy          | 수치 계산, 배열 연산                     | BSD      |
| Pandas         | 데이터프레임 처리                        | BSD      |
| Matplotlib     | 시각화                                  | PSF      |
| Scikit-learn   | 머신러닝 모델 학습/평가                  | BSD      |
| SHAP           | 모델 해석 (SHAP 값 계산 및 시각화)       | MIT      |
| SciPy          | 수치 계산, 클러스터링, 수학 함수         | BSD      |

자세한 내용은 각 라이브러리의 공식 문서와 라이선스를 참조하세요:

- **Praat:** [https://www.fon.hum.uva.nl/praat/](https://www.fon.hum.uva.nl/praat/)  
- **Parselmouth:** [https://parselmouth.readthedocs.io/en/stable/](https://parselmouth.readthedocs.io/en/stable/)  
- **NumPy:** [https://numpy.org/license.html](https://numpy.org/license.html)  
- **Pandas:** [https://pandas.pydata.org/docs/getting_started/overview.html#license](https://pandas.pydata.org/docs/getting_started/overview.html#license)  
- **Matplotlib:** [https://matplotlib.org/stable/users/license.html](https://matplotlib.org/stable/users/license.html)  
- **Scikit-learn:** [https://scikit-learn.org/stable/about.html#license](https://scikit-learn.org/stable/about.html#license)  
- **SHAP:** [https://github.com/slundberg/shap/blob/master/LICENSE](https://github.com/slundberg/shap/blob/master/LICENSE)  
- **SciPy:** [https://www.scipy.org/scipylib/license.html](https://www.scipy.org/scipylib/license.html)  
