# Voice Feature Extraction

# 음성 특징 추출

이 저장소는 주로 `parselmouth` 라이브러리를 사용하여 오디오 파일에서 다양한 음성 특징을 추출하는 Python 스크립트를 포함합니다.

## 프로젝트 구조

```
audio_features/
│
├── __init__.py
├── base_utils.py                # 공용 유틸 (파일 로드, NaN 처리 등)
├── duration_energy.py           # Duration, Energy
├── pitch_features.py            # Pitch, Voiced ratio
├── jitter_shimmer.py            # Jitter, Shimmer
├── formant_features.py          # Formants & Bandwidths
├── harmonicity_features.py      # HNR
├── intensity_features.py        # Intensity
├── mfcc_features.py             # MFCC
├── spectrum_features.py         # Spectrum, Spectrogram
├── extractor.py                 # 통합 관리 (extract_all_features)
└── exam.py                      # 사용 예제
```

## License

`parselmouth`는 Praat 소프트웨어를 위한 Python 라이브러리입니다. 이 라이브러리는 GNU General Public License (GPL) 버전 3 이상에 따라 라이선스가 부여됩니다.

자세한 내용은 Praat 및 Parselmouth 공식 문서와 라이선스 정보를 참조하세요:
*   **Praat:** [https://www.fon.hum.uva.nl/praat/](https://www.fon.hum.uva.nl/praat/)
*   **Parselmouth:** [https://parselmouth.readthedocs.io/en/stable/](https://parselmouth.readthedocs.io/en/stable/)
