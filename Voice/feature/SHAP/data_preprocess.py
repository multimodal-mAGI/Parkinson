"""
data_preprocess.py

음성 특징 CSV 데이터를 불러오고 전처리하는 모듈입니다.
- 성별 컬럼을 0(Female), 1(Male)로 변환
- 문자열 범주형 컬럼을 숫자 코드로 변환
- 입력(X)과 레이블(y)로 분리
"""

import pandas as pd

def load_and_preprocess(path: str = "voice_features.csv"):
    """
    CSV 파일을 불러와 전처리 후 입력 데이터(X)와 레이블(y) 반환

    Parameters
    ----------
    path : str
        CSV 파일 경로 (기본값: "voice_features.csv")

    Returns
    -------
    X : pd.DataFrame
        모델 입력용 특징 데이터
    y : pd.Series
        레이블
    """

    # 1️⃣ 데이터 로드
    df = pd.read_csv(path)
    print(f"[DATA] Loaded: {df.shape} (rows, columns)")

    # 2️⃣ 성별 컬럼 변환 (Female -> 0, Male -> 1)
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'Female': 0, 'Male': 1})
        print("[DATA] Gender column converted: Female->0, Male->1")

    # 3️⃣ 문자열 범주형 컬럼 -> 숫자 코드
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category').cat.codes
        print(f"[DATA] Column '{col}' converted from categorical -> numeric codes")

    # 4️⃣ 입력(X)과 레이블(y) 분리
    X = df.drop(columns=['label'])
    y = df['label']

    return X, y
