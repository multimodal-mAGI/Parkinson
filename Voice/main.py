import os
import sys
import torch
import numpy as np
import warnings
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ensemble import StackingEnsemble
from evaluation import ModelEvaluator
from utils import load_audio_data, cross_validation_ensemble



# 학습용 데이터 경로
HEALTHY_PATH = r"Voice\data\EN\healthy"
PARKINSON_PATH = r"Voice\data\EN\parkinson"

# 예측용 데이터 경로 (디렉토리, 파일 리스트, 단일 파일 모두 가능)
PREDICT_DATA_PATH = r"data\testdata_KO\healthy"
# PREDICT_DATA_PATH = ["audio1.wav", "audio2.wav"]  # 파일 리스트
# PREDICT_DATA_PATH = "./audio.wav"  # 단일 파일

# 모델 저장 경로
MODEL_SAVE_DIR = "./models/finetune"

# 훈련 파라미터
EPOCHS = 20
BATCH_SIZE = 8

# 실행 모드: 'train' 또는 'predict'
MODE = 'train'



def setup_device():
    """GPU 설정"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"사용 장치: {device}")

    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    return device


def train():
    """모델 학습"""
    print("\n" + "="*60)
    print("모델 학습 시작")
    print("="*60)

    # 경로 검증
    if not os.path.exists(HEALTHY_PATH):
        print(f"오류: 건강한 사람 데이터 경로 없음: {HEALTHY_PATH}")
        return

    if not os.path.exists(PARKINSON_PATH):
        print(f"오류: 파킨슨 환자 데이터 경로 없음: {PARKINSON_PATH}")
        return

    # 데이터 로드
    print("\n=== 데이터 로딩 ===")
    X, y = load_audio_data(HEALTHY_PATH, PARKINSON_PATH)

    if len(X) == 0:
        print("데이터를 찾을 수 없습니다.")
        return

    # 장치 설정
    device = setup_device()

    # 모델 생성
    ensemble = StackingEnsemble(device=device)

    # 데이터 분할
    print("\n=== 데이터 분할 ===")
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"훈련: {len(train_paths)}개 (HC: {np.sum(train_labels == 0)}, PD: {np.sum(train_labels == 1)})")
    print(f"테스트: {len(test_paths)}개 (HC: {np.sum(test_labels == 0)}, PD: {np.sum(test_labels == 1)})")

    # 모델 훈련
    print(f"\n=== 모델 훈련 (Epochs: {EPOCHS}, Batch: {BATCH_SIZE}) ===")
    try:
        base_predictions = ensemble.train_base_models(train_paths, train_labels,
                                                     epochs=EPOCHS, batch_size=BATCH_SIZE)
        ensemble.train_meta_learners(base_predictions, train_labels)
        print("\n훈련 완료!")
    except Exception as e:
        print(f"훈련 오류: {e}")
        return

    # 평가
    print("\n=== 성능 평가 ===")
    try:
        evaluator = ModelEvaluator()
        results = evaluator.evaluate_models(ensemble, test_paths, test_labels,
                                           save_plots=True, plots_dir="./evaluation_results")
        print("평가 완료!")
    except Exception as e:
        print(f"평가 오류: {e}")

    # 모델 저장
    print("\n=== 모델 저장 ===")
    try:
        ensemble.save_models(MODEL_SAVE_DIR)
        print(f"저장 완료: {MODEL_SAVE_DIR}")
    except Exception as e:
        print(f"저장 오류: {e}")

    print("\n" + "="*60)
    print("학습 완료!")
    print("="*60)


def predict():
    """예측 실행"""
    print("\n" + "="*60)
    print("예측 시작")
    print("="*60)

    # 모델 존재 확인
    if not os.path.exists(MODEL_SAVE_DIR) or not os.path.exists(os.path.join(MODEL_SAVE_DIR, "metadata.json")):
        print(f"오류: 모델이 없습니다. 먼저 train 모드를 실행하세요.")
        return

    # 장치 설정
    device = setup_device()

    # 모델 로드
    print("\n=== 모델 로딩 ===")
    ensemble = StackingEnsemble(device=device)

    try:
        ensemble.load_models(MODEL_SAVE_DIR)
        print("모델 로드 완료!")
    except Exception as e:
        print(f"모델 로드 오류: {e}")
        return

    # 데이터 경로 파싱
    audio_paths = []

    if isinstance(PREDICT_DATA_PATH, list):
        audio_paths = PREDICT_DATA_PATH
    elif os.path.isdir(PREDICT_DATA_PATH):
        import glob
        for ext in ["*.wav", "*.mp3"]:
            audio_paths.extend(glob.glob(os.path.join(PREDICT_DATA_PATH, ext)))
    elif os.path.isfile(PREDICT_DATA_PATH):
        audio_paths = [PREDICT_DATA_PATH]
    else:
        print(f"오류: 유효하지 않은 데이터 경로: {PREDICT_DATA_PATH}")
        return

    # 파일 존재 확인
    valid_paths = [p for p in audio_paths if os.path.exists(p)]

    if not valid_paths:
        print("유효한 오디오 파일이 없습니다.")
        return

    print(f"\n총 {len(valid_paths)}개 파일")

    # 예측 수행
    print("\n=== 예측 수행 ===")
    try:
        predictions, _ = ensemble.predict(valid_paths)

        print("\n" + "="*60)
        print("예측 결과")
        print("="*60)

        for i, path in enumerate(valid_paths):
            filename = os.path.basename(path)
            print(f"\n[{i+1}/{len(valid_paths)}] {filename}")
            print("-" * 60)

            for meta_name, result in predictions.items():
                pred_label = "Parkinson Disease" if result['predictions'][i] == 1 else "Healthy Control"
                pd_prob = result['probabilities'][i][1]
                hc_prob = result['probabilities'][i][0]
                confidence = max(pd_prob, hc_prob)

                print(f"  {meta_name.upper()}: {pred_label} (신뢰도: {confidence:.1%})")
                print(f"    HC={hc_prob:.1%}, PD={pd_prob:.1%}")

        print("\n" + "="*60)

    except Exception as e:
        print(f"예측 오류: {e}")
        return

    print("\n예측 완료!")


if __name__ == "__main__":
    if MODE == 'train':
        train()
    elif MODE == 'predict':
        predict()
    else:
        print(f"오류: MODE는 'train' 또는 'predict'여야 합니다. 현재: {MODE}")
