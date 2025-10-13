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
HEALTHY_PATH = r".\data\EN\healthy"
PARKINSON_PATH = r".\data\EN\parkinson"

# 예측용 데이터 경로 (디렉토리, 파일 리스트, 단일 파일 모두 가능)
PREDICT_DATA_PATH = r"data\testdata_KO\healthy"
# PREDICT_DATA_PATH = ["audio1.wav", "audio2.wav"]  # 파일 리스트
# PREDICT_DATA_PATH = "./audio.wav"  # 단일 파일

# 모델 저장 경로
MODEL_SAVE_DIR = r"models\finetune\ensemble_models_KO"

# 훈련 파라미터
EPOCHS = 20
BATCH_SIZE = 8

# 실행 모드: 'train' 또는 'predict'
MODE = 'predict'



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
                                           save_plots=True, plots_dir=".train_result/evaluation_results")
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
        predictions, base_preds = ensemble.predict(valid_paths)

        print("\n" + "="*60)
        print("예측 결과")
        print("="*60)

        # 통계 집계를 위한 변수
        base_stats = {name: {'HC': 0, 'PD': 0} for name in base_preds.keys()}
        meta_stats = {name: {'HC': 0, 'PD': 0} for name in predictions.keys()}

        # CSV 데이터 준비
        csv_data = []
        csv_header = ['파일명']

        # 헤더 구성
        for base_name in base_preds.keys():
            csv_header.extend([f'{base_name.upper()}_예측', f'{base_name.upper()}_HC확률', f'{base_name.upper()}_PD확률'])
        for meta_name in predictions.keys():
            csv_header.extend([f'{meta_name.upper()}_예측', f'{meta_name.upper()}_HC확률', f'{meta_name.upper()}_PD확률'])

        csv_data.append(csv_header)

        # 각 파일별 예측 결과
        for i, path in enumerate(valid_paths):
            filename = os.path.basename(path)
            print(f"\n[{i+1}/{len(valid_paths)}] {filename}")
            print("-" * 60)

            row_data = [filename]

            # 베이스 모델 예측
            print("\n  [베이스 모델 예측]")
            for base_name, base_pred in base_preds.items():
                hc_prob = base_pred[i][0]
                pd_prob = base_pred[i][1]
                pred_label = "PD" if pd_prob > hc_prob else "HC"

                # 통계 집계
                if pred_label == "PD":
                    base_stats[base_name]['PD'] += 1
                else:
                    base_stats[base_name]['HC'] += 1

                print(f"    {base_name.upper()}: {pred_label} (HC={hc_prob:.1%}, PD={pd_prob:.1%})")
                row_data.extend([pred_label, f"{hc_prob:.4f}", f"{pd_prob:.4f}"])

            # 메타 러너 예측 (최종)
            print("\n  [최종 예측 - 메타 러너]")
            for meta_name, result in predictions.items():
                pred_label = "PD" if result['predictions'][i] == 1 else "HC"
                pd_prob = result['probabilities'][i][1]
                hc_prob = result['probabilities'][i][0]
                confidence = max(pd_prob, hc_prob)

                # 통계 집계
                if pred_label == "PD":
                    meta_stats[meta_name]['PD'] += 1
                else:
                    meta_stats[meta_name]['HC'] += 1

                print(f"    {meta_name.upper()}: {pred_label} (신뢰도: {confidence:.1%}, HC={hc_prob:.1%}, PD={pd_prob:.1%})")
                row_data.extend([pred_label, f"{hc_prob:.4f}", f"{pd_prob:.4f}"])

            csv_data.append(row_data)

        print("\n" + "="*60)

        # 통계 출력
        print("\n=== 예측 통계 ===")
        print(f"총 파일 수: {len(valid_paths)}개\n")

        print("[베이스 모델 통계]")
        for base_name, stats in base_stats.items():
            print(f"  {base_name.upper()}: HC {stats['HC']}개 ({stats['HC']/len(valid_paths)*100:.1f}%), "
                  f"PD {stats['PD']}개 ({stats['PD']/len(valid_paths)*100:.1f}%)")

        print("\n[메타 러너 통계 (최종)]")
        for meta_name, stats in meta_stats.items():
            print(f"  {meta_name.upper()}: HC {stats['HC']}개 ({stats['HC']/len(valid_paths)*100:.1f}%), "
                  f"PD {stats['PD']}개 ({stats['PD']/len(valid_paths)*100:.1f}%)")

        # CSV 파일 저장
        save_results_csv(csv_data)

        # TXT 파일 저장
        save_results_txt(valid_paths, base_preds, predictions, base_stats, meta_stats)

    except Exception as e:
        print(f"예측 오류: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n예측 완료!")


def save_results_csv(csv_data):
    """예측 결과를 CSV로 저장"""
    import csv
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"prediction_results_{timestamp}.csv"

    try:
        with open(csv_filename, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerows(csv_data)
        print(f"\n✓ CSV 저장 완료: {csv_filename}")
    except Exception as e:
        print(f"\n✗ CSV 저장 실패: {e}")


def save_results_txt(valid_paths, base_preds, predictions, base_stats, meta_stats):
    """예측 결과를 TXT로 저장"""
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_filename = f"prediction_results_{timestamp}.txt"

    try:
        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("Parkinson Disease 예측 결과\n")
            f.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n\n")

            # 개별 파일 결과
            for i, path in enumerate(valid_paths):
                filename = os.path.basename(path)
                f.write(f"[{i+1}/{len(valid_paths)}] {filename}\n")
                f.write("-" * 60 + "\n")

                # 베이스 모델
                f.write("\n[베이스 모델 예측]\n")
                for base_name, base_pred in base_preds.items():
                    hc_prob = base_pred[i][0]
                    pd_prob = base_pred[i][1]
                    pred_label = "PD" if pd_prob > hc_prob else "HC"
                    f.write(f"  {base_name.upper()}: {pred_label} (HC={hc_prob:.1%}, PD={pd_prob:.1%})\n")

                # 메타 러너
                f.write("\n[최종 예측 - 메타 러너]\n")
                for meta_name, result in predictions.items():
                    pred_label = "PD" if result['predictions'][i] == 1 else "HC"
                    pd_prob = result['probabilities'][i][1]
                    hc_prob = result['probabilities'][i][0]
                    confidence = max(pd_prob, hc_prob)
                    f.write(f"  {meta_name.upper()}: {pred_label} (신뢰도: {confidence:.1%}, HC={hc_prob:.1%}, PD={pd_prob:.1%})\n")

                f.write("\n")

            # 통계
            f.write("="*60 + "\n")
            f.write("예측 통계\n")
            f.write("="*60 + "\n\n")
            f.write(f"총 파일 수: {len(valid_paths)}개\n\n")

            f.write("[베이스 모델 통계]\n")
            for base_name, stats in base_stats.items():
                f.write(f"  {base_name.upper()}: HC {stats['HC']}개 ({stats['HC']/len(valid_paths)*100:.1f}%), "
                       f"PD {stats['PD']}개 ({stats['PD']/len(valid_paths)*100:.1f}%)\n")

            f.write("\n[메타 러너 통계 (최종)]\n")
            for meta_name, stats in meta_stats.items():
                f.write(f"  {meta_name.upper()}: HC {stats['HC']}개 ({stats['HC']/len(valid_paths)*100:.1f}%), "
                       f"PD {stats['PD']}개 ({stats['PD']/len(valid_paths)*100:.1f}%)\n")

        print(f"✓ TXT 저장 완료: {txt_filename}")
    except Exception as e:
        print(f"✗ TXT 저장 실패: {e}")


if __name__ == "__main__":
    if MODE == 'train':
        train()
    elif MODE == 'predict':
        predict()
    else:
        print(f"오류: MODE는 'train' 또는 'predict'여야 합니다. 현재: {MODE}")
