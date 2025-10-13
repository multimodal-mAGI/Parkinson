import os
import sys
import torch
import numpy as np
import warnings
from sklearn.model_selection import train_test_split

# 경고 메시지 숨기기
warnings.filterwarnings('ignore')

# 프로젝트 루트 디렉터리를 시스템 패스에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ensemble import StackingEnsemble
from evaluation import ModelEvaluator
from utils import load_audio_data, cross_validation_ensemble


def setup_device():
    """GPU 확인 및 메모리 최적화"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"사용 장치: {device}")

    if device == 'cuda':
        print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
        print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

        # 메모리 최적화 설정
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()

        # 메모리 분할 설정 (OutOfMemoryError 방지)
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    return device


def get_data_paths():
    """데이터 경로 입력받기"""
    print("\n=== 데이터 경로 설정 ===")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    default_healthy = os.path.join(current_dir, "data", "testdata_KO", "healthy")
    default_parkinson = os.path.join(current_dir, "data", "testdata_KO", "parkinson")

    print(f"기본 건강한 사람 데이터 경로: {default_healthy}")
    healthy_path = input("건강한 사람 데이터 경로 (Enter: 기본값 사용): ").strip()
    if not healthy_path:
        healthy_path = default_healthy

    print(f"기본 파킨슨 환자 데이터 경로: {default_parkinson}")
    parkinson_path = input("파킨슨 환자 데이터 경로 (Enter: 기본값 사용): ").strip()
    if not parkinson_path:
        parkinson_path = default_parkinson

    # 경로 검증
    if not os.path.exists(healthy_path):
        print(f"경고: 건강한 사람 데이터 경로가 존재하지 않습니다: {healthy_path}")
        return None, None

    if not os.path.exists(parkinson_path):
        print(f"경고: 파킨슨 환자 데이터 경로가 존재하지 않습니다: {parkinson_path}")
        return None, None

    return healthy_path, parkinson_path


def get_model_save_path():
    """모델 저장 경로 입력받기"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    default_path = os.path.join(current_dir, "models", "finetune")

    print(f"\n기본 모델 저장 경로: {default_path}")
    model_path = input("모델 저장 경로 (Enter: 기본값 사용): ").strip()

    if not model_path:
        model_path = default_path

    return model_path


def train_mode():
    """모델 학습 모드"""
    print("\n" + "="*60)
    print("모드 1: 모델 학습 및 훈련")
    print("="*60)

    # 데이터 경로 입력
    healthy_path, parkinson_path = get_data_paths()
    if healthy_path is None or parkinson_path is None:
        print("데이터 경로 설정 실패")
        return

    # 모델 저장 경로 입력
    model_save_dir = get_model_save_path()

    # 데이터 로드
    print("\n=== 데이터 로딩 ===")
    X, y = load_audio_data(healthy_path, parkinson_path)

    if len(X) == 0:
        print("데이터를 찾을 수 없습니다.")
        return

    # 장치 설정
    device = setup_device()

    # 앙상블 모델 생성
    ensemble = StackingEnsemble(device=device)

    # 훈련/테스트 분할
    print("\n=== 데이터 분할 ===")
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"훈련 데이터: {len(train_paths)}개 (HC: {np.sum(train_labels == 0)}, PD: {np.sum(train_labels == 1)})")
    print(f"테스트 데이터: {len(test_paths)}개 (HC: {np.sum(test_labels == 0)}, PD: {np.sum(test_labels == 1)})")

    # 훈련 파라미터 설정
    print("\n=== 훈련 파라미터 설정 ===")
    try:
        epochs = int(input("Epochs (기본값: 20): ").strip() or "20")
        batch_size = int(input("Batch Size (기본값: 8): ").strip() or "8")
    except ValueError:
        print("잘못된 입력입니다. 기본값을 사용합니다.")
        epochs = 20
        batch_size = 8

    # 모델 훈련
    print("\n=== 모델 훈련 시작 ===")
    try:
        base_predictions = ensemble.train_base_models(train_paths, train_labels, epochs=epochs, batch_size=batch_size)
        ensemble.train_meta_learners(base_predictions, train_labels)

        print("\n" + "="*60)
        print("훈련 완료!")
        print("="*60)

    except Exception as e:
        print(f"훈련 중 오류 발생: {e}")
        print("메모리 부족이 원인일 수 있습니다. batch_size를 줄이거나 epochs를 줄여보세요.")
        return

    # 훈련 결과 평가
    print("\n=== 훈련 결과 평가 ===")
    try:
        evaluation_dir = "./evaluation_results"
        evaluator = ModelEvaluator()
        results = evaluator.evaluate_models(ensemble, test_paths, test_labels,
                                           save_plots=True, plots_dir=evaluation_dir)

        print("\n성능 평가 완료!")

        # 결과 요약 출력
        if results and results.get('plot_paths'):
            print(f"\n분석 파일 저장 위치: {evaluation_dir}")
            print("생성된 파일:")
            for file_type, path in results['plot_paths'].items():
                print(f"  • {file_type}: {os.path.basename(path)}")

    except Exception as e:
        print(f"평가 중 오류 발생: {e}")

    # 모델 저장
    print("\n=== 모델 저장 ===")
    save_choice = input("훈련된 모델을 저장하시겠습니까? (y/n, 기본값: y): ").lower().strip()

    if save_choice != 'n':
        try:
            saved_dir = ensemble.save_models(model_save_dir)
            print(f"모델이 {saved_dir}에 저장되었습니다.")
        except Exception as e:
            print(f"모델 저장 실패: {e}")

    # 교차 검증 (선택사항)
    print("\n=== 교차 검증 (선택사항) ===")
    cv_choice = input("교차 검증을 실행하시겠습니까? (y/n, 기본값: n): ").lower().strip()

    if cv_choice == 'y':
        try:
            print("교차 검증 시작... (시간이 오래 걸릴 수 있습니다)")
            cv_folds = int(input("CV Folds (기본값: 3): ").strip() or "3")
            base_avg, meta_avg = cross_validation_ensemble(X, y, cv_folds=cv_folds)

            print("\n교차 검증 결과:")
            print("\n[Base Models]")
            for model_name, metrics in base_avg.items():
                print(f"\n{model_name.upper()}:")
                for metric_name, values in metrics.items():
                    print(f"  {metric_name}: {values['mean']:.4f} (+/- {values['std']:.4f})")

            print("\n[Meta Learners]")
            for model_name, metrics in meta_avg.items():
                print(f"\n{model_name.upper()}:")
                for metric_name, values in metrics.items():
                    print(f"  {metric_name}: {values['mean']:.4f} (+/- {values['std']:.4f})")

        except Exception as e:
            print(f"교차 검증 중 오류 발생: {e}")

    print("\n" + "="*60)
    print("학습 모드 완료!")
    print("="*60)


def predict_mode():
    """예측 전용 모드"""
    print("\n" + "="*60)
    print("모드 2: 학습된 모델로 예측")
    print("="*60)

    # 모델 경로 입력
    model_save_dir = get_model_save_path()

    # 모델 존재 확인
    if not os.path.exists(model_save_dir):
        print(f"오류: 모델 디렉토리가 존재하지 않습니다: {model_save_dir}")
        print("먼저 모드 1에서 모델을 훈련하고 저장해야 합니다.")
        return

    if not os.path.exists(os.path.join(model_save_dir, "metadata.json")):
        print(f"오류: 유효한 모델이 없습니다: {model_save_dir}")
        return

    # 장치 설정
    device = setup_device()

    # 앙상블 모델 생성 및 로드
    print("\n=== 모델 로딩 ===")
    ensemble = StackingEnsemble(device=device)

    try:
        ensemble.load_models(model_save_dir)
        print("모델 로드 완료!")
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        return

    # 예측 데이터 경로 입력
    print("\n=== 예측 데이터 설정 ===")
    print("옵션 1: 디렉토리 경로 입력 (WAV/MP3 파일들이 있는 폴더)")
    print("옵션 2: 파일 경로 입력 (쉼표로 구분)")

    data_input = input("\n데이터 경로 입력: ").strip()

    if not data_input:
        print("데이터 경로가 입력되지 않았습니다.")
        return

    # 경로 파싱
    audio_paths = []

    # 디렉토리인 경우
    if os.path.isdir(data_input):
        print(f"디렉토리에서 오디오 파일 검색 중: {data_input}")
        import glob
        for ext in ["*.wav", "*.mp3"]:
            audio_paths.extend(glob.glob(os.path.join(data_input, ext)))

        if not audio_paths:
            print("오디오 파일을 찾을 수 없습니다.")
            return

    # 파일 경로들인 경우
    else:
        audio_paths = [path.strip() for path in data_input.split(',')]

    # 파일 존재 확인
    valid_paths = []
    for path in audio_paths:
        if os.path.exists(path):
            valid_paths.append(path)
        else:
            print(f"경고: 파일을 찾을 수 없습니다: {path}")

    if not valid_paths:
        print("유효한 오디오 파일이 없습니다.")
        return

    print(f"\n총 {len(valid_paths)}개 파일 발견")

    # 예측 수행
    print("\n=== 예측 수행 중 ===")
    try:
        predictions, base_preds = ensemble.predict(valid_paths)

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

                print(f"\n  {meta_name.upper()}:")
                print(f"    예측 결과: {pred_label}")
                print(f"    신뢰도: {confidence:.1%}")
                print(f"    확률 분포: HC={hc_prob:.1%}, PD={pd_prob:.1%}")

        print("\n" + "="*60)

        # 결과 저장 옵션
        save_result = input("\n결과를 파일로 저장하시겠습니까? (y/n, 기본값: n): ").lower().strip()

        if save_result == 'y':
            result_file = input("저장 파일명 (기본값: prediction_results.txt): ").strip()
            if not result_file:
                result_file = "prediction_results.txt"

            try:
                with open(result_file, 'w', encoding='utf-8') as f:
                    f.write("="*60 + "\n")
                    f.write("Parkinson Disease 예측 결과\n")
                    f.write("="*60 + "\n\n")

                    for i, path in enumerate(valid_paths):
                        filename = os.path.basename(path)
                        f.write(f"[{i+1}/{len(valid_paths)}] {filename}\n")
                        f.write("-" * 60 + "\n")

                        for meta_name, result in predictions.items():
                            pred_label = "Parkinson Disease" if result['predictions'][i] == 1 else "Healthy Control"
                            pd_prob = result['probabilities'][i][1]
                            hc_prob = result['probabilities'][i][0]
                            confidence = max(pd_prob, hc_prob)

                            f.write(f"\n{meta_name.upper()}:\n")
                            f.write(f"  예측 결과: {pred_label}\n")
                            f.write(f"  신뢰도: {confidence:.1%}\n")
                            f.write(f"  확률 분포: HC={hc_prob:.1%}, PD={pd_prob:.1%}\n")

                        f.write("\n")

                print(f"결과가 {result_file}에 저장되었습니다.")

            except Exception as e:
                print(f"결과 저장 실패: {e}")

    except Exception as e:
        print(f"예측 중 오류 발생: {e}")
        return

    print("\n" + "="*60)
    print("예측 모드 완료!")
    print("="*60)


def main():
    """메인 실행 함수"""
    print("="*60)
    print("Parkinson's Disease Voice Classification System")
    print("="*60)

    print("\n실행 모드를 선택하세요:")
    print("1. 모델 학습 및 훈련 (데이터 학습 → 모델 저장 → 성능 평가)")
    print("2. 학습된 모델로 예측 (저장된 모델 로드 → 새 데이터 예측)")
    print("3. 종료")

    mode = input("\n모드 선택 (1-3): ").strip()

    if mode == '1':
        train_mode()

    elif mode == '2':
        predict_mode()

    elif mode == '3':
        print("프로그램을 종료합니다.")
        return

    else:
        print("잘못된 선택입니다. 1, 2, 또는 3을 입력하세요.")
        return

    # 추가 작업 여부 확인
    print("\n")
    continue_choice = input("다른 작업을 수행하시겠습니까? (y/n): ").lower().strip()
    if continue_choice == 'y':
        main()
    else:
        print("\n프로그램을 종료합니다.")


if __name__ == "__main__":
    main()
