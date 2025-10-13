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


def load_or_train_models(ensemble, X, y, model_save_dir):
    """기존 모델 로드 또는 새로 훈련"""
    load_existing = False
    
    if os.path.exists(model_save_dir) and os.path.exists(os.path.join(model_save_dir, "metadata.json")):
        print(f"\n기존 훈련된 모델이 {model_save_dir}에서 발견되었습니다.")
        user_choice = input("기존 모델을 로드하시겠습니까? (y/n, 기본값: y): ").lower().strip()
        
        if user_choice != 'n':
            try:
                ensemble.load_models(model_save_dir)
                load_existing = True
                print("기존 모델 로드 완료!")
            except Exception as e:
                print(f"모델 로드 실패: {e}")
                print("새로 훈련을 진행합니다.")
                load_existing = False
    
    if not load_existing:
        print("\n1. Ensemble Model Training")
        
        # 훈련/테스트 분할
        train_paths, test_paths, train_labels, test_labels = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"훈련 데이터: {len(train_paths)}개 (HC: {np.sum(train_labels == 0)}, PD: {np.sum(train_labels == 1)})")
        print(f"테스트 데이터: {len(test_paths)}개 (HC: {np.sum(test_labels == 0)}, PD: {np.sum(test_labels == 1)})")
        
        # 모델 훈련 (메모리 절약을 위한 보수적 설정)
        try:
            base_predictions = ensemble.train_base_models(train_paths, train_labels, epochs=20, batch_size=8)
            ensemble.train_meta_learners(base_predictions, train_labels)
            
            print("\n=== 훈련 완료 ===")
            print("베이스 모델들과 메타 러너들이 성공적으로 훈련되었습니다.")
            
            # 모델 저장
            save_choice = input("훈련된 모델을 저장하시겠습니까? (y/n, 기본값: y): ").lower().strip()
            if save_choice != 'n':
                try:
                    saved_dir = ensemble.save_models(model_save_dir)
                    print(f"모델이 {saved_dir}에 저장되었습니다.")
                except Exception as e:
                    print(f"모델 저장 실패: {e}")
        
        except Exception as e:
            print(f"훈련 중 오류 발생: {e}")
            print("메모리 부족이 원인일 수 있습니다. batch_size를 줄이거나 epochs를 줄여보세요.")
            return None, None, None, None
    
    else:
        # 기존 모델을 로드한 경우에도 테스트 데이터 분할
        _, test_paths, _, test_labels = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    
    return ensemble, test_paths, test_labels, load_existing


def evaluate_model_performance(ensemble, test_paths, test_labels):
    """모델 성능 평가"""
    print("\n2. Model Evaluation with Visualization")
    if ensemble.is_trained:
        try:
            # 평가 결과 저장 디렉토리 설정
            evaluation_dir = "./evaluation_results"
            
            evaluator = ModelEvaluator()
            results = evaluator.evaluate_models(ensemble, test_paths, test_labels, 
                                               save_plots=True, plots_dir=evaluation_dir)
            print("성능 평가 및 시각화 완료!")
            
            # 생성된 파일들 요약 출력
            if results['plot_paths']:
                print(f"\n=== Generated Analysis Files ===")
                print(f"Location: {evaluation_dir}")
                print("Files created:")
                for file_type, path in results['plot_paths'].items():
                    print(f"  • {file_type}: {os.path.basename(path)}")
            
            return results
        
        except Exception as e:
            print(f"평가 중 오류 발생: {e}")
            return None
    else:
        print("훈련된 모델이 없어 평가를 수행할 수 없습니다.")
        return None


def demonstrate_prediction(ensemble, test_paths, test_labels):
    """새로운 데이터 예측 예시"""
    print("\n3. New Data Prediction Example")
    if ensemble.is_trained and len(test_paths) > 0:
        # 테스트 데이터 중 일부를 사용한 예측 예시
        sample_paths = test_paths[:5]  # 처음 5개 샘플
        print(f"Performing predictions on {len(sample_paths)} sample files...")
        
        try:
            predictions, base_preds = ensemble.predict(sample_paths)
            
            print("\n=== Prediction Results ===")
            for i, path in enumerate(sample_paths):
                filename = os.path.basename(path)
                actual_label = "Parkinson Disease" if test_labels[i] == 1 else "Healthy Control"
                print(f"\nFile: {filename}")
                print(f"Ground Truth: {actual_label}")
                
                print("Model Predictions:")
                for meta_name, result in predictions.items():
                    pred_label = "Parkinson Disease" if result['predictions'][i] == 1 else "Healthy Control"
                    confidence = result['probabilities'][i].max()
                    pd_prob = result['probabilities'][i][1]  # Parkinson Disease probability
                    hc_prob = result['probabilities'][i][0]  # Healthy Control probability
                    
                    print(f"  {meta_name.upper()}:")
                    print(f"    Prediction: {pred_label}")
                    print(f"    Confidence: {confidence:.3f}")
                    print(f"    Probabilities: HC={hc_prob:.3f}, PD={pd_prob:.3f}")
                print("-" * 50)
        
        except Exception as e:
            print(f"예측 중 오류 발생: {e}")


def run_cross_validation(X, y, load_existing):
    """교차 검증 실행"""
    print("\n4. Cross Validation (Optional)")
    if not load_existing:  # 새로 훈련한 경우에만 제공
        print("교차 검증을 실행하시겠습니까? 시간이 오래 걸릴 수 있습니다.")
        user_input = input("실행하려면 'y'를 입력하세요 (기본값: n): ").lower().strip()
        
        if user_input == 'y':
            try:
                print("교차 검증 시작...")
                base_avg, meta_avg = cross_validation_ensemble(X, y, cv_folds=3)
                
                print("\n=== 교차 검증 결과 요약 ===")
                print("\n--- Base Models Average Performance ---")
                for model_name, metrics in base_avg.items():
                    print(f"\n{model_name.upper()}:")
                    for metric_name, values in metrics.items():
                        print(f"  {metric_name}: {values['mean']:.4f} (+/- {values['std']:.4f})")
                
                print("\n--- Meta Learners Average Performance ---")
                for model_name, metrics in meta_avg.items():
                    print(f"\n{model_name.upper()}:")
                    for metric_name, values in metrics.items():
                        print(f"  {metric_name}: {values['mean']:.4f} (+/- {values['std']:.4f})")
            
            except Exception as e:
                print(f"교차 검증 중 오류 발생: {e}")
        else:
            print("교차 검증을 건너뜁니다.")
    else:
        print("기존 모델을 로드했으므로 교차 검증을 건너뜁니다.")


def prediction_only_mode(model_save_dir, audio_paths=None):
    """Prediction 전용 모드"""
    print("\n=== Prediction-Only Mode ===")

    # 장치 설정
    device = setup_device()

    # 앙상블 모델 생성
    ensemble = StackingEnsemble(device=device)

    # 저장된 모델 로드
    if not os.path.exists(model_save_dir):
        print(f"Error: 모델 디렉토리 {model_save_dir}가 존재하지 않습니다.")
        print("먼저 모델을 훈련하고 저장해야 합니다.")
        return None

    try:
        ensemble.load_models(model_save_dir)
        print("모델 로드 완료!")
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        return None

    # 예측할 오디오 파일 경로 입력
    if audio_paths is None:
        print("\n예측할 오디오 파일 경로를 입력하세요 (쉼표로 구분):")
        print("예: path/to/audio1.wav, path/to/audio2.wav")
        user_input = input(">>> ").strip()

        if not user_input:
            print("오디오 파일 경로가 입력되지 않았습니다.")
            return None

        audio_paths = [path.strip() for path in user_input.split(',')]

    # 파일 존재 확인
    valid_paths = []
    for path in audio_paths:
        if os.path.exists(path):
            valid_paths.append(path)
        else:
            print(f"Warning: 파일을 찾을 수 없습니다: {path}")

    if not valid_paths:
        print("유효한 오디오 파일이 없습니다.")
        return None

    print(f"\n{len(valid_paths)}개 파일 예측 중...")

    try:
        predictions, base_preds = ensemble.predict(valid_paths)

        print("\n=== Prediction Results ===")
        for i, path in enumerate(valid_paths):
            filename = os.path.basename(path)
            print(f"\n[{i+1}] File: {filename}")

            for meta_name, result in predictions.items():
                pred_label = "Parkinson Disease" if result['predictions'][i] == 1 else "Healthy Control"
                pd_prob = result['probabilities'][i][1]
                hc_prob = result['probabilities'][i][0]
                confidence = max(pd_prob, hc_prob)

                print(f"  {meta_name.upper()}:")
                print(f"    Prediction: {pred_label}")
                print(f"    Confidence: {confidence:.1%}")
                print(f"    Probabilities: HC={hc_prob:.1%}, PD={pd_prob:.1%}")
            print("-" * 60)

        return predictions

    except Exception as e:
        print(f"예측 중 오류 발생: {e}")
        return None


def main():
    """메인 실행 함수"""
    print("=== Parkinson's Disease Voice Classification System ===\n")
    print("실행 모드를 선택하세요:")
    print("1. 전체 실행 (훈련/평가/예측)")
    print("2. Prediction만 수행 (기존 모델 사용)")

    mode = input("\n모드 선택 (1 or 2, 기본값: 1): ").strip()

    # 데이터 및 모델 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    healthy_path = os.path.join(current_dir, "data", "testdata_KO", "healthy")
    parkinson_path = os.path.join(current_dir, "data", "testdata_KO", "parkinson")
    model_save_dir = os.path.join(current_dir, "models", "finetune")

    if mode == '2':
        # Prediction 전용 모드
        prediction_only_mode(model_save_dir)
        return

    # 전체 실행 모드 (기본)
    print("\n=== Full Pipeline Mode ===")

    # 데이터 로드
    X, y = load_audio_data(healthy_path, parkinson_path)

    # 장치 설정
    device = setup_device()

    # 앙상블 모델 생성
    ensemble = StackingEnsemble(device=device)

    # 모델 훈련 또는 로드
    ensemble, test_paths, test_labels, load_existing = load_or_train_models(
        ensemble, X, y, model_save_dir
    )

    if ensemble is None:
        print("모델 초기화에 실패했습니다.")
        return

    # 성능 평가
    results = evaluate_model_performance(ensemble, test_paths, test_labels)

    # 예측 예시
    demonstrate_prediction(ensemble, test_paths, test_labels)

    # 교차 검증
    run_cross_validation(X, y, load_existing)

    print("\n=== Program Completed Successfully! ===")
    print("Summary:")
    print("• Models have been trained and saved for future use")
    print("• Comprehensive evaluation metrics and visualizations generated")
    print("• Ready for deployment or further analysis")
    print("\nNext time you run this script:")
    print("• Choose mode 2 for quick predictions with saved models")
    print("• All analysis files are preserved for reference")


if __name__ == "__main__":
    main()