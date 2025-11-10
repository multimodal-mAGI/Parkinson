

1. **환경 설정**
   pip install -r requirements.txt

## ⚙️ 전체 파이프라인 실행

conda env create -f environment.yml
conda activate parkinson_pose_env
python main.py

pip install -r requirements.txt

Pose npy 데이터: data/prefinal_preprocessed/

COM 시각화 영상: results/video_outputs_pose_only/

학습된 모델: results/models/best_pose_model.h5

평가 그래프: results/plots/confusion_matrix.png


