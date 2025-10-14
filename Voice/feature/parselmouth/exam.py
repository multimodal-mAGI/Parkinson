import os
import pandas as pd
from extractor import extract_all_features

base_folder = "./kr3/"
data = []

for folder_name in os.listdir(base_folder):
    if folder_name in ['HC', 'PD']:
        label = folder_name
        full_path = os.path.join(base_folder, folder_name)
        for filename in os.listdir(full_path):
            if filename.endswith(".wav"):
                file_path = os.path.join(full_path, filename)
                features = extract_all_features(file_path)
                features['label'] = label
                data.append(features)

df = pd.DataFrame(data)
df.to_csv("음성_특징_데이터.csv", index=False)
print("음성 특징 추출 및 CSV 파일 저장이 완료되었습니다.")
