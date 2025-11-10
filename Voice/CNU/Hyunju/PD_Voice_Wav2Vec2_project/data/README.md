
### 1. IPVS 데이터셋
- Italian_Parkinsons_Voice_and_Speech
- 다운로드 경로 : https://huggingface.co/datasets/birgermoell/Italian_Parkinsons_Voice_and_Speech

>> 훈련용 데이터 : D1, D2, VA1~VU2 사용
>> 테스트 데이터 : B1, B2, FB1, FB2 사용



### 2. Voice Samples 데이터셋
- Voice Samples for Patients with Parkinson’s Disease and Healthy Controls
- 다운로드 경로 : https://figshare.com/articles/dataset/Voice_Samples_for_Patients_with_Parkinson_s_Disease_and_Healthy_Controls/23849127

>> 훈련용 데이터 : 전체 데이터셋 사용


### 3. 훈련 데이터셋

- IPVS 데이터 + Voice Sapmles 데이터 합쳐서 사용.

### 4. 테스트 데이터셋

- IPVS 데이터 중 B1, B2, FB1, FB2 사용 (훈련에 사용되지 않은 데이터)
