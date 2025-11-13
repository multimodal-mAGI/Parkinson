import torch
import numpy as np
import soundfile as sf
from tqdm import tqdm
from PIL import Image
from torchvision import transforms, models
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from torchvision.models import ResNet18_Weights

def extract_audio_embeddings(file_list, labels, model, processor, device):
    feats, y = [], []
    for path, label in tqdm(zip(file_list, labels), total=len(file_list), desc="Audio Embedding"):
        audio, _ = sf.read(path)
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
            pooled = torch.mean(out.hidden_states[-1], dim=1).squeeze().cpu().numpy()
        feats.append(pooled)
        y.append(label)
    return np.array(feats), np.array(y)


# mfcc 시각화 이미지 임베딩 진행 -> Resnet18 모델 사용
def extract_mfcc_embeddings(img_paths, labels, device):
    feats, y = [], []
    cnn_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    cnn_model.fc = torch.nn.Identity()
    cnn_model.eval().to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    for path, label in tqdm(zip(img_paths, labels), total=len(img_paths), desc="Image Embedding"):
        img = Image.open(path).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = cnn_model(img).squeeze().cpu().numpy()
        feats.append(feat)
        y.append(label)
    return np.array(feats), np.array(y)
