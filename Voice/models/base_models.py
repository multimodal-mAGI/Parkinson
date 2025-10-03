import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import Wav2Vec2Model


class CNNModel(nn.Module):
    """CNN 기반 모델 (ResNet-50)"""
    
    def __init__(self, num_classes=2):
        super(CNNModel, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(2048, num_classes)
        
    def forward(self, x):
        return self.resnet(x)


class RNNModel(nn.Module):
    """RNN 기반 모델 (LSTM)"""
    
    def __init__(self, input_size=64, hidden_size=128, num_layers=2, num_classes=2):
        super(RNNModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.3, bidirectional=True)
        self.attention = nn.MultiheadAttention(hidden_size*2, num_heads=4, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size*2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        pooled = torch.mean(attn_out, dim=1)
        return self.classifier(pooled)


class TransformerModel(nn.Module):
    """수정된 Transformer 기반 모델 - 수치 안정성 개선"""
    
    def __init__(self, num_classes=2):
        super(TransformerModel, self).__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base",
            low_cpu_mem_usage=True
        )
        
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
        
        self.feature_projection = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.LayerNorm(64),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        if torch.isnan(x).any():
            print("Warning: NaN detected in Transformer input")
            x = torch.nan_to_num(x, nan=0.0)
        
        with torch.no_grad():
            outputs = self.wav2vec2(x)
        
        pooled = torch.mean(outputs.last_hidden_state, dim=1)
        
        if torch.isnan(pooled).any():
            print("Warning: NaN detected after Wav2Vec2")
            pooled = torch.nan_to_num(pooled, nan=0.0)
        
        features = self.feature_projection(pooled)
        
        if torch.isnan(features).any():
            print("Warning: NaN detected after feature projection")
            features = torch.nan_to_num(features, nan=0.0)
        
        output = self.classifier(features)
        
        if torch.isnan(output).any():
            print("Warning: NaN detected in final output")
            output = torch.nan_to_num(output, nan=0.0)
        
        return output


class HybridModel(nn.Module):
    """CNN-LSTM 하이브리드 모델"""
    
    def __init__(self, num_classes=2):
        super(HybridModel, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        
        self.lstm = nn.LSTM(128, 64, num_layers=2,
                           batch_first=True, dropout=0.3, bidirectional=True)
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        conv_out = self.conv_layers(x)
        conv_out = conv_out.transpose(1, 2)
        lstm_out, _ = self.lstm(conv_out)
        pooled = torch.mean(lstm_out, dim=1)
        return self.classifier(pooled)