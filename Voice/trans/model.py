"""
Transformer Cross-Attention 기반 파킨슨병 예측 모델

아키텍처:
1. CNN 특징 (2048-dim)과 MFCC 특징 (12-dim)을 입력으로 받음
2. Cross-Attention을 사용하여 두 modality 간 상호작용 학습
3. 융합된 특징으로 파킨슨병 분류 수행

사전학습 모델:
- PyTorch의 nn.TransformerEncoderLayer 사용 (기본 아키텍처)
- BERT와 유사한 구조이지만 작은 규모로 학습
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Positional Encoding for Transformer"""

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: [seq_len, batch, d_model]
        """
        return x + self.pe[:x.size(0), :]


class CrossAttentionLayer(nn.Module):
    """Cross-Attention between CNN and MFCC features"""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(CrossAttentionLayer, self).__init__()

        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # Feed forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, query, key, value):
        """
        Args:
            query: [batch, seq_len, d_model] - CNN features
            key: [batch, seq_len, d_model] - MFCC features
            value: [batch, seq_len, d_model] - MFCC features

        Returns:
            output: [batch, seq_len, d_model]
        """
        # Multi-head attention
        attn_output, _ = self.multihead_attn(query, key, value)
        query = query + self.dropout1(attn_output)
        query = self.norm1(query)

        # Feed forward
        ff_output = self.linear2(self.dropout(F.relu(self.linear1(query))))
        query = query + self.dropout2(ff_output)
        query = self.norm2(query)

        return query


class TransformerCrossAttentionModel(nn.Module):
    """
    Transformer Cross-Attention 모델

    CNN 특징과 MFCC 특징을 cross-attention으로 융합하여 파킨슨병 예측
    """

    def __init__(
        self,
        cnn_feature_dim=2048,
        mfcc_feature_dim=12,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        num_classes=2
    ):
        """
        Args:
            cnn_feature_dim: CNN 특징 차원 (2048)
            mfcc_feature_dim: MFCC 특징 차원 (12)
            d_model: Transformer 모델 차원
            nhead: Multi-head attention의 head 개수
            num_layers: Transformer layer 개수
            dim_feedforward: Feed-forward 네트워크 차원
            dropout: Dropout 비율
            num_classes: 분류 클래스 개수 (2: HC/PD)
        """
        super(TransformerCrossAttentionModel, self).__init__()

        # Feature projection
        self.cnn_projection = nn.Linear(cnn_feature_dim, d_model)
        self.mfcc_projection = nn.Linear(mfcc_feature_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Self-attention for CNN features
        self.cnn_self_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers // 2
        )

        # Self-attention for MFCC features
        self.mfcc_self_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers // 2
        )

        # Cross-attention layers
        self.cross_attn_cnn_to_mfcc = CrossAttentionLayer(
            d_model, nhead, dim_feedforward, dropout
        )
        self.cross_attn_mfcc_to_cnn = CrossAttentionLayer(
            d_model, nhead, dim_feedforward, dropout
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Classification head
        self.classifier = nn.Linear(d_model // 2, num_classes)

        self.dropout = nn.Dropout(dropout)

    def forward(self, cnn_features, mfcc_features):
        """
        Args:
            cnn_features: [batch, 2048] CNN 특징
            mfcc_features: [batch, 12] MFCC 특징

        Returns:
            output: [batch, num_classes] 분류 로짓
        """
        # Feature projection: [batch, d_model]
        cnn_proj = self.cnn_projection(cnn_features)  # [batch, d_model]
        mfcc_proj = self.mfcc_projection(mfcc_features)  # [batch, d_model]

        # [batch, 1, d_model]로 확장 (sequence length = 1)
        cnn_proj = cnn_proj.unsqueeze(1)
        mfcc_proj = mfcc_proj.unsqueeze(1)

        # Self-attention
        cnn_self = self.cnn_self_attn(cnn_proj)  # [batch, 1, d_model]
        mfcc_self = self.mfcc_self_attn(mfcc_proj)  # [batch, 1, d_model]

        # Cross-attention
        # CNN attends to MFCC
        cnn_attended = self.cross_attn_cnn_to_mfcc(
            query=cnn_self,
            key=mfcc_self,
            value=mfcc_self
        )  # [batch, 1, d_model]

        # MFCC attends to CNN
        mfcc_attended = self.cross_attn_mfcc_to_cnn(
            query=mfcc_self,
            key=cnn_self,
            value=cnn_self
        )  # [batch, 1, d_model]

        # Concatenate attended features
        # [batch, 1, d_model * 2]
        fused = torch.cat([cnn_attended, mfcc_attended], dim=-1)

        # Remove sequence dimension: [batch, d_model * 2]
        fused = fused.squeeze(1)

        # Fusion
        fused = self.fusion(fused)  # [batch, d_model // 2]

        # Classification
        output = self.classifier(fused)  # [batch, num_classes]

        return output


class SimpleTransformerModel(nn.Module):
    """
    단순 Transformer 모델 (비교용)

    CNN과 MFCC 특징을 단순 concat 후 Transformer로 처리
    """

    def __init__(
        self,
        cnn_feature_dim=2048,
        mfcc_feature_dim=12,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        num_classes=2
    ):
        super(SimpleTransformerModel, self).__init__()

        # Feature projection
        input_dim = cnn_feature_dim + mfcc_feature_dim
        self.input_projection = nn.Linear(input_dim, d_model)

        # Transformer encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, cnn_features, mfcc_features):
        """
        Args:
            cnn_features: [batch, 2048]
            mfcc_features: [batch, 12]

        Returns:
            output: [batch, num_classes]
        """
        # Concatenate features
        combined = torch.cat([cnn_features, mfcc_features], dim=-1)  # [batch, 2060]

        # Project to d_model
        x = self.input_projection(combined)  # [batch, d_model]

        # Add sequence dimension
        x = x.unsqueeze(1)  # [batch, 1, d_model]

        # Transformer
        x = self.transformer(x)  # [batch, 1, d_model]

        # Remove sequence dimension
        x = x.squeeze(1)  # [batch, d_model]

        # Classification
        output = self.classifier(x)  # [batch, num_classes]

        return output


if __name__ == "__main__":
    # 모델 테스트
    print("=== Testing Transformer Cross-Attention Model ===\n")

    batch_size = 4
    cnn_dim = 2048
    mfcc_dim = 12

    # 더미 입력 생성
    cnn_features = torch.randn(batch_size, cnn_dim)
    mfcc_features = torch.randn(batch_size, mfcc_dim)

    # Cross-Attention 모델
    model_cross = TransformerCrossAttentionModel(
        cnn_feature_dim=cnn_dim,
        mfcc_feature_dim=mfcc_dim,
        d_model=256,
        nhead=8,
        num_layers=4,
        num_classes=2
    )

    # Simple Transformer 모델
    model_simple = SimpleTransformerModel(
        cnn_feature_dim=cnn_dim,
        mfcc_feature_dim=mfcc_dim,
        d_model=256,
        nhead=8,
        num_layers=4,
        num_classes=2
    )

    # Forward pass
    output_cross = model_cross(cnn_features, mfcc_features)
    output_simple = model_simple(cnn_features, mfcc_features)

    print(f"Input shapes:")
    print(f"  CNN features: {cnn_features.shape}")
    print(f"  MFCC features: {mfcc_features.shape}")
    print(f"\nOutput shapes:")
    print(f"  Cross-Attention model: {output_cross.shape}")
    print(f"  Simple model: {output_simple.shape}")
    print(f"\nModel parameters:")
    print(f"  Cross-Attention model: {sum(p.numel() for p in model_cross.parameters()):,}")
    print(f"  Simple model: {sum(p.numel() for p in model_simple.parameters()):,}")

    # Softmax 적용
    probs_cross = F.softmax(output_cross, dim=1)
    probs_simple = F.softmax(output_simple, dim=1)

    print(f"\nSample predictions (Cross-Attention):")
    print(f"  {probs_cross[0].detach().numpy()}")
    print(f"\nSample predictions (Simple):")
    print(f"  {probs_simple[0].detach().numpy()}")
