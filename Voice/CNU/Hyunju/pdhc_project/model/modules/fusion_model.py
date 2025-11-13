import torch
import torch.nn as nn
from transformers import BertModel

class CrossAttentionFusion(nn.Module):
    def __init__(self, audio_dim=768, img_dim=512, hidden_dim=512, num_heads=8):
        super().__init__()
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.img_proj = nn.Linear(img_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, audio_feat, img_feat):
        q = self.audio_proj(audio_feat).unsqueeze(1)
        k = self.img_proj(img_feat).unsqueeze(1)
        v = k
        out, _ = self.cross_attn(q, k, v)
        return self.fc(self.norm(out + q)).squeeze(1)


class MultiModalClassifier(nn.Module):
    def __init__(self, hidden_dim=512):
        super().__init__()
        self.fusion = CrossAttentionFusion(hidden_dim=hidden_dim)
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.fc_in = nn.Linear(hidden_dim, 768)
        self.fc_out = nn.Linear(768, 2)

    def forward(self, audio_feat, img_feat):
        fused = self.fusion(audio_feat, img_feat)
        x = self.fc_in(fused).unsqueeze(1)
        outputs = self.bert(inputs_embeds=x)
        return self.fc_out(outputs.pooler_output)
