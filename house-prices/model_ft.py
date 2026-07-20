import torch
import torch.nn as nn


class FeatureTokenizer(nn.Module):
    """각 특성을 d차원 토큰으로 변환 (수치형은 스케일링된 벡터, 범주형은 임베딩)"""
    def __init__(self, n_num, cardinalities, d_token):
        super().__init__()
        # 수치형: 특성마다 고유 벡터를 곱하고 편향을 더함 → token = x_i * W_i + b_i
        self.num_weight = nn.Parameter(torch.empty(n_num, d_token))
        self.num_bias   = nn.Parameter(torch.empty(n_num, d_token))
        nn.init.normal_(self.num_weight, std=0.02)
        nn.init.normal_(self.num_bias,   std=0.02)

        # 범주형: 각 범주에 학습되는 벡터
        self.cat_embeds = nn.ModuleList([nn.Embedding(c, d_token) for c in cardinalities])

        # CLS 토큰 — 최종 예측에 사용할 요약 토큰
        self.cls = nn.Parameter(torch.empty(1, 1, d_token))
        nn.init.normal_(self.cls, std=0.02)

    def forward(self, x_num, x_cat):
        B = x_num.size(0)
        num_tokens = x_num.unsqueeze(-1) * self.num_weight + self.num_bias   # (B, n_num, d)
        cat_tokens = torch.stack([e(x_cat[:, i]) for i, e in enumerate(self.cat_embeds)],
                                 dim=1)                                       # (B, n_cat, d)
        cls = self.cls.expand(B, -1, -1)                                      # (B, 1, d)
        return torch.cat([cls, num_tokens, cat_tokens], dim=1)                # (B, 1+n_num+n_cat, d)


class TransformerBlock(nn.Module):
    """Pre-norm 트랜스포머 블록 (어텐션 + FFN, 각각 잔차 연결)"""
    def __init__(self, d_token, n_heads, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_token)
        self.attn  = nn.MultiheadAttention(d_token, n_heads,
                                           dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_token)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, d_token * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_token * 2, d_token),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)      # 특성 간 상호작용 학습
        x = x + self.drop(a)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class FTTransformer(nn.Module):
    """Feature Tokenizer + Transformer (간소화 버전)
       ※ d_token은 n_heads로 나누어떨어져야 함"""
    def __init__(self, n_num, cardinalities, d_token=32,
                 n_blocks=2, n_heads=4, dropout=0.1):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_num, cardinalities, d_token)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_token, n_heads, dropout) for _ in range(n_blocks)
        ])
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, 1)

    def forward(self, x_num, x_cat):
        x = self.tokenizer(x_num, x_cat)
        for blk in self.blocks:
            x = blk(x)
        cls = self.norm(x[:, 0])                            # CLS 토큰만 사용
        return self.head(cls)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())