import torch
import torch.nn as nn


def emb_dim(cardinality, mult=1, cap=16):
    """카디널리티별 임베딩 차원.
       mult: 배수 (1이면 √카디널리티, 2면 그 2배)
       cap : 상한선
       ※ v23에서 cap만으로는 차원을 키울 수 없음이 확인되어 mult를 도입"""
    return min(cap, max(2, int(mult * (cardinality ** 0.5))))


class HousePriceEmbedModel(nn.Module):
    """범주형은 임베딩(밀집 벡터), 수치형은 그대로 → 결합 후 MLP"""
    def __init__(self, n_num, cardinalities, hidden=(32, 8), p=0.1,
                 dim_mult=1, dim_cap=16):                        # ★[v24] dim_mult 추가
        super().__init__()
        dims = [emb_dim(c, dim_mult, dim_cap) for c in cardinalities]
        self.embeds = nn.ModuleList([nn.Embedding(c, d)
                                     for c, d in zip(cardinalities, dims)])
        self.emb_total = sum(dims)
        self.input_dim = n_num + self.emb_total

        layers, prev = [], self.input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_num, x_cat):
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeds)]
        x = torch.cat([x_num] + embs, dim=1)
        return self.mlp(x)