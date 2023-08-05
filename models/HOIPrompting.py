"""
HOI Prompting Module
"""

from clip.model import QuickGELU
from timm.models.layers import trunc_normal_
import torch
from torch import nn
import sys
sys.path.append("../")


class MulitHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            # nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            # nn.init.xavier_uniform_(m.weight)

    def forward(self, q, k, v):
        B, N, C = q.shape
        B, M, C = k.shape
        q = self.q_proj(q).reshape(B, N, self.num_heads, C //
                                   self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(k).reshape(B, M, self.num_heads, C //
                                   self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(v).reshape(B, M, self.num_heads, C //
                                   self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PromptGeneratorLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.,
    ):
        super().__init__()
        self.cross_attn = MulitHeadAttention(d_model, nhead, proj_drop=dropout)
        # self.cross_attn = nn.MultiheadAttention(d_model, nhead)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            QuickGELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            # nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            # nn.init.xavier_uniform_(m.weight)

    def forward(self, x, visual):
        q = k = v = self.norm1(x)
        x = x + self.cross_attn(q, visual, visual)
        # x = x + self.cross_attn(q, visual, visual, need_weights=False)
        x = x + self.dropout(self.mlp(self.norm3(x)))
        return x


class HOIPrompt(nn.Module):
    def __init__(self, layers=2, embed_dim=64, alpha=1):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.decoder = nn.ModuleList([PromptGeneratorLayer(
            embed_dim, embed_dim//64) for _ in range(layers)])
        self.alpha = nn.Parameter(torch.ones(embed_dim) * alpha)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            # nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            # nn.init.xavier_uniform_(m.weight)

    def forward(self, text, visual):
        # B, N, C = visual.shape
        visual = self.norm(visual)
        for layer in self.decoder:
            text = layer(text, visual)
            # _text = layer(text, visual)
            # text = text + _text
        # print("alpha: ", self.alpha)

        # return self.alpha * text
        return self.alpha * text + text


if __name__ == '__main__':
    f_t = torch.randn(10, 50, 64)  # Q
    f_i = torch.randn(10, 1, 64)   # K, V
    model = HOIPrompt()
    prompt_fea = model(f_t, f_i)
    print("prompt feature: ", prompt_fea)
    print(prompt_fea.shape)  # torch.Size([10, 50, 64])
