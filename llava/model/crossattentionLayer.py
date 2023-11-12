import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import einsum

class BroadMultiHeadAttention(nn.Module):
    def __init__(self, dim, heads):
        super(BroadMultiHeadAttention, self).__init__()
        self.dim = dim
        self.heads = heads
        self.scale = (dim/heads) ** -0.5
        self.attend = nn.Softmax(dim=-1)

    def attend_with_rpe(self, Q, K):
        if Q.shape[0] == 1:
            Q = rearrange(Q.squeeze(0), 'i (heads d) -> heads i d', heads=self.heads)
            K = rearrange(K, 'b j (heads d) -> b heads j d', heads=self.heads)

            dots = einsum('hid, bhjd -> bhij', Q, K) * self.scale # (b hw) heads 1 pointnum
        else:
            Q = rearrange(Q, 'b i (heads d) -> b heads i d', heads=self.heads)
            K = rearrange(K, 'b j (heads d) -> b heads j d', heads=self.heads)

            dots = einsum('bhid, bhjd -> bhij', Q, K) * self.scale # (b hw) heads 1 pointnum

        return self.attend(dots)

    def forward(self, Q, K, V):
        attn = self.attend_with_rpe(Q, K)
        B, _, _ = K.shape
        _, N, _ = Q.shape

        V = rearrange(V, 'b j (heads d) -> b heads j d', heads=self.heads)

        out = einsum('bhij, bhjd -> bhid', attn, V)
        out = rearrange(out, 'b heads n d -> b n (heads d)', b=B, n=N)

        return out

class CrossAttentionLayer(nn.Module):
    def __init__(self, qk_dim, v_dim, query_token_dim, tgt_token_dim, num_heads=8, attn_drop=0., proj_drop=0., drop_path=0., dropout=0.):
        super(CrossAttentionLayer, self).__init__()
        assert qk_dim % num_heads == 0, f"dim {qk_dim} should be divided by num_heads {num_heads}."
        assert v_dim % num_heads == 0, f"dim {v_dim} should be divided by num_heads {num_heads}."
        """
            Query Token:    [N, C]  -> [N, qk_dim]  (Q)
            Target Token:   [M, D]  -> [M, qk_dim]  (K),    [M, v_dim]  (V)
        """
        self.num_heads = num_heads
        head_dim = qk_dim // num_heads
        self.scale = head_dim ** -0.5

        self.norm1 = nn.LayerNorm(query_token_dim)
        self.norm2 = nn.LayerNorm(query_token_dim)
        self.multi_head_attn = BroadMultiHeadAttention(qk_dim, num_heads)
        self.q, self.k, self.v = nn.Linear(query_token_dim, qk_dim, bias=True), nn.Linear(tgt_token_dim, qk_dim, bias=True), nn.Linear(tgt_token_dim, v_dim, bias=True)

        self.proj = nn.Linear(v_dim, query_token_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        #self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.ffn = nn.Sequential(
            nn.Linear(query_token_dim, query_token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(query_token_dim, query_token_dim),
            nn.Dropout(dropout)
        )

    def forward(self, query, tgt_token):
        """
            input:
            query --> latent tokens: [1,N,D]
            tgt_token --> image features: [B,M,D]
            output: x: [B,N,D]
        """

        short_cut = query
        query = self.norm1(query)
        q, k, v = self.q(query), self.k(tgt_token), self.v(tgt_token)
        x = self.multi_head_attn(q, k, v)
        x = short_cut + self.proj_drop(self.proj(x))

        #x = x + self.drop_path(self.ffn(self.norm2(x)))
        x = x + self.ffn(self.norm2(x))

        return x


