"""
FTCN baseline for video forgery detection; use BaselineModelWrapper with BTCHW video input.

Exploring Temporal Coherence for More General Video Face Forgery Detection @ ICCV'2021.
Copyright (c) Xiamen University and its affiliates.
Modified by Yinglin Zheng from https://github.com/yinglinzheng/FTCN
"""

import random

import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat

# Import OpenAI-CLIP helper package used by this baseline.
from .clip import clip


# ============================================================================
# Transformer Components (integrated from time_transformer.py, now part of FTCN module)
# ============================================================================

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        return x

class TimeTransformer(nn.Module):
    """Time transformer module for temporal modeling in FTCN."""
    def __init__(self, num_patches, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.num_patches = num_patches
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        b, n, _ = x.shape  # batch, num_patches, channels

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x, mask=None)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


# ============================================================================
# FTCN Model Components
# ============================================================================

class RandomPatchPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # batch,channel,16,7x7
        b, c, t, h, w = x.shape
        x = x.reshape(b, c, t, h * w)
        # Random selection during training, deterministic (center) during evaluation
        if self.training:
            while True:
                idx = random.randint(0, h * w - 1)
                i = idx // h
                j = idx % h
                if j == 0 or i == h - 1 or j == h - 1:
                    continue
                else:
                    break
        else:
            idx = h * w // 2
        x = x[..., idx]
        return x


def valid_idx(idx, h):
    i = idx // h
    j = idx % h
    if j == 0 or i == h - 1 or j == h - 1:
        return False
    else:
        return True


class RandomAvgPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # batch,channel,16,7x7
        b, c, t, h, w = x.shape
        x = x.reshape(b, c, t, h * w)
        candidates = list(range(h * w))
        candidates = [idx for idx in candidates if valid_idx(idx, h)]
        max_k = len(candidates)
        # Random selection during training (use 8 candidates), all candidates during evaluation
        if self.training:
            k = min(8, max_k)  # Use 8 candidates during training (default from original code)
        else:
            k = max_k
        candidates = random.sample(candidates, k)
        x = x[..., candidates].mean(-1)
        return x

class TransformerHead(nn.Module):
    def __init__(self, spatial_size=7, time_size=8, in_channels=2048):
        super().__init__()
        # if my_cfg.model.inco.no_time_pool:
        #     time_size = time_size * 2
        patch_type = 'time'
        if patch_type == "time":
            self.pool = nn.AvgPool3d((1, spatial_size, spatial_size))
            self.num_patches = time_size
        elif patch_type == "spatial":
            self.pool = nn.AvgPool3d((time_size, 1, 1))
            self.num_patches = spatial_size ** 2
        elif patch_type == "random":
            self.pool = RandomPatchPool()
            self.num_patches = time_size
        elif patch_type == "random_avg":
            self.pool = RandomAvgPool()
            self.num_patches = time_size
        elif patch_type == "all":
            self.pool = nn.Identity()
            self.num_patches = time_size * spatial_size * spatial_size
        else:
            raise NotImplementedError(patch_type)

        self.dim = -1
        if self.dim == -1:
            self.dim = in_channels

        self.in_channels = in_channels

        if self.dim != self.in_channels:
            self.fc = nn.Linear(self.in_channels, self.dim)

        default_params = dict(
            dim=self.dim, depth=6, heads=16, mlp_dim=2048, dropout=0.1, emb_dropout=0.1,
        )
        
        self.time_T = TimeTransformer(
            num_patches=self.num_patches, num_classes=1, **default_params
        )


    def forward(self, x):
        x = self.pool(x)
        x = x.reshape(-1, self.in_channels, self.num_patches)
        x = x.permute(0, 2, 1)
        if self.dim != self.in_channels:
            x = self.fc(x.reshape(-1, self.in_channels))
            x = x.reshape(-1, self.num_patches, self.dim)
        x = self.time_T(x)

        return x


class ViT_B_FTCN(nn.Module):
    def __init__(
        self, channel_size=512, class_num=1
    ):
        super(ViT_B_FTCN, self).__init__()
        self.clip_model, preprocess = clip.load('ViT-B-16')
        self.clip_model = self.clip_model.float()

        self.head = TransformerHead(spatial_size=14, time_size=8, in_channels=512)

    def forward(self, x):
        b, t, _, h, w = x.shape
        images = x.view(b * t, 3, h, w)
        sequence_output = self.clip_model.encode_image(images)
        _, _, c = sequence_output.shape
        sequence_output = sequence_output.view(b, t, 14, 14, c)
        sequence_output = sequence_output.permute(0, 4, 1, 2, 3)

        res = self.head(sequence_output)

        return res
if __name__ == '__main__':
    model = ViT_B_FTCN()
    model = model.cuda()
    dummy_input = torch.randn(4,8,3,224,224)
    dummy_input = dummy_input.cuda()
    model(dummy_input)
