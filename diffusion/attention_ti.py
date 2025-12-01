# copy from https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/attention.py
import math
from inspect import isfunction
import logging
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn, einsum
from typing import Any, Optional
from .tacm_nn import checkpoint, linear, timestep_embedding

logpy = logging.getLogger(__name__)

def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = (
            nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
            if not glu
            else GEGLU(dim, inner_dim)
        )

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class AlphaBlender(nn.Module):
    strategies = ["learned", "fixed", "learned_with_images"]

    def __init__(
        self,
        alpha: float,
        merge_strategy: str = "learned_with_images",
        rearrange_pattern: str = "b t -> (b t) 1 1",
    ):
        super().__init__()
        self.merge_strategy = merge_strategy
        self.rearrange_pattern = rearrange_pattern

        assert (
            merge_strategy in self.strategies
        ), f"merge_strategy needs to be in {self.strategies}"

        if self.merge_strategy == "fixed":
            self.register_buffer("mix_factor", torch.Tensor([alpha]))
        elif (
            self.merge_strategy == "learned"
            or self.merge_strategy == "learned_with_images"
        ):
            self.register_parameter(
                "mix_factor", torch.nn.Parameter(torch.Tensor([alpha]))
            )
        else:
            raise ValueError(f"unknown merge strategy {self.merge_strategy}")

    def get_alpha(self, image_only_indicator: torch.Tensor) -> torch.Tensor:
        if self.merge_strategy == "fixed":
            alpha = self.mix_factor
        elif self.merge_strategy == "learned":
            alpha = torch.sigmoid(self.mix_factor)
        elif self.merge_strategy == "learned_with_images":
            assert image_only_indicator is not None, "need image_only_indicator ..."
            alpha = torch.where(
                image_only_indicator.bool(),
                torch.ones(1, 1, device=image_only_indicator.device),
                rearrange(torch.sigmoid(self.mix_factor), "... -> ... 1"),
            )
            alpha = rearrange(alpha, self.rearrange_pattern)
        else:
            raise NotImplementedError
        return alpha

    def forward(
        self,
        x_spatial: torch.Tensor,
        x_temporal: torch.Tensor,
        image_only_indicator: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        alpha = self.get_alpha(image_only_indicator)
        x = (
            alpha.to(x_spatial.dtype) * x_spatial
            + (1.0 - alpha).to(x_spatial.dtype) * x_temporal
        )
        return x

#https://github.com/srpkdyy/VideoLDM/blob/master/blocks.py#L193
class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_pos=512):
        super().__init__()

        pos = torch.arange(max_pos)

        freq = torch.arange(dim//2) / dim
        freq = (freq * torch.tensor(10000).log()).exp()

        x = rearrange(pos, 'L -> L 1') / freq
        x = rearrange(x, 'L d -> L d 1')

        pe = torch.cat((x.sin(), x.cos()), dim=-1)
        self.pe = rearrange(pe, 'L d sc -> L (d sc)')

        self.dummy = nn.Parameter(torch.rand(1))

    def forward(self, length):
        enc = self.pe[:length]
        enc = enc.to(self.dummy.device)
        return enc

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0, backend=None):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

        self.backend = backend

    def forward(self, x, context=None, mask=None, additional_tokens=None, n_times_crossframe_attn_in_self=0,):
        h = self.heads

        if additional_tokens is not None:
            # get the number of masked tokens at the beginning of the output sequence
            n_tokens_to_mask = additional_tokens.shape[1]
            # add additional token
            x = torch.cat([additional_tokens, x], dim=1)

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        if n_times_crossframe_attn_in_self:
            # reprogramming cross-frame attention as in https://arxiv.org/abs/2303.13439
            assert x.shape[0] % n_times_crossframe_attn_in_self == 0
            n_cp = x.shape[0] // n_times_crossframe_attn_in_self
            k = repeat(
                k[::n_times_crossframe_attn_in_self], "b ... -> (b n) ...", n=n_times_crossframe_attn_in_self
            )
            v = repeat(
                v[::n_times_crossframe_attn_in_self], "b ... -> (b n) ...", n=n_times_crossframe_attn_in_self
            )


        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))



        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        disable_self_attn=False,
        attn_mode="softmax",
        sdp_backend=None,):
        super().__init__()
        self.disable_self_attn = disable_self_attn
        self.attn1 = CrossAttention(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=context_dim if self.disable_self_attn else None,
            backend=sdp_backend,
        )  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            backend=sdp_backend,)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None, additional_tokens=None, n_times_crossframe_attn_in_self=0):
        x = self.attn1(
            self.norm1(x),
            context=context if self.disable_self_attn else None,
            additional_tokens=additional_tokens,
            n_times_crossframe_attn_in_self=n_times_crossframe_attn_in_self if not self.disable_self_attn else 0,
        ) + x
        x_t_output = self.attn2(
            self.norm2(x),
            context=context[:,:-1],
            additional_tokens=additional_tokens) + x
        x_i_output = self.attn2(
            self.norm2(x),
            context=context[:,-1].unsqueeze(1),
            additional_tokens=additional_tokens) + x
        x_output = x_t_output+x_i_output
        x = self.ff(self.norm3(x_output)) + x_output
        return x

class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False, use_linear=False,
                 attn_type="softmax", use_checkpoint=True,
                 sdp_backend=None):
        super().__init__()

        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        if exists(context_dim) and isinstance(context_dim, list):
            if depth != len(context_dim):
                logpy.warn(
                    f"{self.__class__.__name__}: Found context dims "
                    f"{context_dim} of depth {len(context_dim)}, which does not "
                    f"match the specified 'depth' of {depth}. Setting context_dim "
                    f"to {depth * [context_dim[0]]} now."
                )
                # depth does not match context dims.
                assert all(
                    map(lambda x: x == context_dim[0], context_dim)
                ), "need homogenous context_dim to match depth automatically"
                context_dim = depth * [context_dim[0]]
        elif context_dim is None:
            context_dim = [None] * depth

        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        if not use_linear:
            self.proj_in = nn.Conv2d(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0
            )
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(
                inner_dim,
                n_heads,
                d_head,
                dropout=dropout,
                context_dim=context_dim[d],
                disable_self_attn=disable_self_attn,
                attn_mode=attn_type,
                checkpoint=use_checkpoint,
                sdp_backend=sdp_backend,
            )
                for d in range(depth)]
        )

        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]

        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            if i > 0 and len(context) == 1:
                i = 0  # use same context for each block
            context[i] = repeat(context[i], "b m d -> (b t) m d", t=16)
            x = block(x, context=context[i], additional_tokens=None, n_times_crossframe_attn_in_self=16)
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in

class TimeMixSequential(nn.Sequential):
    def forward(self, x, context=None, timesteps=None):
        for layer in self:
            x = layer(x, context, timesteps)

        return x


