import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
import einops
from einops import rearrange
from utils.mynet.paper2.vision_lstm2 import  SequenceTraversal,LinearHeadwiseExpand,CausalConv1d,LayerNorm,MultiHeadLayerNorm,MatrixLSTMCell,small_init_,wang_init_,bias_linspace_init_
from utils.mynet.paper2.vision_lstm_util import interpolate_sincos, to_ntuple, VitPatchEmbed, VitPosEmbed2d, DropPath, SequenceConv2d
from torch.cuda.amp import autocast
from thop import profile, clever_format
import time

class SSAMLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        num_bands: int = 4,
        grid_range: list = [-1, 1],
        enable_scale: bool = True,
    ):
        super().__init__()
      
        assert grid_range[0] < grid_range[1] 
        assert spline_order >= 1
        self.in_features = in_features  
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.num_bands = num_bands
        self.enable_scale = enable_scale

      
        self.spline_num = grid_size + spline_order
        self.spline_weights = nn.ParameterList([
            nn.Parameter(torch.Tensor(self.out_features, self.in_features, self.spline_num))
            for _ in range(num_bands)
        ])
        
        if enable_scale:
            self.scalers = nn.ParameterList([
                nn.Parameter(torch.Tensor(self.out_features, self.in_features))
                for _ in range(num_bands)
            ])
        
        
        h = (grid_range[1] - grid_range[0]) / grid_size
        self.register_buffer("grid", 
            torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]
        )
        self.reset_parameters()

       
        self.feature_transform = nn.Sequential(
            nn.Conv2d(self.in_features, self.in_features, kernel_size=3, padding=1, groups=self.in_features, bias=False),
            nn.BatchNorm2d(self.in_features),
            nn.ReLU(),
            nn.Conv2d(self.in_features, self.in_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.in_features)
        )

        
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.in_features, max(4, self.in_features // 8), kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(max(4, self.in_features // 8), self.in_features, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

    def reset_parameters(self):
        for w in self.spline_weights:
            nn.init.normal_(w, std=1/(math.sqrt(self.in_features) * math.sqrt(self.spline_num)))
        if self.enable_scale:
            for s in self.scalers:
                nn.init.uniform_(s, 0.9, 1.1)

    def _spectral_decompose(self, x):
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)  
        x_fft = torch.fft.rfft(x_flat, dim=-1)
        band_size = x_fft.size(-1) // self.num_bands
        decomposed = []
        for i in range(self.num_bands):
            start = i * band_size
            end = None if i == self.num_bands-1 else (i+1)*band_size
         
            band_fft = torch.zeros_like(x_fft)
            band_fft[..., start:end] = x_fft[..., start:end]
            band = torch.fft.irfft(band_fft, n=C)
            decomposed.append(band.view(B, H, W, C))  
        return torch.stack(decomposed, dim=1)  
    
    def _compute_splines(self, x):
        """
        核心函数：利用 Cox-de Boor 递归算法实现高阶多项式拟合
        """
        B, Band, H, W, C = x.shape
        x_flat = x.view(B, Band, H * W, C) 
        outputs = []
        
        for b in range(self.num_bands):
            band_x = x_flat[:, b, :, :] # (B, L, C)
            grid = self.grid
            x_un = band_x.unsqueeze(-1) # (B, L, C, 1)
            
  
            bases = (x_un >= grid[:-1]) & (x_un < grid[1:])
            bases = bases.to(band_x.dtype)
            
          
            for k in range(1, self.spline_order + 1):
                bases = (
                    (x_un - grid[:-(k + 1)]) / 
                    (grid[k:-1] - grid[:-(k + 1)]).clamp_min(1e-6) * bases[..., :-1]
                ) + (
                    (grid[k + 1:] - x_un) / 
                    (grid[k + 1:] - grid[1:-k]).clamp_min(1e-6) * bases[..., 1:]
                )
            
       
            weight = self.spline_weights[b] 
            if self.enable_scale:
                weight = weight * self.scalers[b].unsqueeze(-1)
            
        
            output = torch.einsum('blcg,ocg->blo', bases, weight) 
            outputs.append(output)
            
        return torch.stack(outputs, dim=1)  

    def _attention(self, x):
        B, Band, H, W, C = x.shape

        x_reshaped = x.permute(0, 1, 4, 2, 3).reshape(B * Band, C, H, W) 
        x_reshaped = self.feature_transform(x_reshaped)
        
        channel_weights = self.channel_attention(x_reshaped)
        x_reshaped = x_reshaped * channel_weights
        
        avg_out = torch.mean(x_reshaped, dim=1, keepdim=True)
        max_out = torch.max(x_reshaped, dim=1, keepdim=True)[0]
        spatial_features = torch.cat([avg_out, max_out], dim=1)
        spatial_weights = self.spatial_attention(spatial_features)
        
        x_reshaped = x_reshaped * spatial_weights
        return x_reshaped.view(B, Band, C, H, W).permute(0, 1, 3, 4, 2)

    def forward(self, x):
        B, C, H, W = x.shape
        x_decomposed = self._spectral_decompose(x) 
        x_att = self._attention(x_decomposed) 
        band_outputs = self._compute_splines(x_att) 
        fused = torch.mean(band_outputs, dim=1) 
        fused = fused.view(B, H, W, self.out_features).permute(0, 3, 1, 2) 
        return fused 

class SSAMBlock(nn.Module):
    def __init__(self, dim, num_bands=4, grid_size=5, spline_order=3, residual=True):
        super().__init__()
        self.kan = SSAMLayer(in_features=dim, out_features=dim, num_bands=num_bands, grid_size=grid_size, spline_order=spline_order)
        self.norm = nn.BatchNorm2d(dim)
        self.act = nn.GELU()
        self.residual = residual

    def forward(self, x):
        res = x
        x = self.kan(x)
        x = self.norm(x)
        x = self.act(x)
        return x + res if self.residual else x

# ---------------------------
# D2Encoder
# ---------------------------
def channel_shuffle(x, groups):
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    return x.view(batch_size, -1, height, width)

class MultiKernelDepthwiseConv(nn.Module):
    def __init__(self, channels, kernel_sizes=[3,5,7,9], groups=4):
        super().__init__()
        self.groups = groups
        self.channels = channels
        self.channels_per_group = self.channels // self.groups
        
        self.convs = nn.ModuleList()
        for i, k in enumerate(kernel_sizes):
            conv = nn.Sequential(
                nn.Conv2d(self.channels_per_group, self.channels_per_group, 
                          kernel_size=k, padding=k//2, groups=self.channels_per_group, bias=False),
                nn.BatchNorm2d(self.channels_per_group)
            )
            self.convs.append(conv)
        
        while len(self.convs) < self.groups:
            self.convs.append(self.convs[-1])

    def forward(self, x):
        x_split = torch.chunk(x, self.groups, dim=1)
        x_outs = []
        
        for i in range(self.groups):
            x_group = x_split[i]
            x_outs.append(self.convs[i](x_group))
        
        x = torch.cat(x_outs, dim=1)
        return channel_shuffle(x, self.groups)

class GMKC(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_ratio=1.5, kernel_sizes=[3,5,7,9]):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.hidden_channels = int(self.in_channels * expansion_ratio)
        self.hidden_channels = max(4, (self.hidden_channels + 3) // 4 * 4)
        
        self.pw_expand = nn.Sequential(
            nn.Conv2d(self.in_channels, self.hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU6(inplace=True)
        )
        
        self.GMKC = MultiKernelDepthwiseConv(self.hidden_channels, kernel_sizes)
        
        self.pw_reduce = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_channels)
        )
        
        self.use_residual = (self.in_channels == self.out_channels)

    def forward(self, x):
        residual = x
        x = self.pw_expand(x)
        x = self.GMKC(x)
        x = self.pw_reduce(x)
        
        if self.use_residual:
            x = x + residual
        
        return x

class DWT_Haar(nn.Module):
    def __init__(self):
        super().__init__()
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
        lh = torch.tensor([[0.5, 0.5], [-0.5, -0.5]], dtype=torch.float32)
        hl = torch.tensor([[0.5, -0.5], [0.5, -0.5]], dtype=torch.float32)
        hh = torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32) * 0.5
        self.register_buffer('ll', ll.view(1, 1, 2, 2))
        self.register_buffer('lh', lh.view(1, 1, 2, 2))
        self.register_buffer('hl', hl.view(1, 1, 2, 2))
        self.register_buffer('hh', hh.view(1, 1, 2, 2))

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B*C, 1, H, W)
        ll = F.conv2d(x, self.ll, stride=2)
        lh = F.conv2d(x, self.lh, stride=2)
        hl = F.conv2d(x, self.hl, stride=2)
        hh = F.conv2d(x, self.hh, stride=2)
        return torch.cat([ll, lh, hl, hh], dim=1).view(B, 4*C, H//2, W//2)

class IWT_Haar(nn.Module):
    def forward(self, x):
        B, C, H, W = x.shape
        assert C % 4 == 0
        C_recon = C // 4
        x = x.view(B, C_recon, 4, H, W)
        x = x.permute(0, 1, 3, 4, 2)
        x = x.contiguous().view(B, C_recon, H, W, 2, 2)
        x = x.permute(0, 1, 4, 2, 5, 3)
        x = x.contiguous().view(B, C_recon, 2*H, 2*W)
        x = x.repeat(1, 4, 1, 1)
        return x

class FrequencyPath(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        
        self.dwt = DWT_Haar()
        self.iwt = IWT_Haar()
        self.ll_conv = nn.Conv2d(self.in_ch, self.in_ch, kernel_size=1, bias=False)
        self.hf_conv = nn.Sequential(
            nn.Conv2d(3*self.in_ch, self.out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.out_ch),
            nn.ReLU()
        )
        self.final_conv = nn.Conv2d(self.in_ch + self.out_ch, self.out_ch, kernel_size=1, bias=False)

    def forward(self, x):
        coeffs = self.dwt(x)
        B, C4, H, W = coeffs.shape
        ll_raw, lh_raw, hl_raw, hh_raw = torch.split(coeffs, C4//4, dim=1)
        
        ll_proc = self.ll_conv(ll_raw)
        hf_cat = torch.cat([lh_raw, hl_raw, hh_raw], dim=1)
        hf_proc = self.hf_conv(hf_cat)
        
        merged = self.final_conv(torch.cat([ll_proc, hf_proc], dim=1))
        return self.iwt(merged)

class D2Encoder(nn.Module):
    def __init__(self, in_ch, out_ch, expansion_ratio=1.5, kernel_sizes=[3,5,7,9]):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        
        self.spatial = GMKC(self.in_ch, self.out_ch, expansion_ratio, kernel_sizes)
        self.freq = FrequencyPath(self.in_ch, self.out_ch)
        
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.out_ch, self.out_ch, bias=False),
            nn.Sigmoid()
        )
        
        self.down = nn.Conv2d(self.out_ch, self.out_ch, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x):
        spatial = self.spatial(x)
        freq = self.freq(x)
        
        gate_spatial = self.gate(spatial).view(-1, self.out_ch, 1, 1)
        gate_freq = self.gate(freq).view(-1, self.out_ch, 1, 1)
        
        fused = gate_spatial * spatial + gate_freq * freq
        out = self.down(fused)
        
        return out

# ---------------------------
# ViL BLOCK
# ---------------------------
class ViLLayer(nn.Module):
    def __init__(
            self,
            dim,
            direction,
            expansion=1.5,
            qkv_block_size=4,
            proj_bias=False,
            norm_bias=False,
            conv_bias=False,
            conv_kernel_size=3,
            conv_kind="2d",
            init_weights="original",
            seqlens=None,
            num_blocks=None,
    ):
        super().__init__()
        self.dim = dim
        self.direction = direction
        self.expansion = expansion
        self.qkv_block_size = qkv_block_size
        self.proj_bias = proj_bias
        self.conv_bias = conv_bias
        self.conv_kind = conv_kind
        self.init_weights = init_weights
        self.num_blocks = num_blocks

        self.inner_dim = int(expansion * self.dim)
        self.inner_dim = max(4, (self.inner_dim + 3) // 4 * 4)
        self.num_heads = self.inner_dim // qkv_block_size
        
        self.proj_up = nn.Linear(self.dim, 2 * self.inner_dim, bias=proj_bias)
        self.q_proj = LinearHeadwiseExpand(dim=self.inner_dim, num_heads=self.num_heads, bias=proj_bias)
        self.k_proj = LinearHeadwiseExpand(dim=self.inner_dim, num_heads=self.num_heads, bias=proj_bias)
        self.v_proj = LinearHeadwiseExpand(dim=self.inner_dim, num_heads=self.num_heads, bias=proj_bias)

        if conv_kind == "causal1d":
            self.conv = CausalConv1d(dim=self.inner_dim, kernel_size=conv_kernel_size, bias=conv_bias)
        elif conv_kind == "2d":
            self.conv = SequenceConv2d(
                in_channels=self.inner_dim,
                out_channels=self.inner_dim,
                kernel_size=conv_kernel_size,
                padding=conv_kernel_size // 2,
                groups=self.inner_dim,
                bias=conv_bias,
                seqlens=seqlens,
            )
        else:
            raise NotImplementedError
            
        self.mlstm_cell = MatrixLSTMCell(
            dim=self.inner_dim,
            num_heads=qkv_block_size,
            norm_bias=norm_bias,
        )
        self.learnable_skip = nn.Parameter(torch.ones(self.inner_dim))
        self.proj_down = nn.Linear(self.inner_dim, self.dim, bias=proj_bias)
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, C = x.shape
        x_inner = self.proj_up(x)
        x_mlstm, z = torch.chunk(x_inner, chunks=2, dim=-1)

        x_mlstm_conv = self.conv(x_mlstm)
        x_mlstm_conv_act = F.silu(x_mlstm_conv)
        q = self.q_proj(x_mlstm_conv_act)
        k = self.k_proj(x_mlstm_conv_act)
        v = self.v_proj(x_mlstm)
        
        h_tilde_state = self.mlstm_cell(q=q, k=k, v=v)
        h_tilde_state_skip = h_tilde_state + (self.learnable_skip * x_mlstm_conv_act)
        h_state = h_tilde_state_skip * F.silu(z)
        x = self.proj_down(h_state)
        return x

    def reset_parameters(self):
        small_init_(self.proj_up.weight, dim=self.dim)
        if self.proj_up.bias is not None:
            nn.init.zeros_(self.proj_up.bias)
        wang_init_(self.proj_down.weight, dim=self.dim, num_blocks=1)
        nn.init.ones_(self.learnable_skip)

class ViLBlock(nn.Module):
    def __init__(
            self,
            dim,
            direction,
            drop_path=0.0,
            conv_kind="2d",
            conv_kernel_size=3,
            proj_bias=False,
            norm_bias=False,
            seqlens=None,
            num_blocks=None,
            init_weights="original",
    ):
        super().__init__()
        self.dim = dim
        self.drop_path = DropPath(drop_prob=drop_path)
        self.norm = LayerNorm(ndim=self.dim, weight=True, bias=norm_bias)
        self.layer = ViLLayer(
            dim=self.dim,
            direction=direction,
            conv_kind=conv_kind,
            conv_kernel_size=conv_kernel_size,
            seqlens=seqlens,
            norm_bias=norm_bias,
            proj_bias=proj_bias,
            num_blocks=num_blocks,
            init_weights=init_weights,
        )

    def _forward_path(self, x):
        x = self.norm(x)
        x = self.layer(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop_path(x, self._forward_path)
        return x

class ViLBlockPair(nn.Module):
    def __init__(self, dim, **kwargs):
        super().__init__()
        self.rowwise_from_top_left = ViLBlock(dim=dim, direction=SequenceTraversal.ROWWISE_FROM_TOP_LEFT, **kwargs)
        self.rowwise_from_bot_right = ViLBlock(dim=dim, direction=SequenceTraversal.ROWWISE_FROM_BOT_RIGHT, **kwargs)

    def forward(self, x):
        x = self.rowwise_from_top_left(x)
        x = self.rowwise_from_bot_right(x)
        return x

# ---------------------------
# DSFDecoder
# ---------------------------
class DynamicSparseConv(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.reduction = 8
        self.condition = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.in_channels, max(4, self.in_channels//self.reduction), kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(max(4, self.in_channels//self.reduction), 9 * self.in_channels, kernel_size=1, bias=False)
        )
        self.conv = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, padding=1, groups=self.in_channels, bias=False)

    def forward(self, x):
        weights = self.condition(x).view(x.size(0), self.in_channels, 9)
        weights = F.softmax(weights, dim=2).view(x.size(0)*x.size(1), 1, 3, 3)
        out = F.conv2d(x.reshape(1, -1, x.size(2), x.size(3)), 
                       weights, padding=1, groups=x.size(0)*x.size(1)).view_as(x)
        return out + x

class DSFDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dynamic_block = nn.Sequential(
            DynamicSparseConv(self.in_channels),
            nn.ReLU(),
            DynamicSparseConv(self.in_channels)
        )
        self.channel_align = nn.Conv2d(2*self.in_channels, self.in_channels, kernel_size=1, bias=False)
        self.upsampler = nn.Sequential(
            nn.Conv2d(self.in_channels, 4*self.in_channels, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, bias=False)
        )
        self.res_enhance = nn.Sequential(
            nn.Conv2d(self.out_channels, max(4, self.out_channels//8), kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(max(4, self.out_channels//8), self.out_channels, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x1, x2):
        x1 = self.dynamic_block(x1)
        x2 = self.dynamic_block(x2)
        fused = self.channel_align(torch.cat([x1, x2], dim=1))
        up = self.upsampler(fused)
        return up + self.res_enhance(up)

class final_outblock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels, 1, kernel_size=4, stride=2, padding=1, bias=False)
    def forward(self, x):
        return self.up_conv(x)

# ---------------------------
# CSB & CASA
# ---------------------------
class CSB(nn.Module):
    def __init__(self, high_ch, low_ch):
        super().__init__()
        self.high_ch_conv = nn.Sequential(nn.Conv2d(high_ch, high_ch, 1, bias=False), nn.BatchNorm2d(high_ch), nn.ReLU())
        self.low_ch_conv = nn.Sequential(nn.Conv2d(low_ch, high_ch, 3, padding=1, bias=False), nn.BatchNorm2d(high_ch), nn.ReLU())
        self.context_extractor = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(high_ch, high_ch, 1, bias=False), nn.Sigmoid())
    def forward(self, high_ch_feat, low_ch_feat): 
        h = self.high_ch_conv(high_ch_feat)
        l = F.adaptive_max_pool2d(self.low_ch_conv(low_ch_feat), output_size=high_ch_feat.shape[-2:])
        return h * self.context_extractor(h) + l

class CASA(nn.Module):
    def __init__(self, channels, groups=16):
        super().__init__()
        self.groups = min(groups, channels)
        self.channels_per_group = channels // self.groups
        self.fc = nn.Sequential(nn.Linear(1, 4, bias=False), nn.ReLU(True), nn.Linear(4, 1, bias=False), nn.Sigmoid())
    def forward(self, x):
        b, c, h, w = x.size()
        x_g = x.view(b, self.groups, self.channels_per_group, h, w)
        s = x_g.mean(dim=[2, 3, 4], keepdim=True)
        w_g = self.fc(s.view(-1, 1)).view(b, self.groups, 1, 1, 1)
        return (x_g * w_g).view(b, c, h, w)

# ---------------------------
# 主模型 
# ---------------------------
class FVBLNet(nn.Module):
    def __init__(self, in_channels=3, classes=1):
        super(FVBLNet, self).__init__()
        
        
        self.dim_16 = 16
        self.dim_32 = 32
        self.dim_48 = 48
        self.dim_64 = 64
        self.dim_128 = 128
        self.dim_256 = 256
        self.dim_512 = 512

        # Stem
        self.conv_stem = nn.Sequential(
            nn.Conv2d(3, self.dim_16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.dim_16),
            nn.ReLU(),
            nn.Conv2d(self.dim_16, self.dim_16, 3, 1, 1, groups=self.dim_16, bias=False),
            nn.BatchNorm2d(self.dim_16),
            nn.ReLU(),
            nn.Conv2d(self.dim_16, self.dim_32, 1, 1, bias=False),
            nn.BatchNorm2d(self.dim_32)
        )

        # Encoders & Decoders
        self.encoder1 = D2Encoder(self.dim_32, self.dim_48)
        self.encoder2 = D2Encoder(self.dim_48, self.dim_64)
        self.encoder3 = D2Encoder(self.dim_64, self.dim_128)
        self.encoder4 = D2Encoder(self.dim_128, self.dim_256)
        
        self.decoder1 = DSFDecoder(self.dim_256, self.dim_128)
        self.decoder2 = DSFDecoder(self.dim_128, self.dim_64)
        self.decoder3 = DSFDecoder(self.dim_64, self.dim_48)
        self.decoder4 = DSFDecoder(self.dim_48, self.dim_32)

        # ViL Blocks
        self.vile3 = ViLBlockPair(self.dim_128)
        self.vile4 = ViLBlockPair(self.dim_256)
        self.vilb  = ViLBlockPair(self.dim_256)
        self.vild1 = ViLBlockPair(self.dim_128)
        self.vild2 = ViLBlockPair(self.dim_64)

        self.block2 = SSAMBlock(dim=self.dim_256)
        self.sg = CASA(self.dim_256)
        self.changeconv = nn.Sequential(
            nn.Conv2d(self.dim_512, self.dim_256, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.dim_256), nn.ReLU(),
            nn.Conv2d(self.dim_256, self.dim_256, 1, bias=False)
        )

        self.cs1 = CSB(self.dim_64, self.dim_48)
        self.cs2 = CSB(self.dim_128, self.dim_64)
        self.cs3 = CSB(self.dim_256, self.dim_128)
        
        self.outblock = final_outblock(self.dim_32)

    def forward(self, x):
        x = self.conv_stem(x)
        
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        skip_2 = self.cs1(x2, x1)
        
        x3 = self.encoder3(x2)
        B, C, H, W = x3.shape
        x3 = self.vile3(x3.reshape(B, C, H*W).transpose(-1, -2)).transpose(-1, -2).reshape(B, C, H, W)
        skip_3 = self.cs2(x3, x2)
        
        x4 = self.encoder4(x3)
        B, C, H, W = x4.shape
        x4 = self.vile4(x4.reshape(B, C, H*W).transpose(-1, -2)).transpose(-1, -2).reshape(B, C, H, W)
        skip_4 = self.cs3(x4, x3)
        
        # Bottleneck
        B, C, H, W = x4.shape
        xb = self.vilb(x4.reshape(B, C, H*W).transpose(-1, -2)).transpose(-1, -2).reshape(B, C, H, W)
        xc = self.block2(xb)
        xb = self.sg(xb)
        xb = self.changeconv(torch.cat([xc, xb], dim=1))
        xb = self.vilb(xb.reshape(B, C, H*W).transpose(-1, -2)).transpose(-1, -2).reshape(B, C, H, W)

        # Decoding
        d1 = self.vild1(self.decoder1(xb, skip_4).reshape(B, self.dim_128, -1).transpose(-1, -2)).transpose(-1, -2).reshape(B, self.dim_128, 14, 14)
        d2 = self.vild2(self.decoder2(d1 + skip_3, skip_3).reshape(B, self.dim_64, -1).transpose(-1, -2)).transpose(-1, -2).reshape(B, self.dim_64, 28, 28)
        d3 = self.decoder3(d2 + skip_2, skip_2)
        d4 = self.decoder4(d3, x1 + d3)
        
        return self.outblock(d4)
