import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
import einops
from einops import rearrange
from vision_lstm2 import  SequenceTraversal,LinearHeadwiseExpand,CausalConv1d,LayerNorm,MultiHeadLayerNorm,MatrixLSTMCell,small_init_,wang_init_,bias_linspace_init_
from vision_lstm_util import interpolate_sincos, to_ntuple, VitPatchEmbed, VitPosEmbed2d, DropPath, SequenceConv2d
from torch.cuda.amp import autocast



# ---------------------------
# SSAM Block
# ---------------------------

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

        self.spline_weights = nn.ParameterList([
            nn.Parameter(torch.Tensor(out_features, in_features))
            for _ in range(num_bands)
        ])
        if enable_scale:
            self.scalers = nn.ParameterList([
                nn.Parameter(torch.Tensor(out_features, in_features))
                for _ in range(num_bands)
            ])
        
        
        h = (grid_range[1] - grid_range[0]) / grid_size
        self.register_buffer("grid", 
            torch.arange(-spline_order, grid_size+spline_order+1) * h + grid_range[0]
        )
        self.reset_parameters()

        
        self.feature_transform = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_features),
            nn.ReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=3, padding=1)
        )

        
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_features, in_features // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_features // 4, in_features, kernel_size=1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def reset_parameters(self):
       
        for w in self.spline_weights:
            nn.init.normal_(w, std=1/math.sqrt(self.in_features))
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
            band = torch.fft.irfft(x_fft[..., start:end], n=C)
            decomposed.append(band.view(B, H, W, C))  
        return torch.stack(decomposed, dim=1)  
    
    def _compute_splines(self, x):
        
        B, Band, H, W, C = x.shape
        x = x.view(B, Band, H * W, C) 
    
        outputs = []
        for b in range(self.num_bands):
            
            band_x = x[:, b, :, :] 
        
            
            bases = (band_x.unsqueeze(-1) >= self.grid[:-1]) & (band_x.unsqueeze(-1) < self.grid[1:])
            bases = bases.to(band_x.dtype)
            for k in range(1, self.spline_order + 1):
                bases = (
                    (band_x.unsqueeze(-1) - self.grid[:-(k + 1)]) / 
                    (self.grid[k:-1] - self.grid[:-(k + 1)]).clamp_min(1e-6) * bases[..., :-1]
                ) + (
                    (self.grid[k + 1:] - band_x.unsqueeze(-1)) / 
                    (self.grid[k + 1:] - self.grid[1:-k]).clamp_min(1e-6) * bases[..., 1:]
                )
        
            
            weight = self.spline_weights[b]
            if self.enable_scale:
                weight = weight * self.scalers[b]
            output = torch.einsum('bci,oi->bci', band_x, weight)  
            outputs.append(output)
    
        return torch.stack(outputs, dim=1)  

    def _attention(self, x):
       
        B, Band, H, W, C = x.shape
        x = x.view(B * Band, C, H, W) 

       
        x = self.feature_transform(x)

       
        channel_weights = self.channel_attention(x)
        x = x * channel_weights

        
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        spatial_features = torch.cat([avg_out, max_out], dim=1)
        spatial_weights = self.spatial_attention(spatial_features)
        x = x * spatial_weights

        x = x.view(B, Band, H, W, C)
        return x

    def forward(self, x):
       
        B, C, H, W = x.shape
       
        assert C == self.in_features, f"Input features {C} != {self.in_features}"
        
       
        x_decomposed = self._spectral_decompose(x)  

        B, Band, H, W, C = x_decomposed.shape
        x_decomposed = x_decomposed.view(B * Band, C, H, W)  
        x_decomposed = self.feature_transform(x_decomposed)
        x_decomposed = x_decomposed.view(B, Band, H, W, C)

        x_decomposed = self._attention(x_decomposed) 
        #band_outputs= self._attention(x_decomposed) 
        band_outputs = self._compute_splines(x_decomposed)  
       
        fused = torch.mean(band_outputs, dim=1)  
        fused = fused.view(B, H, W, self.out_features).permute(0, 3, 1, 2)  
        return fused  
   

class SSAMBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_bands: int = 4,
        grid_size: int = 5,
        spline_order: int = 3,
        residual: bool = True,
        norm: nn.Module = nn.BatchNorm2d,
        act: nn.Module = nn.GELU(),
    ):
        super().__init__()
        self.kan = SSAMLayer(
            in_features=dim,
            out_features=dim,
            num_bands=num_bands,
            grid_size=grid_size,
            spline_order=spline_order
        )
        self.norm = norm(dim) if norm is not None else nn.Identity()
        self.act = act
        self.residual = residual

    def forward(self, x):
       
        identity = x
        x = self.kan(x)
        x = self.norm(x)
        x = self.act(x)
        return x + identity if self.residual else x



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
        self.convs = nn.ModuleList()
        for k in kernel_sizes:
           
            conv = nn.Sequential(
                nn.Conv2d(channels//groups, channels//groups, 
                          kernel_size=k, padding=k//2, groups=channels//groups, bias=False),
                nn.BatchNorm2d(channels//groups)
            )
            self.convs.append(conv)

    def forward(self, x):
       
        x_split = torch.chunk(x, self.groups, dim=1)
        x_outs = []
        for i, conv in enumerate(self.convs):
           
            x_group = x_split[i]
            x_outs.append(conv(x_group))
        
        x = torch.cat(x_outs, dim=1)
        return channel_shuffle(x, self.groups)

class GMKC(nn.Module):
    
    def __init__(self, in_channels, out_channels, expansion_ratio=2, kernel_sizes=[3,5,7,9]):
        super().__init__()
        hidden_channels = int(in_channels * expansion_ratio)
        self.pw_expand = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU6(inplace=True)  
        )
        self.GMKC = MultiKernelDepthwiseConv(hidden_channels, kernel_sizes)
        self.pw_reduce = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.use_residual = (in_channels == out_channels)

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
        assert out_ch % 4 == 0
        self.dwt = DWT_Haar()
        self.iwt = IWT_Haar()
        self.ll_conv = nn.Conv2d(in_ch, in_ch, 1)
        self.hf_conv = nn.Sequential(
            nn.Conv2d(3*in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.final_conv = nn.Conv2d(in_ch + out_ch, out_ch, 1)

    def forward(self, x):
        coeffs = self.dwt(x)
        B, C4, H, W = coeffs.shape
        ll, lh, hl, hh = torch.split(coeffs, C4//4, dim=1)
        ll = self.ll_conv(ll)
        hf = self.hf_conv(torch.cat([lh, hl, hh], dim=1))
        merged = self.final_conv(torch.cat([ll, hf], dim=1))
        return self.iwt(merged)

class D2Encoder(nn.Module):
    def __init__(self, in_ch, out_ch, expansion_ratio=2, kernel_sizes=[3,5,7,9]):
        super().__init__()
        self.spatial = GMKC(in_ch, out_ch, expansion_ratio, kernel_sizes)
        self.freq = FrequencyPath(in_ch, out_ch)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2 * out_ch, out_ch),
            nn.Sigmoid()
        )
        self.down = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)
    
    def forward(self, x):
        spatial = self.spatial(x)
        freq = self.freq(x)
        gate_input = torch.cat([spatial, freq], dim=1)
        gate = self.gate(gate_input)
        gate = gate.view(gate.shape[0], gate.shape[1], 1, 1)
        gate = gate.expand_as(spatial)
        fused = gate * spatial + (1 - gate) * freq
        return self.down(fused)
        # return self.down(spatial)

# class D2Encoder(nn.Module):
#     def __init__(self, in_channels=1, out_channels=64):
#         super().__init__()
#         self.conv_block = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )

#     def forward(self, x):
#         return self.conv_block(x)







# ---------------------------
# BVIL BLOCK
# ---------------------------


class ViLLayer(nn.Module):
    def __init__(
            self,
            dim,
            direction,
            expansion=2,
            qkv_block_size=4,
            proj_bias=True,
            norm_bias=True,
            conv_bias=True,
            conv_kernel_size=4,
            conv_kind="2d",
            init_weights="original",
            seqlens=None,
            num_blocks=None,
    ):
        super().__init__()
        assert dim % qkv_block_size == 0
        self.dim = dim
        self.direction = direction
        self.expansion = expansion
        self.qkv_block_size = qkv_block_size
        self.proj_bias = proj_bias
        self.conv_bias = conv_bias
        self.conv_kernel_size = conv_kernel_size
        self.conv_kind = conv_kind
        self.init_weights = init_weights
        self.num_blocks = num_blocks

        inner_dim = expansion * dim
        num_heads = inner_dim // qkv_block_size
        self.proj_up = nn.Linear(
            in_features=dim,
            out_features=2 * inner_dim,
            bias=proj_bias,
        )
        self.q_proj = LinearHeadwiseExpand(
            dim=inner_dim,
            num_heads=num_heads,
            bias=proj_bias,
        )
        self.k_proj = LinearHeadwiseExpand(
            dim=inner_dim,
            num_heads=num_heads,
            bias=proj_bias,
        )
        self.v_proj = LinearHeadwiseExpand(
            dim=inner_dim,
            num_heads=num_heads,
            bias=proj_bias,
        )

        if conv_kind == "causal1d":
            self.conv = CausalConv1d(
                dim=inner_dim,
                kernel_size=conv_kernel_size,
                bias=conv_bias,
            )
        elif conv_kind == "2d":
            assert conv_kernel_size % 2 == 1, \
                f"same output shape as input shape is required -> even kernel sizes not supported"
            self.conv = SequenceConv2d(
                in_channels=inner_dim,
                out_channels=inner_dim,
                kernel_size=conv_kernel_size,
                padding=conv_kernel_size // 2,
                groups=inner_dim,
                bias=conv_bias,
                seqlens=seqlens,
            )
        else:
            raise NotImplementedError
        self.mlstm_cell = MatrixLSTMCell(
            dim=inner_dim,
            num_heads=qkv_block_size,
            norm_bias=norm_bias,
        )
        self.learnable_skip = nn.Parameter(torch.ones(inner_dim))

        self.proj_down = nn.Linear(
            in_features=inner_dim,
            out_features=dim,
            bias=proj_bias,
        )
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape


        x_inner = self.proj_up(x)
        x_mlstm, z = torch.chunk(x_inner, chunks=2, dim=-1)

        # mlstm branch
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
        
        if self.init_weights == "original":
            wang_init_(self.proj_down.weight, dim=self.dim, num_blocks=1)
        elif self.init_weights == "original-fixed":
            wang_init_(self.proj_down.weight, dim=self.dim, num_blocks=self.num_blocks)
        else:
            raise NotImplementedError
        if self.proj_down.bias is not None:
            nn.init.zeros_(self.proj_down.bias)

        nn.init.ones_(self.learnable_skip)

        def _init_qkv_proj(qkv_proj: LinearHeadwiseExpand):
           
            small_init_(qkv_proj.weight, dim=self.dim)
            if qkv_proj.bias is not None:
                nn.init.zeros_(qkv_proj.bias)

        _init_qkv_proj(self.q_proj)
        _init_qkv_proj(self.k_proj)
        _init_qkv_proj(self.v_proj)

        self.mlstm_cell.reset_parameters()


class ViLBlock(nn.Module):
    def __init__(
            self,
            dim,
            direction,
            drop_path=0.0,
            conv_kind="2d",
            conv_kernel_size=3,
            proj_bias=True,
            norm_bias=True,
            seqlens=None,
            num_blocks=None,
            init_weights="original",
    ):
        super().__init__()
        self.dim = dim
        self.direction = direction
        self.drop_path = drop_path
        self.conv_kind = conv_kind
        self.conv_kernel_size = conv_kernel_size
        self.init_weights = init_weights

        self.drop_path = DropPath(drop_prob=drop_path)
        self.norm = LayerNorm(ndim=dim, weight=True, bias=norm_bias)
        self.layer = ViLLayer(
            dim=dim,
            direction=direction,
            conv_kind=conv_kind,
            conv_kernel_size=conv_kernel_size,
            seqlens=seqlens,
            norm_bias=norm_bias,
            proj_bias=proj_bias,
            num_blocks=num_blocks,
            init_weights=init_weights,
        )

        self.reset_parameters()

    def _forward_path(self, x):
        x = self.norm(x)
        x = self.layer(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop_path(x, self._forward_path)
        return x

    def reset_parameters(self):
        self.layer.reset_parameters()
        self.norm.reset_parameters()


class ViLBlockPair(nn.Module):
    def __init__(
            self,
            dim,
            drop_path=0.0,
            conv_kind="2d",
            conv_kernel_size=3,
            proj_bias=True,
            norm_bias=True,
            seqlens=None,
            num_blocks=None,
            init_weights="original",
    ):
        super().__init__()
        self.rowwise_from_top_left = ViLBlock(
            dim=dim,
            direction=SequenceTraversal.ROWWISE_FROM_TOP_LEFT,
            drop_path=drop_path,
            conv_kind=conv_kind,
            conv_kernel_size=conv_kernel_size,
            proj_bias=proj_bias,
            norm_bias=norm_bias,
            seqlens=seqlens,
            num_blocks=num_blocks,
            init_weights=init_weights,
        )
        self.rowwise_from_bot_right = ViLBlock(
            dim=dim,
            direction=SequenceTraversal.ROWWISE_FROM_BOT_RIGHT,
            drop_path=drop_path,
            conv_kind=conv_kind,
            conv_kernel_size=conv_kernel_size,
            proj_bias=proj_bias,
            norm_bias=norm_bias,
            seqlens=seqlens,
            num_blocks=num_blocks,
            init_weights=init_weights,
        )

    def forward(self, x):
        x = self.rowwise_from_top_left(x)
        x = self.rowwise_from_bot_right(x)
        return x


    




# ---------------------------
# DSFDecoder
# ---------------------------


class DynamicSparseConv(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.condition = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels//reduction, 1),
            nn.GELU(),
            nn.Conv2d(in_channels//reduction, 9 * in_channels, 1)
        )
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels)

    def forward(self, x):
        weights = self.condition(x).view(x.size(0), self.conv.weight.size(0), 9)
        weights = F.softmax(weights, dim=2).view(x.size(0)*x.size(1), 1, 3, 3)
        # return F.conv2d(x.view(1, -1, x.size(2), x.size(3)), 
        return F.conv2d(x.reshape(1, -1, x.size(2), x.size(3)), 
                       weights, padding=1, groups=x.size(0)*x.size(1)).view_as(x) + x

class DSFDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dynamic_block = nn.Sequential(
            DynamicSparseConv(in_channels),
            nn.ReLU(),
            DynamicSparseConv(in_channels)
        )
        self.channel_align = nn.Conv2d(2*in_channels, in_channels, 1)
        self.upsampler = nn.Sequential(
            nn.Conv2d(in_channels, 4*in_channels, 3, padding=1),  
            nn.PixelShuffle(2), 
            nn.Conv2d(in_channels, in_channels//2, 1)  
        )
        self.res_enhance = nn.Sequential(
            nn.Conv2d(in_channels//2, in_channels//8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels//8, in_channels//2, 3, padding=1)
        )

    def forward(self, x1, x2):
        
        x1 = self.dynamic_block(x1)
        x2 = self.dynamic_block(x2)
        fused = self.channel_align(torch.cat([x1, x2], dim=1))
        up = self.upsampler(fused)
        return up + self.res_enhance(up)


# class DSFDecoder(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv1 = nn.Conv2d(2 * in_channels, in_channels, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
#         self.up_conv = nn.ConvTranspose2d(
#             out_channels, 
#             out_channels, 
#             kernel_size=4, 
#             stride=2, 
#             padding=1
#         )
#         self.bn1 = nn.BatchNorm2d(in_channels)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x1, x2):
#         x = torch.cat((x1, x2), dim=1)  # [N, 2*C_in, H, W]
#         x = self.conv1(x)    # [N, C_in, H, W]
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.conv2(x)    # [N, C_out, H, W]
#         x = self.bn2(x)
#         x = self.relu(x)  
#         out = self.up_conv(x)  # [N, C_out, 2H, 2W]
#         return out

# ---------------------------
# 最终输出
# ---------------------------

class final_outblock(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.up_conv = nn.ConvTranspose2d(
            in_channels=64, 
            out_channels=1,  
            kernel_size=4,   
            stride=2,      
            padding=1      
        )
    
    def forward(self, x):
        return self.up_conv(x)




# ---------------------------
# CSB
# ---------------------------


class CSB(nn.Module):
   
    def __init__(self, high_ch, low_ch): 
        super().__init__()
        self.high_ch_conv = nn.Sequential(
            nn.Conv2d(high_ch, high_ch, 1),
            nn.BatchNorm2d(high_ch),
            nn.ReLU()
        )
        self.low_ch_conv = nn.Sequential(
            nn.Conv2d(low_ch, high_ch, 3, padding=1),
            nn.BatchNorm2d(high_ch),
            nn.ReLU()
        )
        self.context_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(high_ch, high_ch, 1),
            nn.Sigmoid()
        )
    
    def forward(self, high_ch_feat, low_ch_feat): 
        high_ch_processed = self.high_ch_conv(high_ch_feat)
        low_ch_processed = self.low_ch_conv(low_ch_feat)
        # low_ch_downsampled = F.interpolate(low_ch_processed, size=high_ch_feat.shape[-2:], mode='bilinear', align_corners=True)
        low_ch_downsampled = F.adaptive_max_pool2d(low_ch_processed, output_size=high_ch_feat.shape[-2:])
        context = self.context_extractor(high_ch_processed)
        fused = high_ch_processed * context + low_ch_downsampled
        
        return fused



class CASA(nn.Module):
    def __init__(self, channels, groups=16, reduction=16):
        super(CASA, self).__init__()
        
       
        assert channels % groups == 0
        self.groups = groups
        self.channels_per_group = channels // groups
        self.fc1 = nn.Linear(1, self.channels_per_group // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(self.channels_per_group // reduction, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        x_group = x.view(batch_size, self.groups, self.channels_per_group, height, width)
        s = x_group.mean(dim=[2, 3, 4], keepdim=False) 
        s = s.view(-1, 1)
        s = self.fc1(s)
        s = self.relu(s)
        s = self.fc2(s)
        weights = self.sigmoid(s)
        weights = weights.view(batch_size, self.groups)  
        spatial_weights = weights.view(batch_size, self.groups, 1, 1, 1)
        y_group = x_group * spatial_weights
        y = y_group.view(batch_size, channels, height, width)
        
        return y



class FVBLNet(nn.Module):
    def __init__(self, in_channels=3, classes=1):
        super(FVBLNet, self).__init__()
        self.conv_stem = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=1, stride=1, bias=False),  # 通道扩展
            nn.BatchNorm2d(64)
          )
        )

        self.vile3=ViLBlockPair(512)
        self.vile4=ViLBlockPair(1024)
        self.vilb=ViLBlockPair(1024)
        self.vild1=ViLBlockPair(512)
        self.vild2=ViLBlockPair(256)
        
        self.encoder1 = D2Encoder(64,128)
        self.encoder2 = D2Encoder(128,256)
        self.encoder3 = D2Encoder(256,512)
        self.encoder4 = D2Encoder(512,1024)
        self.decoder1 = DSFDecoder(1024,512)
        self.decoder2 = DSFDecoder(512,256)
        self.decoder3 = DSFDecoder(256,128)
        self.decoder4 = DSFDecoder(128,64)


        self.block2 = SSAMBlock(dim=1024)
        self.outblock=final_outblock()
        self.changeconv = nn.Sequential(
            nn.Conv2d(2048, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, 1)  # 增加非线性融合能力
        )

        self.cs1=CSB(256,128)
        self.cs2=CSB(512,256)
        self.cs3=CSB(1024,512)

       
        self.sg=CASA(1024)
        
    def forward(self, x):
 
    
        x=self.conv_stem(x)
        x1=self.encoder1(x)
        


        x2=self.encoder2(x1)
        skip_2 = self.cs1(x2,x1)
        

        x3=self.encoder3(x2)
        B,C,H,W=x3.shape
        x3 = x3.reshape(B, C, H * W).transpose(-1, -2) 
        x3 =self.vile3(x3)
        x3 = x3.transpose(-1, -2).reshape(B, C, H, W)  # 转换回 [B, C, H, W]
        skip_3 = self.cs2(x3,x2)
       


        x4 =self.encoder4(x3)
        B,C,H,W=x4.shape
        x4 = x4.reshape(B, C, H * W).transpose(-1, -2) 
        x4 =self.vile4(x4)
        x4 = x4.transpose(-1, -2).reshape(B, C, H, W)  # 转换回 [B, C, H, W]
        skip_4 = self.cs3(x4,x3)
       



        B,C,H,W=x4.shape
        xb = x4.reshape(B, C, H * W).transpose(-1, -2) 
        xb =self.vilb(xb)
        xb = xb.transpose(-1, -2).reshape(B, C, H, W)  # 转换回 [B, C, H, W]
        xc = self.block2(xb)
        # xb = self.block2(xb)
        xb = self.sg(xb)
        # xc = self.block2(x4)
        # xb = self.sg(x4)
        xb = self.changeconv(torch.cat([xc, xb], dim=1))
        
        xb= xb.reshape(B, C, H * W).transpose(-1, -2) 
        xb =self.vilb(xb)
        xb = xb.transpose(-1, -2).reshape(B, C, H, W)  # 转换回 [B, C, H, W]
       
       
        d1=self.decoder1(xb,skip_4)#512,14
        # d1=self.decoder1(xb,x4)
        B,C,H,W=d1.shape
        d1 = d1.reshape(B, C, H * W).transpose(-1, -2) 
        d1 =self.vild1(d1)
        d1 = d1.transpose(-1, -2).reshape(B, C, H, W)  # 转换回 [B, C, H, W]
        skip_3=torch.add(d1,skip_3)
        # x3=torch.add(d1,x3)
        d2=self.decoder2(d1,skip_3)#256,28
        # d2=self.decoder2(d1,x3)
        B,C,H,W=d2.shape
        d2 = d2.reshape(B, C, H * W).transpose(-1, -2) 
        d2 =self.vild2(d2)
        d2 = d2.transpose(-1, -2).reshape(B, C, H, W)  # 转换回 [B, C, H, W]
        skip_2=torch.add(d2,skip_2)
        # x2=torch.add(d2,x2)
        d3=self.decoder3(d2,skip_2)#128,56
        # d3=self.decoder3(d2,x2)
        x1=torch.add(d3,x1)
        d4=self.decoder4(d3,x1)#64,112
       
        Final_out=self.outblock(d4)
        return Final_out
