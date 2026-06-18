import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



print(torch.__version__)
print(torch.version.cuda)


class FreqAwarePool(nn.Module):


    def __init__(self, channels: int, axis: str = 'W'):
        super().__init__()
        assert axis in ('W', 'H')
        self.axis = axis
        # per-channel learnable blend weight (sigmoid → [0,1])
        self.alpha = nn.Parameter(torch.full((1, channels, 1, 1), 0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dim = 2 if self.axis == 'W' else 3  # collapse H→dim2, W→dim3
        avg = x.mean(dim=dim, keepdim=True)
        mx = x.amax(dim=dim, keepdim=True)
        w = torch.sigmoid(self.alpha)
        out = w * avg + (1.0 - w) * mx
        # ACAM convention: reshape to (B, C, 1, W) or (B, C, H, 1)
        if self.axis == 'W':
            return out.permute(0, 1, 3, 2)  # (B,C,1,W)
        return out  # (B,C,H,1)



class CrossAxisGate(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (B,C,1,1)
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, branch_feat: torch.Tensor) -> torch.Tensor:
        return self.gate(branch_feat)  # (B,C,1,1)

class MSDWConv(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        # 3×3 depthwise (padding=1 to preserve spatial size)
        self.dw3 = nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 1), groups=channels, bias=False)
        # 5×5 depthwise (padding=2 to preserve spatial size)
        self.dw5 = nn.Conv2d(channels, channels, kernel_size=(1, 5), padding=(0, 2), groups=channels, bias=False)
        # fuse: concat → pointwise → BN → ReLU
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([self.dw3(x), self.dw5(x)], dim=1))



class ChannelAttention(nn.Module):

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.se(x)



class DACAM(nn.Module):

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        # ① frequency-aware pooling (one per axis)
        self.pool_w = FreqAwarePool(channels, axis='W')  # → (B,C,1,W)
        self.pool_h = FreqAwarePool(channels, axis='H')  # → (B,C,H,1)

        # mix & split
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # ② cross-axis gates (each branch gates the other)
        self.gate_w = CrossAxisGate(channels)  # uses H-feat to gate W-branch
        self.gate_h = CrossAxisGate(channels)  # uses W-feat to gate H-branch

        # ③ MS-DWConv for each branch
        self.ms_w = MSDWConv(channels)
        self.ms_h = MSDWConv(channels)

        # spatial attention heads (1×1 conv → sigmoid per axis)
        self.attn_w = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False),
            nn.Sigmoid()
        )
        self.attn_h = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False),
            nn.Sigmoid()
        )

        # ④ channel attention (on global context before split)
        self.ch_attn = ChannelAttention(channels, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # ── ① Frequency-aware pooling ──────────────────────────
        fw = self.pool_w(x)  # (B, C, 1, W)
        fh = self.pool_h(x)  # (B, C, H, 1)

        # Combine: broadcast-multiply + 1×1 conv
        # reshape both to (B, C, H, W) for element-wise mult
        fw_b = fw.expand(B, C, H, W)
        fh_b = fh.expand(B, C, H, W)
        mixed = self.conv1x1(fw_b * fh_b)  # (B, C, H, W)

        # Split into two equal channel halves
        split = mixed.chunk(2, dim=1)  # each (B, C//2, H, W) if C even
        # For generality use full C in both branches (shared mixed)
        feat_w = mixed  # W-branch input
        feat_h = mixed  # H-branch input

        # Pool back to strip tensors for branch processing
        feat_w_strip = feat_w.mean(dim=2, keepdim=True)  # (B,C,1,W)
        feat_h_strip = feat_h.mean(dim=3, keepdim=True)  # (B,C,H,1)

        # ── ② Cross-axis gate ──────────────────────────────────
        # W-branch is gated by H context, H-branch gated by W context
        gate_for_w = self.gate_w(feat_h_strip)  # (B,C,1,1)
        gate_for_h = self.gate_h(feat_w_strip)  # (B,C,1,1)

        feat_w_strip = feat_w_strip * gate_for_w  # gated W-strip
        feat_h_strip = feat_h_strip * gate_for_h  # gated H-strip

        # ── ③ Multi-scale depthwise conv ──────────────────────
        feat_w_strip = self.ms_w(feat_w_strip)  # (B,C,1,W)
        feat_h_strip = self.ms_h(feat_h_strip)  # (B,C,H,1)

        # ── Spatial attention maps ─────────────────────────────
        attn_w = self.attn_w(feat_w_strip)  # (B,1,1,W)
        attn_h = self.attn_h(feat_h_strip)  # (B,1,H,1)

        # Combine into full spatial map (B,1,H,W) via outer product
        spatial_map = attn_w * attn_h  # broadcasts to (B,1,H,W)

        # ── ④ Channel-spatial joint re-weighting ──────────────
        ch_map = self.ch_attn(mixed)  # (B,C,1,1)

        # Apply: x * spatial_map * ch_map
        out = x * spatial_map * ch_map

        return out


class MS_SERN(nn.Module):
    def __init__(self, channels, dilRate):
        super().__init__()
        C = channels

        ### multi-scale spatial energy response normalization (MS-SERN)

        # multi-scale branches (keep C, H, W)
        self.dw1 = nn.Conv2d(C, C, kernel_size=3, padding=1, groups=C)
        self.bn1 = nn.BatchNorm2d(num_features=C)
        self.act1 = nn.GELU()
        self.pw1 = nn.Conv2d(C, C, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(num_features=C)

        self.dw2 = nn.Conv2d(C, C, kernel_size=5, padding=2, groups=C)
        self.bn3 = nn.BatchNorm2d(num_features=C)
        self.act2 = nn.GELU()
        self.pw2 = nn.Conv2d(C, C, kernel_size=1)
        self.bn4 = nn.BatchNorm2d(num_features=C)

        self.convd = nn.Conv2d(C, C, kernel_size=3, padding=dilRate, dilation=dilRate, groups=C)
        self.bn5 = nn.BatchNorm2d(num_features=C)
        self.act3 = nn.GELU()
        self.pw3 = nn.Conv2d(C, C, kernel_size=1)
        self.bn6 = nn.BatchNorm2d(num_features=C)

        # attention per branch (spatial)
        self.attn1 = nn.Conv2d(C, 1, kernel_size=1)
        self.attn2 = nn.Conv2d(C, 1, kernel_size=1)
        self.attn3 = nn.Conv2d(C, 1, kernel_size=1)

        # learnable fusion weights (softmax-normalized)
        self.w = nn.Parameter(torch.ones(3))

        # learnable fusion weights (softmax-normalized)
        self.w1_br = nn.Parameter(torch.ones(3))



        # GRN params
        self.gamma = nn.Parameter(torch.zeros(1, C, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, C, 1, 1))

    def branch_energy(self, F_i, attn_layer):
        A = torch.sigmoid(attn_layer(F_i))  # (N,1,H,W)
        E = torch.sum(A * (F_i ** 2), dim=(2, 3), keepdim=True)  # (N,C,1,1)
        return E

    def forward(self, x):
        # branches
        F1 = self.dw1(x)
        F1 = self.bn1(F1)
        F1 = self.act1(F1)
        F1 = self.pw1(F1)
        F1 = self.bn2(F1)

        F2 = self.dw2(x)
        F2 = self.bn3(F2)
        F2 = self.act2(F2)
        F2 = self.pw2(F2)
        F2 = self.bn4(F2)

        F3 = self.convd(x)
        F3 = self.bn5(F3)
        F3 = self.act3(F3)
        F3 = self.pw3(F3)
        F3 = self.bn6(F3)

        w1_br = F.softmax(self.w1_br, dim=0)
        fusedBracnh=w1_br[0]*F1 + w1_br[1]*F2 + w1_br[2]*F3

        # energies
        E1 = self.branch_energy(F1, self.attn1)
        E2 = self.branch_energy(F2, self.attn2)
        E3 = self.branch_energy(F3, self.attn3)

        w = F.softmax(self.w, dim=0)
        E = w[0] * E1 + w[1] * E2 + w[2] * E3

        mean_E = E.mean(dim=1, keepdim=True) + 1e-6
        E_hat = E / mean_E

        y = x * (self.gamma * E_hat + self.beta) + x
        y = y+fusedBracnh
        return y



class GRIB(nn.Module):

    def __init__(self, channels: int, window_size: int = 4):
        super().__init__()
        self.window_size = window_size
        self.scale = (channels // 4) ** -0.5

        # Depthwise-separable projections for Q, K, V
        def dw_proj(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, in_c, 3, padding=1, groups=in_c, bias=False),
                nn.Conv2d(in_c, out_c, 1, bias=False),
                nn.BatchNorm2d(out_c),
            )

        half = channels // 4
        self.q = dw_proj(channels, half)
        self.k = dw_proj(channels, half)
        self.v = dw_proj(channels, channels)

        # Output gating
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.out_norm = nn.BatchNorm2d(channels)

    def _window_partition(self, x: torch.Tensor, w: int):
        B, C, H, W = x.shape
        # Pad if needed
        pad_h = (w - H % w) % w
        pad_w = (w - W % w) % w
        x = F.pad(x, (0, pad_w, 0, pad_h))
        _, _, Hp, Wp = x.shape
        # (B, C, nH, w, nW, w) → (B*nH*nW, C, w, w)
        x = x.view(B, C, Hp // w, w, Wp // w, w)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        return x.view(-1, C, w, w), (Hp, Wp)

    def _window_reverse(self, x: torch.Tensor, w: int, B: int, Hp: int, Wp: int):
        C = x.shape[1]
        x = x.view(B, Hp // w, Wp // w, C, w, w)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        return x.view(B, C, Hp, Wp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        w = self.window_size

        q = self.q(x)  # (B, C//4, H, W)
        k = self.k(x)
        v = self.v(x)  # (B, C, H, W)

        # Partition into local windows
        q_w, (Hp, Wp) = self._window_partition(q, w)  # (B*nW, C//4, w, w)
        k_w, _ = self._window_partition(k, w)
        v_w, _ = self._window_partition(v, w)

        # Flatten spatial dims for attention
        Bw, Cq, _, _ = q_w.shape
        q_f = q_w.view(Bw, Cq, -1).permute(0, 2, 1)  # (Bw, w*w, Cq)
        k_f = k_w.view(Bw, Cq, -1)  # (Bw, Cq, w*w)
        v_f = v_w.view(Bw, C, -1).permute(0, 2, 1)  # (Bw, w*w, C)

        attn = torch.softmax(torch.bmm(q_f, k_f) * self.scale, dim=-1)
        out = torch.bmm(attn, v_f)  # (Bw, w*w, C)
        out = out.permute(0, 2, 1).view(Bw, C, w, w)

        # Reverse windows
        out = self._window_reverse(out, w, B, Hp, Wp)
        out = out[:, :, :H, :W]  # remove padding

        # Gated residual
        g = self.gate(out)
        return self.out_norm(x + out * g)


class Block(nn.Module):

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x


class ConvNextGithub(nn.Module):

    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            if i==0:
                stage.add_module('CustomBlock1',MS_SERN(channels=96,dilRate=5))
                stage.add_module('CustomBlock1_a', MS_SERN(channels=96, dilRate=5))
            elif i==1:
                stage.add_module('CustomBlock2',MS_SERN(channels=192,dilRate=5))
                stage.add_module('CustomBlock2_a', MS_SERN(channels=192, dilRate=5))
            elif i==2:
                stage.add_module('CustomBlock3',DACAM(channels=384,reduction=16))
                stage.add_module('CustomBlock3_a', DACAM(channels=384, reduction=16))
            elif i==3:
                stage.add_module('CustomBlock4',DACAM(channels=768, reduction=16))
                stage.add_module('CustomBlock4_a', DACAM(channels=768, reduction=16))
                stage.add_module('Attention',GRIB(channels=768,window_size=4))


            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)



    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class LayerNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


model_urls = {
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",

}


@register_model
def convnext_small(pretrained=False, in_22k=False, **kwargs):
    model = ConvNextGithub(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_small_22k'] if in_22k else model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

