import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision import ops
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs import SwinT


# Layer Norm
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
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


# SE
class SqueezeExcitation(nn.Module):
    def __init__(self, dim, shrinkage_rate=0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden_dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(x)


# MBConv: Conv1*1 -> DW Conv3*3 -> [SE] -> Conv1*1
class MBConv(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.mbconv = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim),
            nn.GELU(),
            SqueezeExcitation(hidden_dim),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0)
        )

    def forward(self, x):
        return self.mbconv(x)


# CCM
class LDEL(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.c1 = nn.Conv2d(dim, hidden_dim, 1, 1, 0)
        self.act = nn.GELU()
        self.ccm = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=dim),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0),
            nn.GELU()
        )

    def forward(self, x):
        x = self.c1(x)
        x= self.ccm(x)
        return  x


class MAML(nn.Module):
    def __init__(self, dim, n_levels=4):
        super().__init__()
        self.n_levels = n_levels
        chunk_dim = dim // n_levels

        # Spatial Weighting
        self.conv1 = nn.Conv2d(chunk_dim, chunk_dim, 1, 1, 0, groups=chunk_dim)
        self.conv3 = nn.Conv2d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim)
        self.conv5 = nn.Conv2d(chunk_dim, chunk_dim, 5, 1, 2, groups=chunk_dim)
        self.conv7 = nn.Conv2d(chunk_dim, chunk_dim, 7, 1, 3, groups=chunk_dim)

        self.linear_1 = nn.Conv2d(chunk_dim, chunk_dim, 1, 1, 0, groups=chunk_dim)
        self.alpha = nn.Parameter(torch.ones((1, chunk_dim, 1, 1)))
        self.belt = nn.Parameter(torch.zeros((1, chunk_dim, 1, 1)))

        # # Feature Aggregation
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)

        # Activation
        self.act = nn.GELU()

    def forward(self, x):
        h, w = x.size()[-2:]
        xc = x.chunk(self.n_levels, dim=1)

        m1 = self.conv1(xc[0])
        m2 = self.conv3(F.adaptive_max_pool2d(xc[1], (h // 2, w // 2)))
        m4 = self.conv5(F.adaptive_max_pool2d(xc[2], (h // 2 ** 2, w // 2 ** 2)))
        m8 = self.conv7(F.adaptive_max_pool2d(xc[3], (h // 2 ** 3, w // 2 ** 3)))

        x_1 = torch.var(xc[0], dim=(-2, -1), keepdim=True)  # 计算方差
        m_1 = self.linear_1(m1 * self.alpha + x_1 * self.belt)
        x_2 = torch.var(xc[1], dim=(-2, -1), keepdim=True)  # 计算方差
        m_2 = F.interpolate(self.linear_1(m2 * self.alpha + x_2 * self.belt), size=(h, w), mode='nearest')
        x_4 = torch.var(xc[2], dim=(-2, -1), keepdim=True)  # 计算方差
        m_4 = F.interpolate(self.linear_1(m4 * self.alpha + x_4 * self.belt), size=(h, w), mode='nearest')
        x_8 = torch.var(xc[3], dim=(-2, -1), keepdim=True)  # 计算方差
        m_8 = F.interpolate(self.linear_1(m8 * self.alpha + x_8 * self.belt), size=(h, w), mode='nearest')

        out = self.aggr(torch.cat((m_1, m_2, m_4, m_8), dim=1))
        out = self.act(out)*x
        return out


class MAMB(nn.Module):
    def __init__(self, dim, ffn_scale=2.0):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.maml = MAML(dim)
        self.ldel = LDEL(dim)
        self.swin = SwinT.SwinT()
        self.c1 = nn.Conv2d(2*dim, dim, 1, 1, 0)
        self.c2 = nn.Conv2d(dim, dim, 1, 1, 0)

    def forward(self, x):
        res = x
        s = self.maml(self.norm(x)) + x
        c = self.ldel(self.norm(x)) + x
        x = self.c1(torch.cat([s, c], dim=1))
        x = self.swin(x)
        x = self.c2(x) + res
        return x


@ARCH_REGISTRY.register()
class MAMN(nn.Module):
    def __init__(self, dim, n_blocks=8, ffn_scale=2.0, upscaling_factor=4):
        super().__init__()
        self.to_feat = nn.Conv2d(3, dim, 3, 1, 1)

        self.feats = nn.Sequential(*[MAMB(dim, ffn_scale) for _ in range(n_blocks)])

        self.to_img = nn.Sequential(
            nn.Conv2d(dim, 3 * upscaling_factor ** 2, 3, 1, 1),
            nn.PixelShuffle(upscaling_factor)
        )

    def forward(self, x):
        x = self.to_feat(x)
        x = self.feats(x) + x
        x = self.to_img(x)
        return x

if __name__== '__main__':
    #############Test Model Complexity #############
    from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis
    # x = torch.randn(1, 3, 640, 360)
    # x = torch.randn(1, 3, 427, 240)
    x = torch.randn(1, 3, 320, 180)
    # x = torch.randn(1, 3, 256, 256)

    model = MAMN(dim=36, n_blocks=8, ffn_scale=2.0, upscaling_factor=4)
    # print(model)
    print(f'params: {sum(map(lambda x: x.numel(), model.parameters()))}')
    print(flop_count_table(FlopCountAnalysis(model, x), activations=ActivationCountAnalysis(model, x)))
    output = model(x)
    print(output.shape)