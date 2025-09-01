import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Sequential as Seq

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath
from timm.models.registry import register_model

import random
import warnings
warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# IMAGENET 
def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 
        'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'logvig': _cfg(crop_pct=0.9, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
}

    
class Stem(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Stem, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_dim, output_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_dim // 2),
            nn.GELU(),
            nn.Conv2d(output_dim // 2, output_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.GELU(),   
        )
        
    def forward(self, x):
        return self.stem(x)

class HighResBranch(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(HighResBranch, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim, output_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_dim // 2),
            nn.GELU(),
            nn.Conv2d(output_dim // 2, output_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.GELU()
        )
    
    def forward(self, x):
        return self.conv(x)

class DepthWiseSeparable(nn.Module):
    def __init__(self, in_dim, kernel, expansion=4):
        super().__init__()

        self.pw1 = nn.Conv2d(in_dim, in_dim * 4, 1)
        self.norm1 = nn.BatchNorm2d(in_dim * 4)
        self.act1 = nn.GELU()
        
        self.dw = nn.Conv2d(in_dim * 4, in_dim * 4, kernel_size=kernel, stride=1, padding=1, groups=in_dim * 4)
        self.norm2 = nn.BatchNorm2d(in_dim * 4)
        self.act2 = nn.GELU()
        
        self.pw2 = nn.Conv2d(in_dim * 4, in_dim, 1)
        self.norm3 = nn.BatchNorm2d(in_dim)

    def forward(self, x):
        x = self.pw1(x)
        x = self.norm1(x)
        x = self.act1(x)
        
        x = self.dw(x)
        x = self.norm2(x)
        x = self.act2(x)
        
        x = self.pw2(x)
        x = self.norm3(x)
        return x

    
class InvertedResidual(nn.Module):
    def __init__(self, dim, kernel, expansion_ratio=4., drop=0., drop_path=0., use_layer_scale=True, layer_scale_init_value=1e-5):
        super().__init__()

        self.dws = DepthWiseSeparable(in_dim=dim, kernel=kernel, expansion=expansion_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.dws(x))
        else:
            x = x + self.drop_path(self.dws(x))
        return x
   


class Log_GraphConv4d(nn.Module):

    def __init__(self, in_channels, out_channels, K):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.K = K


    def forward(self, x):
        B, C, H, W = x.shape
        H = torch.tensor([H])
        W = torch.tensor([W])
        x_j = x - x
        Hbit = H.detach().item().bit_length()
        Wbit = W.detach().item().bit_length()
        for i in range(1, Hbit):
            x_c = x - torch.cat(
                [x[:, :, -(self.K**i - 1) :, :], x[:, :, : -(self.K**i - 1), :]],
                dim=2,
            )
            x_j = torch.max(x_j, x_c)
        for i in range(1, Hbit):
            x_c = x - torch.cat(
                [x[:, :, (self.K**i - 1) :, :], x[:, :, : (self.K**i - 1), :]],
                dim=2,
            )
            x_j = torch.max(x_j, x_c)

        for i in range(1, Wbit):
            x_r = x - torch.cat(
                [x[:, :, :, -(self.K**i - 1) :], x[:, :, :, : -(self.K**i - 1)]],
                dim=3,
            )
            x_j = torch.max(x_j, x_r)
        for i in range(1, Wbit):
            x_r = x - torch.cat(
                [x[:, :, :, (self.K**i - 1) :], x[:, :, :, : (self.K**i - 1)]],
                dim=3,
            )
            x_j = torch.max(x_j, x_r)

        x = torch.cat([x, x_j], dim=1)
        return self.nn(x)


# class ConditionalPositionEncoding(nn.Module):
#     """
#     Implementation of conditional positional encoding. For more details refer to paper: 
#     `Conditional Positional Encodings for Vision Transformers <https://arxiv.org/pdf/2102.10882.pdf>`_
#     """
#     def __init__(self, in_channels, kernel_size):
#         super().__init__()
#         self.pe = nn.Conv2d(
#             in_channels=in_channels,
#             out_channels=in_channels,
#             kernel_size=kernel_size,
#             stride=1,
#             padding=kernel_size // 2,
#             bias=True,
#             groups=in_channels
#         )

#     def forward(self, x):
#         x = self.pe(x) + x
#         return x

class RepCPE(nn.Module):
    """
    This implementation of reparameterized conditional positional encoding was originally implemented
    in the following repository: https://github.com/apple/ml-fastvit
    
    Implementation of conditional positional encoding.

    For more details refer to paper:
    `Conditional Positional Encodings for Vision Transformers <https://arxiv.org/pdf/2102.10882.pdf>`_
    """

    def __init__(
        self,
        in_channels,
        embed_dim,
        spatial_shape = (7, 7),
        inference_mode=False,
    ) -> None:
        """Build reparameterizable conditional positional encoding

        Args:
            in_channels: Number of input channels.
            embed_dim: Number of embedding dimensions. Default: 768
            spatial_shape: Spatial shape of kernel for positional encoding. Default: (7, 7)
            inference_mode: Flag to instantiate block in inference mode. Default: ``False``
        """
        super(RepCPE, self).__init__()
        self.spatial_shape = spatial_shape
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        self.groups = embed_dim

        if inference_mode:
            self.reparam_conv = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.embed_dim,
                kernel_size=self.spatial_shape,
                stride=1,
                padding=int(self.spatial_shape[0] // 2),
                groups=self.embed_dim,
                bias=True,
            )
        else:
            self.pe = nn.Conv2d(
                in_channels,
                embed_dim,
                spatial_shape,
                1,
                int(spatial_shape[0] // 2),
                bias=True,
                groups=embed_dim,
            )

    def forward(self, x: torch.Tensor):
        if hasattr(self, "reparam_conv"):
            x = self.reparam_conv(x)
            return x
        else:
            x = self.pe(x) + x
            return x

    def reparameterize(self):
        # Build equivalent Id tensor
        input_dim = self.in_channels // self.groups
        kernel_value = torch.zeros(
            (
                self.in_channels,
                input_dim,
                self.spatial_shape[0],
                self.spatial_shape[1],
            ),
            dtype=self.pe.weight.dtype,
            device=self.pe.weight.device,
        )
        for i in range(self.in_channels):
            kernel_value[
                i,
                i % input_dim,
                self.spatial_shape[0] // 2,
                self.spatial_shape[1] // 2,
            ] = 1
        id_tensor = kernel_value

        # Reparameterize Id tensor and conv
        w_final = id_tensor + self.pe.weight
        b_final = self.pe.bias

        # Introduce reparam conv
        self.reparam_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.embed_dim,
            kernel_size=self.spatial_shape,
            stride=1,
            padding=int(self.spatial_shape[0] // 2),
            groups=self.embed_dim,
            bias=True,
        )
        self.reparam_conv.weight.data = w_final
        self.reparam_conv.bias.data = b_final

        for para in self.parameters():
            para.detach_()
        self.__delattr__("pe")


class Grapher(nn.Module):
    def __init__(self, in_channels, K):
        super(Grapher, self).__init__()
        self.channels = in_channels
        self.K = K

        self.cpe = RepCPE(in_channels=in_channels, embed_dim=in_channels, spatial_shape=(7, 7))
        # self.cpe = ConditionalPositionEncoding(in_channels, kernel_size=7)
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.graph_conv = Log_GraphConv4d(
            in_channels, in_channels * 2, K=self.K
        )
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )

       
    def forward(self, x):
        x = self.cpe(x)
        x = self.fc1(x)
        x = self.graph_conv(x)
        x = self.fc2(x)

        return x


class Merge(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # Ensure the convolution layer expects the correct number of channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_dim),
            nn.GELU()
        )
        
    def forward(self, x_low, x_high):
        x_low = x_low.to(dtype=torch.float32, device=x_high.device)
        x_high = x_high.to(dtype=torch.float32, device=x_high.device)

        if x_low.shape[-2:] != x_high.shape[-2:]:
            x_low = F.interpolate(x_low, size=x_high.shape[-2:], mode='bilinear', align_corners=False)

        if x_low.shape[1] != x_high.shape[1]:
            x_high = self.adjust_channels(x_high, x_low.shape[1])

        x = x_low + x_high

        return self.conv(x)

    def adjust_channels(self, x, target_channels):
        conv_adjust = nn.Conv2d(x.shape[1], target_channels, kernel_size=1, stride=1, padding=0)
        conv_adjust.to(x.device, x.dtype)
        return conv_adjust(x)


class Log_ConvBlock(nn.Module):
    def __init__(self, in_dim, drop_path=0., K=2, use_layer_scale=True, layer_scale_init_value=1e-5):
        super().__init__()
        
        self.mixer = Grapher(in_dim, K)
        self.ffn = nn.Sequential(
            nn.Conv2d(in_dim, in_dim * 4, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_dim * 4),
            nn.GELU(),
            nn.Conv2d(in_dim * 4, in_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_dim),
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones(in_dim), requires_grad=True) 
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones(in_dim), requires_grad=True) 
        
    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.mixer(x))
            x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.ffn(x))
        else:
            x = x + self.drop_path(self.mixer(x))
            x = x + self.drop_path(self.ffn(x))
        return x


class Downsample(nn.Module):
    """ 
    Convolution-based downsample
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
        )
    def forward(self, x):
        x = self.conv(x)
        return x


class LogViG(torch.nn.Module):
    def __init__(self, blocks, channels, kernels, stride,
                 act_func, dropout=0., drop_path=0., emb_dims=512,
                 K=2, distillation=True, num_classes=1000):
        super(LogViG, self).__init__()

        self.distillation = distillation
        self.stage_names = ['stem', 'local_1', 'local_2', 'local_3', 'global']
        
        n_blocks = sum([sum(x) for x in blocks])
        dpr = [x.item() for x in torch.linspace(0, drop_path, n_blocks)]  # stochastic depth decay rule 
        dpr_idx = 0

        self.stem = Stem(input_dim=3, output_dim=channels[0])

        self.high_res_branch = HighResBranch(input_dim=channels[0], output_dim=channels[1])
        
        self.backbone = []
        for i in range(len(blocks)):
            stage = []
            local_stages = blocks[i][0]
            global_stages = blocks[i][1]
            if i > 0:
                stage.append(Downsample(channels[i-1], channels[i]))
            for _ in range(local_stages):
                stage.append(InvertedResidual(dim=channels[i], kernel=3, expansion_ratio=4, drop_path=dpr[dpr_idx]))
                dpr_idx += 1
            for _ in range(global_stages):
                stage.append(Log_ConvBlock(channels[i], drop_path=dpr[dpr_idx], K=K[i]))
                dpr_idx += 1

            self.backbone.append(nn.Sequential(*stage))
            
        # Merge module to combine high and low-resolution features
        self.merge = Merge(in_dim=channels[i], out_dim=channels[i])

        self.backbone = nn.Sequential(*self.backbone)
                
        self.prediction = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                        nn.Conv2d(channels[-1], emb_dims, kernel_size=1, bias=True),
                                        nn.BatchNorm2d(emb_dims),
                                        nn.GELU(),
                                        nn.Dropout(dropout))
        
        self.head = nn.Conv2d(emb_dims, num_classes, kernel_size=1, bias=True)
        
        if self.distillation:
            self.dist_head = nn.Conv2d(emb_dims, num_classes, 1, bias=True)
        
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, inputs):
        x = self.stem(inputs)
        B, C, H, W = x.shape

        x_high = self.high_res_branch(x)

        x = self.backbone(x)

        # Merge low and high-res features
        x = self.merge(x, x_high)
            
        x = self.prediction(x)
            
        if self.distillation:
            x = self.head(x).squeeze(-1).squeeze(-1), self.dist_head(x).squeeze(-1).squeeze(-1)
            if not self.training:
                x = (x[0] + x[1]) / 2
        else:
            x = self.head(x).squeeze(-1).squeeze(-1)
        return x


@register_model
def Ti_LogViG(pretrained=False, **kwargs):
    model = LogViG(blocks=[[3,3], [3,3], [9,3], [3,3]],
                      channels=[32, 64, 128, 256],
                      kernels=3,
                      stride=1,
                      act_func='gelu',
                      dropout=0.,
                      drop_path=0.1,
                      emb_dims=512,
                      K=[2, 2, 2, 2],
                      distillation=True,
                      num_classes=1000)
    model.default_cfg = default_cfgs['logvig']
    return model
    
@register_model
def Ti_LogViG_3(pretrained=False, **kwargs):
    model = LogViG(blocks=[[3,3], [3,3], [9,3], [3,3]],
                      channels=[32, 64, 128, 224],
                      kernels=3,
                      stride=1,
                      act_func='gelu',
                      dropout=0.,
                      drop_path=0.1,
                      emb_dims=768,
                      K=[3, 3, 2, 2],
                      distillation=True,
                      num_classes=1000)
    model.default_cfg = default_cfgs['logvig']
    return model
    
@register_model
def S_LogViG(pretrained=False, **kwargs):
    model = LogViG(blocks=[[5,5], [5,5], [15,5], [5,5]],
                      channels=[32, 64, 128, 256],
                      kernels=3,
                      stride=1,
                      act_func='gelu',
                      dropout=0.,
                      drop_path=0.1,
                      emb_dims=768,
                      K=[2, 2, 2, 2],
                      distillation=True,
                      num_classes=1000)
    model.default_cfg = default_cfgs['logvig']
    return model

@register_model
def B_LogViG(pretrained=False, **kwargs):
    model = LogViG(blocks=[[5,5], [5,5], [18,6], [5,5]],
                      channels=[48, 96, 192, 384],
                      kernels=3,
                      stride=1,
                      act_func='gelu',
                      dropout=0.,
                      drop_path=0.1,
                      emb_dims=768,
                      K=[2, 2, 2, 2],
                      distillation=True,
                      num_classes=1000)
    model.default_cfg = default_cfgs['logvig']
    return model

def reparameterize_model(model: torch.nn.Module) -> nn.Module:
    """ Method returns a model where a multi-branched structure
        used in training is re-parameterized into a single branch
        for inference.

    :param model: MobileOne model in train mode.
    :return: MobileOne model in inference mode.
    """
    model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'reparameterize'):
            module.reparameterize()
    return model