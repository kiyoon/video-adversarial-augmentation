from torch import nn

from .i3d_head import I3DHead
from .swin_transformer import SwinTransformer3D


class SwinTransformer3DWithHead(nn.Module):
    def __init__(
        self,
        num_classes,
        pretrained=None,
        pretrained2d=True,
        patch_size=(4, 4, 4),
        in_chans=3,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=(2, 7, 7),
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        norm_layer=nn.LayerNorm,
        patch_norm=False,
        frozen_stages=-1,
        use_checkpoint=False,
        # Head
        head_in_channels=768,
        head_spatial_type="avg",
        head_dropout_ratio=0.5,
    ):
        super().__init__()
        self.backbone = SwinTransformer3D(
            pretrained=pretrained,
            pretrained2d=pretrained2d,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            patch_norm=patch_norm,
            frozen_stages=frozen_stages,
            use_checkpoint=use_checkpoint,
        )

        self.cls_head = I3DHead(
            num_classes=num_classes,
            in_channels=head_in_channels,
            spatial_type=head_spatial_type,
            dropout_ratio=head_dropout_ratio,
        )

    def get_optim_policies(self):
        return self.parameters()

    def features(self, x):
        N, C, T, H, W = x.shape
        if C != self.backbone.in_chans and T == self.backbone.in_chans:
            # NTCHW -> NCTHW
            x = x.transpose(1, 2)
        return self.backbone(x)

    def logits(self, features):
        return self.cls_head(features)

    def forward(self, x):
        return self.logits(self.features(x))


def video_swin_tiny_imagenet(num_classes, input_channel_num=3):
    model = SwinTransformer3DWithHead(
        num_classes=num_classes,
        patch_size=(2, 4, 4),
        in_chans=input_channel_num,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=(8, 7, 7),
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        patch_norm=True,
        head_in_channels=768,
        head_spatial_type="avg",
        head_dropout_ratio=0.5,
        pretrained2d=True,
        pretrained="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth",
    )

    model.backbone.init_weights()

    return model
