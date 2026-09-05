from torch import nn

from src.modules.dws_conv2d import DWSConv2d


class ForensicBranch(nn.Module):
    """Преобразует forensic-карту stride 8 в пирамиду признаков stride 8/16/32."""

    def __init__(self, in_ch, channels):
        super().__init__()

        self.channels_by_stride = {
            8: channels[0],
            16: channels[1],
            32: channels[2]
        }

        self.input_norm = nn.BatchNorm2d(in_ch)

        ch8, ch16, ch32 = channels
        self.input_projection = nn.Sequential(
            nn.Conv2d(in_ch, ch8, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch8),
            nn.GELU(),
        )

        self.refine_s8 = DWSConv2d(ch8, ch8)
        self.downsample_s16 = DWSConv2d(ch8, ch16, stride=2)
        self.downsample_s32 = DWSConv2d(ch16, ch32, stride=2)

    def forward(self, forensic_map):
        feat_s8 = self.input_norm(forensic_map)
        feat_s8 = self.input_projection(feat_s8)
        feat_s8 = self.refine_s8(feat_s8)

        feat_s16 = self.downsample_s16(feat_s8)
        feat_s32 = self.downsample_s32(feat_s16)

        return {
            8: feat_s8,
            16: feat_s16,
            32: feat_s32,
        }
