import torch.nn as nn


from examples import Swish




class BnActivate(nn.Module):
    def __init__(self, num_features):
        self._seq = nn.Sequential(
            nn.BatchNorm2d(num_features),

        )




class SELayer(nn.Module):
    """
    Squeeze and Excitation (SE)
    - A Channel-Wise Gating Layer

    ABOUT:
        Auto-encoder over the averaged pooled channels of an image
        then multiplies the original input channels, by the sigmoid activations.

    DIAGRAM:
        ---> (B, C, H, W) ----------------------------------------------------- * ---> (B, C, H, W)
                  \                                                            /
                   \--> (B, C, 1, 1) --> (B, C//reduction) --> (B, C, 1, 1) --/  multiply by sigmoid
                          pooling            encoder              decoder        activation of decoder output
    """

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # reduce
        y = self.avg_pool(x)
        # auto-encode
        y = self.fc(y.view(b, c)).view(b, c, 1, 1)
        # multiply
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self._seq = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=5, padding=2),
            nn.Conv2d(dim, dim, kernel_size=1),
                nn.BatchNorm2d(dim),
                Swish(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                SELayer(dim),
        )

    def forward(self, x):
        # TODO: why is the 0.1 here? what is the point of this.
        #       I dont think it should be here?
        return x + 0.1 * self._seq(x)


class EncoderResidualBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=5, padding=2),
            nn.Conv2d(dim, dim, kernel_size=1),
                nn.BatchNorm2d(dim),
                Swish(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                SELayer(dim),
        )

    def forward(self, x):
        # TODO: why is the 0.1 here? what is the point of this.
        #       I dont think it should be here?
        return x + 0.1 * self.seq(x)


class DecoderResidualBlock(nn.Module):

    def __init__(self, dim, n_group):
        super().__init__()

        self._cell = nn.Sequential(
            # TODO: there should be BN here
            nn.Conv2d(dim, n_group * dim, kernel_size=1),
                nn.BatchNorm2d(n_group * dim),
                Swish(),
            nn.Conv2d(n_group * dim, n_group * dim, kernel_size=5, padding=2, groups=n_group),
                nn.BatchNorm2d(n_group * dim),
                Swish(),
            nn.Conv2d(n_group * dim, dim, kernel_size=1),
                nn.BatchNorm2d(dim),
            SELayer(dim),
        )

    def forward(self, x):
        # TODO: why is the 0.1 here? what is the point of this.
        #       I dont think it should be here?
        return x + 0.1 * self._cell(x)
