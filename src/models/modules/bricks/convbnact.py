import torch.nn as nn

class ConvBnAct(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding,
                 stride=1,
                 bias=False,
                 act_layer=nn.ReLU):
        """Init ConvBnAct.
        
        Args:
            in_channels:
            out_channels:
            kernel_size:
            padding:
            stride:
            bias:
            act_layer:
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x
