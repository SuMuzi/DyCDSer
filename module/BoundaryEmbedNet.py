


def depthwise_separable_conv_block(in_c, out_c, kernel_size=3, padding=1):
    return nn.Sequential(
        # Depthwise Convolution
        nn.Conv2d(in_c, in_c, kernel_size=kernel_size, padding=padding, groups=in_c, bias=False),
        nn.BatchNorm2d(in_c),
        nn.ReLU(inplace=True),

        # Pointwise Convolution
        nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),

        # 第二个深度可分离卷积
        nn.Conv2d(out_c, out_c, kernel_size=kernel_size, padding=padding, groups=out_c, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),

        nn.Conv2d(out_c, out_c, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )