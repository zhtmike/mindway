import mindspore.mint as mint
import mindspore.mint.nn.functional as F
from mindspore import Tensor

class Conv1d(mint.nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 dtype=None):
        assert isinstance(kernel_size, int)
        assert isinstance(stride, int)
        assert isinstance(dilation, int)
        kernel_size = (1, kernel_size)
        stride = (1, stride)
        dilation = (1, dilation)

        if isinstance(padding, int):
            padding = (0, padding)
        else:
            assert isinstance(padding, str)

        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, dtype=dtype)

    def construct(self, x: Tensor):
        x = mint.unsqueeze(x, dim=-2)
        x = super().construct(x)
        x = mint.squeeze(x, dim=-2)
        return x


class ConvTranspose1d(mint.nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0,
                 groups=1, bias=True, dilation=1, padding_mode="zeros", dtype=None):
        assert isinstance(kernel_size, int)
        assert isinstance(stride, int)
        assert isinstance(dilation, int)
        kernel_size = (1, kernel_size)
        stride = (1, stride)
        dilation = (1, dilation)

        assert isinstance(padding, int)
        assert isinstance(output_padding, int)
        padding = (0, padding)
        output_padding = (0, output_padding)

        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias, dilation=dilation, padding_mode=padding_mode, dtype=dtype)

    def construct(self, x: Tensor):
        x = mint.unsqueeze(x, dim=-2)
        x = super().construct(x)
        x = mint.squeeze(x, dim=-2)
        return x
    

def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    assert isinstance(stride, int)
    assert isinstance(dilation, int)

    stride = (1, stride)
    dilation = (1, dilation)
    if isinstance(padding, int):
        padding = (0, padding)
    else:
        assert isinstance(padding, str)

    input = mint.unsqueeze(input, dim=-2)
    weight = mint.unsqueeze(weight, dim=-2)
    output = F.conv2d(input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    output = mint.squeeze(output, dim=-2)
    return output


def conv_transpose1d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    assert isinstance(stride, int)
    assert isinstance(dilation, int)
    assert isinstance(padding, int)
    assert isinstance(output_padding, int)
    stride = (1, stride)
    dilation = (1, dilation)
    padding = (0, padding)
    output_padding = (0, output_padding)

    input = mint.unsqueeze(input, dim=-2)
    weight = mint.unsqueeze(weight, dim=-2)
    output = F.conv_transpose2d(input, weight, bias=bias, stride=stride, padding=padding, output_padding=output_padding, groups=groups, dilation=dilation)
    output = mint.squeeze(output, dim=-2)
    return output
