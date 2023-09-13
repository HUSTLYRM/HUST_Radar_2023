class Conv2d(Module):
  __parameters__ = ["weight", "bias", ]
  __buffers__ = []
  weight : Tensor
  bias : Optional[Tensor]
  training : bool
  _is_full_backward_hook : None
  transposed : bool
  _reversed_padding_repeated_twice : Tuple[int, int, int, int]
  out_channels : Final[int] = 576
  kernel_size : Final[Tuple[int, int]] = (3, 3)
  in_channels : Final[int] = 576
  output_padding : Final[Tuple[int, int]] = (0, 0)
  dilation : Final[Tuple[int, int]] = (1, 1)
  stride : Final[Tuple[int, int]] = (2, 2)
  padding : Final[Tuple[int, int]] = (1, 1)
  groups : Final[int] = 576
  padding_mode : Final[str] = "zeros"
  def forward(self: __torch__.torch.nn.modules.conv.___torch_mangle_45.Conv2d,
    input: Tensor) -> Tensor:
    _0 = (self)._conv_forward(input, self.weight, self.bias, )
    return _0
  def _conv_forward(self: __torch__.torch.nn.modules.conv.___torch_mangle_45.Conv2d,
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor]) -> Tensor:
    _1 = torch.conv2d(input, weight, bias, [2, 2], [1, 1], [1, 1], 576)
    return _1
