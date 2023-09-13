class Conv3d(Module):
  __parameters__ = ["weight", "bias", ]
  __buffers__ = []
  weight : Tensor
  bias : Optional[Tensor]
  training : bool
  _is_full_backward_hook : None
  transposed : bool
  _reversed_padding_repeated_twice : Tuple[int, int, int, int, int, int]
  out_channels : Final[int] = 8
  kernel_size : Final[Tuple[int, int, int]] = (3, 3, 3)
  in_channels : Final[int] = 8
  output_padding : Final[Tuple[int, int, int]] = (0, 0, 0)
  dilation : Final[Tuple[int, int, int]] = (1, 1, 1)
  stride : Final[Tuple[int, int, int]] = (1, 1, 1)
  padding : Final[Tuple[int, int, int]] = (1, 1, 1)
  groups : Final[int] = 1
  padding_mode : Final[str] = "zeros"
  def forward(self: __torch__.torch.nn.modules.conv.___torch_mangle_131.Conv3d,
    input: Tensor) -> Tensor:
    _0 = torch.conv3d(input, self.weight, self.bias, [1, 1, 1], [1, 1, 1], [1, 1, 1], 1)
    return _0
