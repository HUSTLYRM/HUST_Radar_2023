class Unfold(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : None
  stride : Final[int] = 1
  kernel_size : Final[Tuple[int, int]] = (1, 49)
  padding : Final[int] = 0
  dilation : Final[int] = 1
  def forward(self: __torch__.torch.nn.modules.fold.Unfold,
    input: Tensor) -> Tensor:
    _0 = __torch__.torch.nn.functional.unfold
    _1 = _0(input, [1, 49], [1, 1], [0, 0], [1, 1], )
    return _1
