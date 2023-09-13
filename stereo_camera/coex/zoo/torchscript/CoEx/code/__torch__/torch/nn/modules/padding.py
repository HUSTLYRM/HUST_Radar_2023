class ZeroPad2d(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : None
  value : Final[float] = 0.
  padding : Final[Tuple[int, int, int, int]] = (48, 0, 0, 0)
  def forward(self: __torch__.torch.nn.modules.padding.ZeroPad2d,
    input: Tensor) -> Tensor:
    _0 = __torch__.torch.nn.functional._pad(input, [48, 0, 0, 0], "constant", 0., )
    return _0
