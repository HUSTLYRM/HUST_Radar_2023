class ReLU6(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : None
  min_val : Final[float] = 0.
  inplace : Final[bool] = True
  max_val : Final[float] = 6.
  def forward(self: __torch__.torch.nn.modules.activation.ReLU6,
    input: Tensor) -> Tensor:
    _0 = __torch__.torch.nn.functional.hardtanh
    return _0(input, 0., 6., True, )
class LeakyReLU(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : None
  negative_slope : Final[float] = 0.01
  inplace : Final[bool] = False
  def forward(self: __torch__.torch.nn.modules.activation.LeakyReLU,
    input: Tensor) -> Tensor:
    _1 = __torch__.torch.nn.functional.leaky_relu
    return _1(input, 0.01, False, )
class ReLU(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : None
  inplace : Final[bool] = False
  def forward(self: __torch__.torch.nn.modules.activation.ReLU,
    input: Tensor) -> Tensor:
    _2 = __torch__.torch.nn.functional.relu(input, False, )
    return _2
