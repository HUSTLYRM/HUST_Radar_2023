class BatchNorm2d(Module):
  __parameters__ = ["weight", "bias", ]
  __buffers__ = ["running_mean", "running_var", "num_batches_tracked", ]
  weight : Tensor
  bias : Tensor
  running_mean : Tensor
  running_var : Tensor
  num_batches_tracked : Tensor
  training : bool
  _is_full_backward_hook : None
  affine : Final[bool] = True
  num_features : Final[int] = 32
  eps : Final[float] = 1.0000000000000001e-05
  track_running_stats : Final[bool] = True
  momentum : Final[float] = 0.10000000000000001
  def forward(self: __torch__.torch.nn.modules.batchnorm.BatchNorm2d,
    input: Tensor) -> Tensor:
    _0 = __torch__.torch.nn.functional.batch_norm
    _1 = (self)._check_input_dim(input, )
    if self.training:
      _2 = torch.add(self.num_batches_tracked, 1, 1)
      self.num_batches_tracked = _2
    else:
      pass
    if self.training:
      bn_training = True
    else:
      bn_training = False
    _3 = _0(input, self.running_mean, self.running_var, self.weight, self.bias, bn_training, 0.10000000000000001, 1.0000000000000001e-05, )
    return _3
  def _check_input_dim(self: __torch__.torch.nn.modules.batchnorm.BatchNorm2d,
    input: Tensor) -> None:
    if torch.ne(torch.dim(input), 4):
      _4 = torch.format("expected 4D input (got {}D input)", torch.dim(input))
      ops.prim.RaiseException(_4)
    else:
      pass
    return None
class BatchNorm3d(Module):
  __parameters__ = ["weight", "bias", ]
  __buffers__ = ["running_mean", "running_var", "num_batches_tracked", ]
  weight : Tensor
  bias : Tensor
  running_mean : Tensor
  running_var : Tensor
  num_batches_tracked : Tensor
  training : bool
  _is_full_backward_hook : None
  affine : Final[bool] = True
  num_features : Final[int] = 8
  eps : Final[float] = 1.0000000000000001e-05
  track_running_stats : Final[bool] = True
  momentum : Final[float] = 0.10000000000000001
  def forward(self: __torch__.torch.nn.modules.batchnorm.BatchNorm3d,
    input: Tensor) -> Tensor:
    _5 = __torch__.torch.nn.functional.batch_norm
    _6 = (self)._check_input_dim(input, )
    if self.training:
      _7 = torch.add(self.num_batches_tracked, 1, 1)
      self.num_batches_tracked = _7
    else:
      pass
    if self.training:
      bn_training = True
    else:
      bn_training = False
    _8 = _5(input, self.running_mean, self.running_var, self.weight, self.bias, bn_training, 0.10000000000000001, 1.0000000000000001e-05, )
    return _8
  def _check_input_dim(self: __torch__.torch.nn.modules.batchnorm.BatchNorm3d,
    input: Tensor) -> None:
    if torch.ne(torch.dim(input), 5):
      _9 = torch.format("expected 5D input (got {}D input)", torch.dim(input))
      ops.prim.RaiseException(_9)
    else:
      pass
    return None
