class Conv2x(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  concat : bool
  is_3d : bool
  conv1 : __torch__.models.stereo.submodules.util_conv.BasicConv
  conv2 : __torch__.models.stereo.submodules.util_conv.___torch_mangle_57.BasicConv
  def forward(self: __torch__.models.stereo.submodules.util_conv.Conv2x,
    x: Tensor,
    rem: Tensor) -> Tensor:
    x0 = (self.conv1).forward(x, )
    _0 = torch.eq(torch.size(x0), torch.size(rem))
    if _0:
      pass
    else:
      ops.prim.RaiseException("AssertionError: ")
    if self.concat:
      x1 = torch.cat([x0, rem], 1)
    else:
      x1 = torch.add(x0, rem, alpha=1)
    return (self.conv2).forward(x1, )
class BasicConv(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  relu : bool
  use_bn : bool
  conv : __torch__.torch.nn.modules.conv.ConvTranspose2d
  bn : __torch__.torch.nn.modules.batchnorm.___torch_mangle_5.BatchNorm2d
  LeakyReLU : __torch__.torch.nn.modules.activation.LeakyReLU
  def forward(self: __torch__.models.stereo.submodules.util_conv.BasicConv,
    x: Tensor) -> Tensor:
    x2 = (self.conv).forward(x, None, )
    if self.use_bn:
      x3 = (self.bn).forward(x2, )
    else:
      x3 = x2
    if self.relu:
      x4 = (self.LeakyReLU).forward(x3, )
    else:
      x4 = x3
    return x4
