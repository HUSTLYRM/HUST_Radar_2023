class Conv2x(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  concat : bool
  is_3d : bool
  conv1 : __torch__.models.stereo.submodules.util_conv.___torch_mangle_64.BasicConv
  conv2 : __torch__.models.stereo.submodules.util_conv.___torch_mangle_67.BasicConv
  def forward(self: __torch__.models.stereo.submodules.util_conv.___torch_mangle_68.Conv2x,
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
