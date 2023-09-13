class ModuleList(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  __annotations__["0"] = __torch__.models.stereo.submodules.util_conv.___torch_mangle_118.BasicConv
  __annotations__["1"] = __torch__.models.stereo.submodules.util_conv.___torch_mangle_120.BasicConv
  __annotations__["2"] = __torch__.models.stereo.submodules.util_conv.___torch_mangle_122.BasicConv
  def forward(self: __torch__.torch.nn.modules.container.___torch_mangle_123.ModuleList) -> None:
    _0 = uninitialized(None)
    ops.prim.RaiseException("")
    return _0
  def __len__(self: __torch__.torch.nn.modules.container.___torch_mangle_123.ModuleList) -> int:
    return 3
