class Sequential(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  __annotations__["0"] = __torch__.timm.models.efficientnet_blocks.DepthwiseSeparableConv
  def forward(self: __torch__.torch.nn.modules.container.Sequential,
    input: Tensor) -> Tensor:
    input0 = (getattr(self, "0")).forward(input, )
    return input0
  def __len__(self: __torch__.torch.nn.modules.container.Sequential) -> int:
    return 1
class ModuleList(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  __annotations__["0"] = __torch__.models.stereo.submodules.utils.___torch_mangle_81.channelAtt
  __annotations__["1"] = __torch__.models.stereo.submodules.utils.___torch_mangle_85.channelAtt
  __annotations__["2"] = __torch__.models.stereo.submodules.utils.___torch_mangle_90.channelAtt
  def forward(self: __torch__.torch.nn.modules.container.ModuleList) -> None:
    _0 = uninitialized(None)
    ops.prim.RaiseException("")
    return _0
  def __len__(self: __torch__.torch.nn.modules.container.ModuleList) -> int:
    return 3
