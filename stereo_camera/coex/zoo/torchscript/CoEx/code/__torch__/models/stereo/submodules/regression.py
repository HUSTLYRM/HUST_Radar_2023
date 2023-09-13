class Regression(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  D : int
  def forward(self: __torch__.models.stereo.submodules.regression.Regression,
    cost: Tensor,
    spg: Tensor) -> List[Tensor]:
    _0 = __torch__.torch.nn.functional.softmax
    _1 = __torch__.models.stereo.submodules.spixel_utils.spixel.upfeat
    b, _2, h, w, = torch.size(spg)
    corr, disp, = (self).topkpool(cost, )
    corr0 = _0(corr, 2, 3, None, )
    disp_4 = torch.sum(torch.mul(corr0, disp), [2], True, dtype=None)
    _3 = [b, torch.floordiv(h, 4), torch.floordiv(w, 4), 1]
    disp_40 = torch.permute(torch.reshape(disp_4, _3), [0, 3, 1, 2])
    disp_1 = _1(disp_40, spg, )
    disp_10 = torch.mul(torch.squeeze(disp_1, 1), 4)
    return [disp_10]
  def topkpool(self: __torch__.models.stereo.submodules.regression.Regression,
    cost: Tensor) -> Tuple[Tensor, Tensor]:
    _4 = torch.arange(self.D, dtype=None, layout=None, device=ops.prim.device(cost), pin_memory=None)
    ind_ = torch.reshape(_4, [1, 1, -1, 1, 1])
    _5 = [1, 1, 1, (torch.size(cost))[-2], (torch.size(cost))[-1]]
    ind = torch.repeat(ind_, _5)
    _6 = torch.slice(torch.argsort(cost, 2, True), 0, 0, 9223372036854775807, 1)
    _7 = torch.slice(_6, 1, 0, 9223372036854775807, 1)
    pool_ind = torch.slice(_7, 2, 0, 2, 1)
    cv = torch.gather(cost, 2, pool_ind, sparse_grad=False)
    _8 = [(torch.size(cv))[0], (torch.size(cv))[1], 1, 1, 1]
    disp = torch.gather(torch.repeat(ind, _8), 2, pool_ind, sparse_grad=False)
    return (cv, disp)
