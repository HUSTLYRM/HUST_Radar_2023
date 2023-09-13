def upfeat(input: Tensor,
    prob: Tensor) -> Tensor:
  _0 = __torch__.torch.nn.functional.___torch_mangle_156.interpolate
  b, c, h, w, = torch.size(input)
  _1 = __torch__.torch.nn.functional.unfold(input, [3, 3], [1, 1], [1, 1], [1, 1], )
  feat = torch.reshape(_1, [b, -1, h, w])
  _2 = _0(feat, None, 4., "nearest", None, None, )
  _3 = [b, -1, 9, torch.mul(h, 4), torch.mul(w, 4)]
  feat0 = torch.reshape(_2, _3)
  _4 = torch.mul(feat0, torch.unsqueeze(prob, 1))
  return torch.sum(_4, [2], False, dtype=None)
