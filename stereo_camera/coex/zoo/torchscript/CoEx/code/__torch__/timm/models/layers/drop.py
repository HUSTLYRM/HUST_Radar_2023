def drop_path(x: Tensor,
    drop_prob: float=0.,
    training: bool=False) -> Tensor:
  if torch.eq(drop_prob, 0.):
    _0 = True
  else:
    _0 = torch.__not__(training)
  if _0:
    _1 = x
  else:
    keep_prob = torch.sub(1, drop_prob)
    _2 = (torch.size(x))[0]
    _3 = torch.sub(torch.dim(x), 1)
    _4 = torch.mul([1], _3)
    shape = torch.add([_2], _4)
    _5 = torch.rand(shape, dtype=ops.prim.dtype(x), layout=None, device=ops.prim.device(x), pin_memory=None)
    random_tensor = torch.add(_5, keep_prob, 1)
    _6 = torch.floor_(random_tensor)
    output = torch.mul(torch.div(x, keep_prob), random_tensor)
    _1 = output
  return _1
