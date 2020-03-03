from __future__ import absolute_import
import numpy as onp
from . import wrapper as wnp
from .numpy_box import ArrayBox
from autodiff.autodiff.tracer import primitiveFunction
from autodiff.autodiff.core import defvjp


def unbroadcast(target, g, broadcast_idx=0):
    while wnp.ndim(g) > wnp.ndim(target):
        g = wnp.sum(g, axis=broadcast_idx)
    for axis, size in enumerate(wnp.shape(target)):
        if size == 1:
            g = wnp.sum(g, axis=axis, keepdims=True)
    if wnp.iscomplexobj(g) and not wnp.iscomplex(target):
        g = wnp.real(g)
    return g

def replace_zero(x, val):
    return wnp.where(x, x, val)



# 定义每个函数的每个参数的梯度
defvjp(wnp.add,         lambda g, ans, x, y : unbroadcast(x, g),
                        lambda g, ans, x, y : unbroadcast(y, g))
defvjp(wnp.multiply,    lambda g, ans, x, y : unbroadcast(x, y * g),
                        lambda g, ans, x, y : unbroadcast(y, x * g))
defvjp(wnp.subtract,    lambda g, ans, x, y : unbroadcast(x, g),
                        lambda g, ans, x, y : unbroadcast(y, -g))
defvjp(wnp.divide,      lambda g, ans, x, y : unbroadcast(x,   g / y),
                        lambda g, ans, x, y : unbroadcast(y, - g * x / y**2))
defvjp(wnp.true_divide, lambda g, ans, x, y : unbroadcast(x,   g / y),
                        lambda g, ans, x, y : unbroadcast(y, - g * x / y**2))
# power的梯度为如下
defvjp(wnp.power,
    lambda g, ans, x, y: unbroadcast(x, g * y * x ** wnp.where(y, y - 1, 1.)),
    lambda g, ans, x, y: unbroadcast(y, g * wnp.log(replace_zero(x, 1.)) * x ** y))


defvjp(wnp.negative, lambda g, ans, x: -g)
defvjp(wnp.exp,    lambda g, ans, x: ans * g)
defvjp(wnp.log,    lambda g, ans, x: g / x)
defvjp(wnp.tanh,   lambda g, ans, x: g / wnp.cosh(x) **2)
defvjp(wnp.sinh,   lambda g, ans, x: g * wnp.cosh(x))
defvjp(wnp.cosh,   lambda g, ans, x: g * wnp.sinh(x))

# relu
defvjp(wnp.where, None,
       lambda g, ans, c, x=None, y=None: wnp.where(c, g, wnp.zeros(g.shape)),
       lambda g, ans, c, x=None, y=None: wnp.where(c, wnp.zeros(g.shape), g))

# reshape the gradient the same way
defvjp(wnp.reshape, lambda g, ans, x, shape, order=None:
       wnp.reshape(g, wnp.shape(x), order=order))

# ----- Dot grads -----

def _dot_vjp_0(g, ans, lhs, rhs):
  if max(wnp.ndim(lhs), wnp.ndim(rhs)) > 2:
    raise NotImplementedError("Current dot vjps only support ndim <= 2.")

  if wnp.ndim(lhs) == 0:
    return wnp.sum(rhs * g)
  if wnp.ndim(lhs) == 1 and wnp.ndim(rhs) == 1:
    return g * rhs
  if wnp.ndim(lhs) == 2 and wnp.ndim(rhs) == 1:
      # add a dim
    return g[:, None] * rhs
  if wnp.ndim(lhs) == 1 and wnp.ndim(rhs) == 2:
    return wnp.dot(rhs, g)
  return wnp.dot(g, rhs.T)

def _dot_vjp_1(g, ans, lhs, rhs):
  if max(wnp.ndim(lhs), wnp.ndim(rhs)) > 2:
    raise NotImplementedError("Current dot vjps only support ndim <= 2.")

  if wnp.ndim(rhs) == 0:
    return wnp.sum(lhs * g)
  if wnp.ndim(lhs) == 1 and wnp.ndim(rhs) == 1:
    return g * lhs
  if wnp.ndim(lhs) == 2 and wnp.ndim(rhs) == 1:
    return wnp.dot(g, lhs)
  if wnp.ndim(lhs) == 1 and wnp.ndim(rhs) == 2:
    return lhs[:, None] * g
  return wnp.dot(lhs.T, g)

defvjp(wnp.dot, _dot_vjp_0, _dot_vjp_1)