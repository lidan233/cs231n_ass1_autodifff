from __future__ import absolute_import
import types
from autodiff.autodiff.tracer import primitiveFunction,notracePrimitiveFunction
import numpy as _np

noGradientFunction = [
    _np.ndim, _np.shape, _np.iscomplexobj, _np.result_type, _np.zeros_like,
    _np.ones_like, _np.floor, _np.ceil, _np.round, _np.rint, _np.around,
    _np.fix, _np.trunc, _np.all, _np.any, _np.argmax, _np.argmin,
    _np.argpartition, _np.argsort, _np.argwhere, _np.nonzero, _np.flatnonzero,
    _np.count_nonzero, _np.searchsorted, _np.sign, _np.ndim, _np.shape,
    _np.floor_divide, _np.logical_and, _np.logical_or, _np.logical_not,
    _np.logical_xor, _np.isfinite, _np.isinf, _np.isnan, _np.isneginf,
    _np.isposinf, _np.allclose, _np.isclose, _np.array_equal, _np.array_equiv,
    _np.greater, _np.greater_equal, _np.less, _np.less_equal, _np.equal,
    _np.not_equal, _np.iscomplexobj, _np.iscomplex, _np.size, _np.isscalar,
    _np.isreal, _np.zeros_like, _np.ones_like, _np.result_type
]

# 类型构造器__new__方法被封装
def wrap_inttype(cls):
    class IntdtypeClass(cls):
        __new__ = notracePrimitiveFunction(cls.__new__)
    return IntdtypeClass(cls)


def wrap_namespace(old,new):
    unchanged_types = {float, int, type(None), type}
    #必须对库内部的int进行wrapper
    int_types = {_np.int, _np.int8, _np.int16, _np.int32, _np.int64, _np.integer}
    # 是否是内建函数 用户定义的函数 以及用C++写的对数组的每个元素进行操作的函数
    function_types = {_np.ufunc, types.FunctionType, types.BuiltinFunctionType}

    for name,obj in old.items():
        print(name)
        if obj in noGradientFunction:
            new[name] = notracePrimitiveFunction(obj)
        elif type(obj) in function_types:
            new[name] = primitiveFunction(obj)
        elif type(obj) in type and obj in int_types:
            new[name] = wrap_inttype(obj)
        elif type(obj) in unchanged_types:
            new[name] = obj



#我们import的时候其实就是将库内函数加入globals 所以现在是执行的wrapper覆盖操作
wrap_namespace(_np.__dict__,globals())



