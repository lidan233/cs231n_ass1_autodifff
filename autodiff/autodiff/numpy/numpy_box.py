from __future__ import absolute_import
import numpy as np
from autodiff.autodiff.tracer import Box,primitiveFunction
from . import wrapper as wnp

Box.__array_primority = 90.0

class ArrayBox(Box):

    # we can add property
    __slots__ = []
    __array_priority__ = 100.0

    @primitiveFunction
    def __getitem__(A, item):return A[item]

    shape = property(lambda self:self._value.shape)
    ndim = property(lambda self: self._value.ndim)
    size = property(lambda self: self._value.size)
    dtype = property(lambda self: self._value.dtype)
    T = property(lambda  self:wnp.transpose(self))

    def __len__(self): return len(self._value)
    def astype(self,*args,**kwargs): return wnp._astype(self,*args,**kwargs)

    def __neg__(self): return wnp.negative(self)

    def __add__(self, other): return wnp.add(self, other)

    def __sub__(self, other): return wnp.subtract(self, other)

    def __mul__(self, other): return wnp.multiply(self, other)

    def __pow__(self, other): return wnp.power(self, other)

    def __div__(self, other): return wnp.divide(self, other)

    def __mod__(self, other): return wnp.mod(self, other)

    def __truediv__(self, other): return wnp.true_divide(self, other)

    def __matmul__(self, other): return wnp.matmul(self, other)

    def __radd__(self, other): return wnp.add(other, self)

    def __rsub__(self, other): return wnp.subtract(other, self)

    def __rmul__(self, other): return wnp.multiply(other, self)

    def __rpow__(self, other): return wnp.power(other, self)

    def __rdiv__(self, other): return wnp.divide(other, self)

    def __rmod__(self, other): return wnp.mod(other, self)

    def __rtruediv__(self, other): return wnp.true_divide(other, self)

    def __rmatmul__(self, other): return wnp.matmul(other, self)

    def __eq__(self, other): return wnp.equal(self, other)

    def __ne__(self, other): return wnp.not_equal(self, other)

    def __gt__(self, other): return wnp.greater(self, other)

    def __ge__(self, other): return wnp.greater_equal(self, other)

    def __lt__(self, other): return wnp.less(self, other)

    def __le__(self, other): return wnp.less_equal(self, other)

    def __abs__(self): return wnp.abs(self)

    def __hash__(self): return id(self)

ArrayBox.register(np.ndarray)
for type_ in [float, np.float64, np.float32, np.float16,
              complex, np.complex64, np.complex128]:
    ArrayBox.register(type_)



# 最后还是通过运算符重载完成的类似工作
# These numpy.ndarray methods are just refs to an equivalent numpy function
#
nondiff_methods = ['all', 'any', 'argmax', 'argmin', 'argpartition',
                   'argsort', 'nonzero', 'searchsorted', 'round']
diff_methods = ['clip', 'compress', 'cumprod', 'cumsum', 'diagonal',
                'max', 'mean', 'min', 'prod', 'ptp', 'ravel', 'repeat',
                'reshape', 'squeeze', 'std', 'sum', 'swapaxes', 'take',
                'trace', 'transpose', 'var']
for method_name in nondiff_methods + diff_methods:
    setattr(ArrayBox, method_name, anp.__dict__[method_name])

# Flatten has no function, only a method.
setattr(ArrayBox, 'flatten', anp.__dict__['ravel'])

    

