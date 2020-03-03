import numpy as np
from .core import make_vjp
from .util import subvals

def grad(fun,argums):
    def gradfun(*args,**kwargs):
        unary_fun = lambda x:fun(*subvals(args,argums,x),**kwargs)
        vjp,ans = make_vjp(unary_fun,args[argums])
        return vjp(np.ones_like(ans))
    return gradfun


