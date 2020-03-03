def subvals(x,all):
    x_ = list(x)
    for i,v in all:
        x_[i] = v
    return tuple(x_)


def subvals(x,i,v):
    x_ = list(x)
    x_[i] = v
    return tuple(x_)

def topoSort(endnode):
    childCount = {}
    stack = [endnode]


    while stack:
        node = stack.pop()
        if node in childCount:
            childCount[node]+=1
        else :
            childCount[node] = 1
            stack.extend(node.parents)


    childless = [endnode]
    while childless:
        node = childless.pop()
        yield  node
        for parent in node.parents:
            if childCount[parent] == 1:
                childless.append(parent)
            else:
                childCount[parent] -= 1


def wraps(fun,namestr="{fun}",docstr="{doc}",**kwargs):
    def _warps(f):
        try:
            f.__name__ = namestr.format(fun=get_name(fun),**kwargs)
            f.__doc__ = docstr.format(fun=get_name(fun),doc=get_doc(fun),**kwargs)
        finally:
            return f
    return _warps

def wrap_nary_f(fun,op,argnum):
    namestr =  "{op}_of_{fun}_wrt_argnum_{argnum}"
    docstr = """\
        {op} of function {fun} with respect to argument number {argnum}. Takes the
        same arguments as {fun} but returns the {op}.
        """
    return wraps(fun, namestr, docstr, op=get_name(op), argnum=argnum)

get_name = lambda f:getattr(f,'__name__','[unknown name]')
get_doc = lambda  f:getattr(f,'__doc__','')

