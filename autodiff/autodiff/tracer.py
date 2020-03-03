
from collections import defaultdict
from contextlib import contextmanager
from .util import subvals,wraps


class Node(object):
    def __init__(self,value,fun,args,kwargs,parent_argnums,parents):
        self.parents = parents
        self.recipe = (fun,value,args,kwargs,parent_argnums)

    def initialize_root(self):
        self.parents = []
        self.recipe = (lambda x: x, None, (), {}, [])

    @classmethod
    def new_root(cls,*args,**kwargs):
        root = cls.__new__(cls)
        root.initialize_root(*args,**kwargs)
        return root

class TraceBack(object):
    def __init__(self):
        self.top = -1
    @contextmanager
    def new_trace(self):
        self.top += 1
        yield self.top
        self.top -= 1


class Box(object):
    type_mapping = {}
    types = set()

    def __init__(self,value,trace_id,node):
        self._value = value
        self._trace_id = trace_id
        self._node = node

    def __bool__(self):
        return bool(self.value)

    __nozero__ = __bool__

    def __str__(self):
        return "Autograd {0} with value {1}".format(
            type(self).__name__, str(self._value))

    @classmethod
    def register(cls,value_type):
        Box.types.add(value_type)
        Box.type_mapping[cls] = cls
        Box.type_mapping[value_type] = cls

box_type_mapping = Box.type_mapping
def new_box(value,trace,node):
    try:
        return box_type_mapping[type(value)](value,trace,node)
    except KeyError:
        raise TypeError("Can't differentiate w.r.t. type {}".format(type(value)))

box_types = Box.types

# lambda is almost 3X faster than isinstance(x, Box)
isbox  = lambda x: type(x) in box_types
getval = lambda x: getval(x._value) if isbox(x) else x

def find_top_boxed_args(args):
    # because the node is 0
    top_trace_id = -1
    top_boxes = []
    for argnum,arg in enumerate(args):
        if isbox(arg):
            if arg._trace_id > top_trace_id :
                top_boxes = [(argnum,args)]
                top_trace_id = arg._trace_id
            elif arg._trace_id == top_trace_id:
                top_boxes.append((argnum,args))
    return top_boxes,top_trace_id





def primitiveFunction(f_raw):
    @wraps(f_raw)
    def wrapper(*args,**kwargs):
        box_args,trace_id = find_top_boxed_args(args)
        if box_args:
            argvals = subvals(args,[(argum,box._value) for argum,box in box_args])
            parents = tuple(box._node for _,box in box_args)
            argnums = tuple(argum for argum,_ in box_args)
            ans = wrapper(*argvals,**kwargs)

            # 构建树操作至关重要的一步 基于输入的最高 构建了树结构 但需要输入必须是box节点 所以要对np.array之类的封装成为box
            node = Node(ans,wrapper,argvals,kwargs,argnums,parents)

            return new_box(ans,trace_id,node)
        else:
            return f_raw(*args,**kwargs)

    return wrapper






def notracePrimitiveFunction(f_raw):
    def wrapper(*args,**kwargs):
        argvals = map(getval,args)
        return f_raw(*argvals,**kwargs)
    return wrapper


trace_stack = TraceBack()


# 这个一个递归函数 trace必须自行调用trace 这样with才有意义 但是其实这并不是一个递归函数 因为traceid可以完全不变 保持0就可以
# 因为一旦改变为分级 那么计算就会变的有先有后
def trace(start_node,fun,x):
    with trace_stack.new_trace() as trace_id:
        startBox = new_box(x,trace_id,start_node)

        # 开始构建整个网络
        endBox  = fun(startBox)

        if isbox(endBox) and endBox._trace_id == startBox._trace_id:
            return endBox._value , endBox._node
        else:
            return endBox,None



