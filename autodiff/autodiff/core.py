from collections import defaultdict
from itertools import count
import numpy as np

from .tracer import Node,trace
from .util import topoSort


primitive_vjps =defaultdict()

def add_autograds(prev_g,g):
    if prev_g is None:
        return g
    return prev_g + g

def defvjp(fun,*args,**kwargs):
    argnums = kwargs.get('argnums',count())
    for argnum ,vjp in zip(argnums,args):
        primitive_vjps[fun][argnum] = vjp

def backword_pass(g,end_node):
    outgrads = {end_node:g}
    for node in topoSort(end_node):
        outgrad = outgrads.pop(node)
        fun,value,args,kwargs,argnums = node.recipe
        for argnum,parent in zip(argnums,node.parents):
            vjp = primitive_vjps[fun][argnum]
            parent_grid = vjp(outgrad,value,*args,**kwargs)
            outgrads[parent] = add_autograds(outgrads.get(parent),parent_grid)
    return outgrad


# vector jorban product 在这个过程中传入的fun必须是能够调用trace的fun
def make_vjp(fun,x):
    startNode = Node.new_root()
    end_value,end_node = trace(startNode,fun,x)
    if end_node is None:
        def vjp(g): return np.zeros_like(x)
    else:
        def vjp(g): return backword_pass(end_value,end_node)

    return vjp,end_value