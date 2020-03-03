import numpy as np
from random import randrange

def eval_numerical_gradient(f,x,verbose=True,h =0.00001):
    """
    lidan
    a naive implementation of numerical gradient of f at x
    :param f:should be a function takes a single argument
    :param x: is the point to evaluate the gradient at
    :param verbose: verbose is the updown of description
    :param h:
    :return:
    """
    fx = f(x)
    grad = np.zeros_like(x)
    # iterate over all indexes in x
    # we know the external loop move the one dimension loop to my code not in the iterator code
    # but why it's more efficient for this
    # we know the buffer flags mode sometimes make the external_loop is more efficient in python mode
    # we know the multi_index is the index of the shape . the index has the same size to shape
    # sometimes we also add the cpython loop in the code
    it = np.nditer(x,flags=['multi_index'],op_flags=['readwrite'])
#     print(it)
    while not it.finished:
        ix = it.multi_index
#         print(ix)
#         print(type(ix))
        oldval = x[ix]
        x[ix] = oldval+h
        fxph = f(x)
        x[ix] = oldval-h
        fmph = f(x)
        x[ix] = oldval

        grad[ix] = (fxph-fmph)/(2*h)
        if verbose:
            print(ix,grad[ix])

        it.iternext()
    return grad


def eval_numerical_gradient_array(f,x,df,h= 1e-5):
    """
    :param f: function
    :param x:input
    :param df:learning rate ???
    :param h:
    :return:
    """

    grad = np.zeros_like(x)
    it = np.nditer(x,flags=['multi_index'],op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        x[ix] = oldval+h
        pos = f(x).copy()
        x[ix] = oldval-h
        neg = f(x).copy()
        x[ix] = oldval

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()

    return grad


def eval_numerical_gradient_blobs(f,input,output,h=1e-5):
    numeric_diff = []
    for input_blob in input:
        diff = np.zeros_like(input_blob.diffs)
        it = np.nditer(input_blob.vals,flags =['multi_index'],op_flags =['readwrite'])
        while not it.finished:
            idx = it.multi_index
            orig = input_blob.vals[idx]

            input_blob.vals[idx] = orig + h
            f(*(input + (output,)))
            pos = np.copy(output.vals)
            input_blob.vals[idx] = orig - h
            f(*(input + (output,)))
            neg = np.copy(output.vals)
            input_blob.vals[idx] = orig

            diff[idx] = np.sum((pos-neg)*output.diffs) / (2.0 * h)
            it.iternext()

        numeric_diff.append(diff)
    return numeric_diff

def eval_numerical_gradient_net(net,input,output,h=1e-5):
    return eval_numerical_gradient_blobs(lambda *args:net.forward(),input,output,h=h)

def grad_check_sparse(f,x,analytic_grad,num_checks=10,h=1e-5):
    for i in range(num_checks):
        ix = tuple([randrange(m) for m in x.shape])

        oldval = x[ix]
        x[ix] = oldval + h  # increment by h
        fxph = f(x)  # evaluate f(x + h)
        x[ix] = oldval - h  # increment by h
        fxmh = f(x)  # evaluate f(x - h)
        x[ix] = oldval  # reset

        grad_numerical = (fxph - fxmh) / (2 * h)
        grad_analytic = analytic_grad[ix]
        rel_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic))
        print('numerical: %f analytic: %f, relative error: %e' % (grad_numerical, grad_analytic, rel_error))


