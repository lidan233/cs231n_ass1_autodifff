import  numpy as np
from random import shuffle

def softmax_loss_naive(W,X,y,reg):
    """
    softmax naive function ,naive implements with loops
    :param W: A numpy array of shape (D, C) containing weights.
    :param X:A numpy array of shape (N, D) containing a minibatch of data.
    :param y:A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
    :param reg:(float) regularization strength
    :return:
            loss : single float for loss
            gradient:  with respect to weights W; an array of same shape as W
    """

    #initialize the return result
    loss = 0.0
    gradient = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!   you know the exp is so big !!!!                                                        #
    #############################################################################

    num_classes = W.shape[1]
    num_train = X.shape[0]

    for i in range(num_train):
        scores = X[i].dot(W)
        shift_scores = scores - max(scores)

        # caculate loss
        lossi = - shift_scores[y[i]] + np.log(sum(np.exp(shift_scores)))
        loss += lossi

        # caculate gradient

        for  j in range(num_classes):
            t_score = np.exp(shift_scores[j])/np.sum(np.exp(shift_scores))

            if j== y[i]:
                gradient[:,j] += (t_score-1)*X[i]
            else:
                gradient[:,j] += t_score * X[i]



    loss /= num_train
    loss += 0.5*reg*np.sum(W*W)

    gradient = gradient/num_train + reg*W

    return loss , gradient


#??? about shape
def softmax_loss_vectorized(W,x,y,reg):
    """

    :param W:  size is (D*C)
    :param x:  size is (N*D)
    :param y:  size is (N)
    :param reg: this is a float
    :return:
    """

    loss = 0

    #  gradient is D *C
    gradient = np.zeros_like(W)

    num_train = x.shape[0]
    num_class = W.shape[1]

    scores = x.dot(W)
    shift_scores = scores - np.max(scores,axis=1).reshape(-1,1)

    t_score = np.exp(shift_scores) / np.sum(np.exp(shift_scores),axis=1).reshape(-1,1)

    # loss = np.sum(-shift_score[range(num_train),list(y)].reshape(-1,1)+np.log(sum(np.exp(shift_scores),axis=1))
    loss = -np.sum(np.log(t_score[range(num_train), list(y)]))
    loss /= num_train
    loss += 0.5* reg*np.sum(W*W)

    gradient = t_score.copy()
    gradient[range(num_train),list(y)] += -1



    # x.T is D*N  gradient is N * C
    gradient = x.T.dot(gradient)
    gradient = gradient/num_train + reg*W


    return loss,gradient
