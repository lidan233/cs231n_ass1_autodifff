import numpy as np
from random import shuffle

def svm_loss_naive(W,x,y,reg):
    """
      Structured SVM loss function, naive implementation (with loops).

      Inputs have dimension D, there are C classes, and we operate on minibatches
      of N examples.

      Inputs:
      - W: A numpy array of shape (D, C) containing weights.
      - X: A numpy array of shape (N, D) containing a minibatch of data.
      - y: A numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where 0 <= c < C.
      - reg: (float) regularization strength

      Returns a tuple of:
      - loss as single float
      - gradient with respect to weights W; an array of same shape as W
      """
    dW = np.zeros(W.shape)

    num_classes = W.shape[1]
    num_train = x.shape[0]

    loss = 0.0

    for i in range(num_train):
        scores = x[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1

            if margin > 0:
                loss += margin
                dW[:,j] += x[i].T
                dW[:,y[i]] += -x[i].T

    loss /= num_train
    dW /= num_train

    loss += 0.5*reg*np.sum(W*W)

    return loss,dW



def svm_loss_vectorized(W,X,y,reg):
    loss = 0.0
    dW = np.zeros(W.shape)

    num_train = X.shape[0]
    num_classes = W.shape[1]
    scores = X.dot(W)
    correct_class_scores = scores[range(num_train), list(y)].reshape(-1, 1)  # (N, 1)
    margins = np.maximum(0, scores - correct_class_scores + 1)
    margins[range(num_train), list(y)] = 0
    loss = np.sum(margins) / num_train + 0.5 * reg * np.sum(W * W)

    coeff_mat = np.zeros((num_train, num_classes))
    coeff_mat[margins > 0] = 1
    coeff_mat[range(num_train), list(y)] = 0
    coeff_mat[range(num_train), list(y)] = -np.sum(coeff_mat, axis=1)

    dW = (X.T).dot(coeff_mat)
    dW = dW / num_train + reg * W

    return loss,dW





