import numpy as np
import matplotlib.pyplot as plt

class TwoLayerNet(object):
    def __init__(self,input_size,hidden_size ,output_size,std=1e-4):
        self.param = {}

        # this is just a random intializer
        self.param['W1'] = std * np.random.randn(input_size,hidden_size)
        self.param['b1'] = np.zeros(hidden_size)

        self.param['W2'] = std * np.random.randn(hidden_size,output_size)
        self.param['b2'] = std * np.random.randn()

        # this is a xvaier intializer
        # self.param['W1'] = std * np.random.normal(0,2/(input_size+hidden_size),size=(input_size,hidden_size))
        # self.param['b1'] = np.zeros(hidden_size)
        # self.param['W2'] = std * np.random.normal(0,2/(hidden_size+output_size),size=(input_size,hidden_size))
        # self.param['b2'] = np.zeros(hidden_size)

    def loss(self,X,y=None,reg=0.0 ):
        W1,b1 = self.param['W1'],self.param['b1']
        W2,b2 = self.param['W2'],self.param['b2']

        N,D = X.shape

        score = None

        # i think relu is very easy to intialize
        h1_out = np.maximum(0,X.dot(W1)+b1)
        score = h1_out.dot(W2) + b2

        if y is None :
            return score

        losst = None

        shift_scores = score - np.max(score,axis=1).reshape(-1,1)
        softmax_output = np.exp(shift_scores) / np.sum(np.exp(shift_scores),axis=1).reshape(-1,1)

        # this is one kind of softmax
        losst = -np.sum(np.log(softmax_output[range(N),list(y)]))

        # # this is another softmax but need input
        # for i in range(N):
        #     losst += -np.log(softmax_output[i,y[i]])
        #

        losst /= N
        losst += 0.5 * reg * (np.sum(W1*W1) +np.sum(W2*W2) )

        grads ={}

        dscores = softmax_output.copy()

        dscores[range(N),list(y)] += -1
        dscores /= N

        # you know the h1_out * W2 =
        # we don't use softmax layer we just use softmax loss
        grads['W2'] = h1_out.T.dot(dscores) + reg*W2
        grads['b2'] = np.sum(dscores, axis=0)


        # you know the in's grad is w
        dh = dscores.dot(W2.T)
        dh_ReLu = (h1_out > 0) * dh

        grads['W1'] = X.T.dot(dh_ReLu) + reg * W1
        grads['b1'] = np.sum(dh_ReLu, axis=0)

        return losst ,grads



    def train(self,X,y,xval,yval,learning_rate=1e-3,learning_rate_decay=0.95
              ,reg=1e-5,num_iters=100,batch_size=200,verbose=False):
        num_train = X.shape[0]

        # maybe cover all the num_train
        iterations_per_epoch = max(num_train / batch_size, 1)

        loss_history = []
        train_acc_history = []
        val_acc_history = []


        for it in range(num_iters):
            X_batch = None
            Y_batch = None

            idx = np.random.choice(num_train,batch_size,replace=True)
            X_batch = X[idx]
            Y_batch = y[idx]

            loss,grads = self.loss(X_batch,Y_batch,reg=reg)
            loss_history.append(loss)

            self.param['W1'] += -learning_rate*grads['W1']
            self.param['b1'] += -learning_rate*grads['b1']
            self.param['W2'] += -learning_rate*grads['W2']
            self.param['b2'] += -learning_rate*grads['b2']

            if verbose and it%100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == Y_batch).mean()
                val_acc = (self.predict(xval) == yval).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self,X):
        h1 = np.maximum(0,X.dot(self.param['W1']) + self.param['b1'])
        output = h1.dot(self.param['W2']) + self.param['b2']

        scores = output - np.max(output,axis=1).reshape(-1,1)
        # softmax_output = np.exp(scores) / np.sum(np.exp(scores), axis=1).reshape(-1, 1)
        #         # return np.argmax(softmax_output,axis=1).reshape(-1,1)
        return  np.argmax(scores,axis=1).reshape(-1,1)

