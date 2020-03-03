import numpy as np

class KnearestNeighbor(object):

    def __init__(self):
        pass

    def train(self,X,y):
        """
            Train the classifier. For k-nearest neighbors this is just
            memorizing the training data.

            you know the x is N*D
            the y is N*1
            Inputs:
            - X: A numpy array of shape (num_train, D) containing the training data
              consisting of num_train samples each of dimension D.
            - y: A numpy array of shape (N,) containing the training labels, where
                 y[i] is the label for X[i].
        """
        self.Xtr = X
        self.Ytr = y


    def predict(self,X,K=1,num_loop = 0):

        """
        predict labels for test data use data classifier
        :param X: A number array of shape (Ntest *D)
        :param K: the K neighbors of the predict
        :param num_loop: decided by numpy api
        :return:
        """

        if num_loop == 0:
            result = self.caculate_distance_no_loop(X)
        elif num_loop == 1 :
            result = self.caculate_distance_one_loop(X)
        elif num_loop == 2 :
            result = self.caculate_distance_two_loop(X)
        else:
            raise ValueError(" this is a mistake value of num loop ")

        return self.predict_label(result,K)



    # use one loop
    def caculate_distance_no_loop(self,X):
        # you know xtr and xte has the same shape of every line
        num_test = X.shape[0]
        num_train = self.Xtr.shape[0]

        result = np.zeros((num_train,num_test) )
        dists = np.sqrt(-2*self.Xtr.T.dot(X)+ np.sum(np.square(self.Xtr),axis=1)+np.transpose([np.sum(np.square(X),axis=1)]))

        return


    def caculate_distance_one_loop(self,X):

        num_test = X.shape[0]
        num_train = self.Xtr.shape[0]

        result = np.zeros((num_train,num_test))
        for i in range(num_test):
            result[:,i] = np.sqrt(np.sum(np.square(self.Xtr-X[i]),axis=1))
        return result

    def caculate_distance_two_loop(self,X):

        num_test = X.shape[0]
        num_train = self.Xtr.shape[0]

        result = np.zeros((num_train,num_test))

        for i in range(num_train):
            for j in range(num_test):
                result[i][j] = np.sqrt(np.sum(np.square(X[i] - self.Xtr[j])))
        return result

    def predict_label(self,Distance,K):
        num_test = Distance.shape[1]
        result = np.zeros(num_test)

        for i in range(num_test):
            temp = self.Ytr[np.argsort(Distance[:,i])[:K]]
            tc = np.argmax(np.bincount(temp))
            result[i] = tc

        return result
