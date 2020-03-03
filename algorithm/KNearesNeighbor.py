import numpy as np

class KNearestNeighbor(object):
    """a knn classifier with L2 distance """

    def __init__(self):
        pass

    def train(self,x,y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.
        :param x: A numpy array of shape (num_train, D) containing the training data
                consisting of num_train samples each of dimension D.
        :param y: A numpy array of shape (N,) containing the training labels, where
         y[i] is the label for X[i].
        :return:
        """
        self.Xtr = x
        self.Ytr = y

    def predict(self,X,k=1,num_loops=0):
        """
        predict labels for test data using the classifier
        :param X: A numpy array of shape (num_test ,D ) containing test data consisting of num_test samples each dimension D
        :param k: the number of knn. the neighborhood number of one point
        :param num_loops:Determines which implementation to use to compute distances
      between training points and testing points.
        :return:
        :result y : A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].
        """

        if num_loops == 0:
            dists = self.compute_distances_no_loop(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loop(X)
        else :
            raise ValueError("InValid value %d for num_loop "%num_loops)

        return self.predict_label(dists,k=k)

    def compute_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0]
        num_train = self.Xtr.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # Compute the l2 distance between the ith test point and the jth    #
                # training point, and store the result in dists[i, j]. You should   #
                # not use a loop over dimension.
                #
                dists[i, j] = np.linalg.norm(self.Xtr[j, :] - X[i, :])
                #####################################################################
                #####################################################################
                #                       END OF YOUR CODE                            #
                #####################################################################
        return dists

    def compute_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            #######################################################################
            # TODO:                                                               #
            # Compute the l2 distance between the ith test point and all training #
            # points, and store the result in dists[i, :].                        #
            dists[i, :] = np.linalg.norm(self.Xtr - X[i, :], axis=1)
            #######################################################################
            #######################################################################
            #                         END OF YOUR CODE                            #
            #######################################################################
        return dists

    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.Xtr.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy.                #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################
        M = np.dot(X, self.Xtr.T)
        print(M.shape)
        te = np.square(X).sum(axis=1)
        print(te.shape)
        tr = np.square(self.Xtr).sum(axis=1)
        print(tr.shape)
        dists = np.sqrt(-2 * M + tr + np.matrix(te).T)  # tr add to line, te add to row
        pass
        #########################################################################
        #                         END OF YOUR CODE                              #
        #########################################################################
        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = []
            #########################################################################
            # TODO:                                                                 #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort.                             #
            #########################################################################
            # closest_yset = np.argsort(dists[i,:]).flatten()
            # print closest_yset.shape
            # a=closest_yset.flatten()
            # print a.shape
            # closest_index = closest_yset[:k]
            # print closest_index.shape
            # closest_y = self.y_train[closest_index].flatten()
            # print closest_y.shape
            labels = self.y_train[np.argsort(dists[i, :])].flatten()
            closest_y = labels[:k]
            #########################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.
            #

            # find the max count of labels from nodes which has the least distance to this test node
            #
            # bincount is count label array
            # argmax return the max value's index
            count_num = np.bincount(closest_y)
            max_index = np.argmax(count_num)
            y_pred[i] = max_index

            return y_pred