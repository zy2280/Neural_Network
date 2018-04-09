import numpy as np
import pickle
import copy
% matplotlib
inline
import matplotlib.pyplot as plt


# Do not import other packages

class Loader:
    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def load_train_data(self):
        '''
        loads training data: 50,000 examples with 3072 features
        '''
        X_train = None
        Y_train = None
        for i in range(1, 6):
            pickleFile = self.unpickle('cifar-10-batches-py/data_batch_{}'.format(i))
            dataX = pickleFile[b'data']
            dataY = pickleFile[b'labels']
            if type(X_train) is np.ndarray:
                X_train = np.concatenate((X_train, dataX))
                Y_train = np.concatenate((Y_train, dataY))
            else:
                X_train = dataX
                Y_train = dataY

        Y_train = Y_train.reshape(Y_train.shape[0], 1)

        return X_train.T, Y_train.T

    def load_test_data(self):
        '''
        loads testing data: 10,000 examples with 3072 features
        '''
        X_test = None
        Y_test = None
        pickleFile = self.unpickle('cifar-10-batches-py/test_batch')
        dataX = pickleFile[b'data']
        dataY = pickleFile[b'labels']
        if type(X_test) is np.ndarray:
            X_test = np.concatenate((X_test, dataX))
            Y_test = np.concatenate((Y_test, dataY))
        else:
            X_test = np.array(dataX)
            Y_test = np.array(dataY)

        Y_test = Y_test.reshape(Y_test.shape[0], 1)

        return X_test.T, Y_test.T


X_train, Y_train = Loader().load_train_data()
X_test, Y_test = Loader().load_test_data()

print("X_Train: {} -> {} examples, {} features".format(X_train.shape, X_train.shape[1], X_train.shape[0]))
print("Y_Train: {} -> {} examples, {} features".format(Y_train.shape, Y_train.shape[1], Y_train.shape[0]))
print("X_Test: {} -> {} examples, {} features".format(X_test.shape, X_test.shape[1], X_test.shape[0]))
print("Y_Test: {} -> {} examples, {} features".format(Y_test.shape, Y_test.shape[1], Y_test.shape[0]))


class FullyConnectedNetwork(object):
    """
    Abstraction of a Fully Connected Network.
    Stores parameters, activations, cached values. 
    You can add more functions in this class, and also modify inputs and outputs of each function 
    """

    def __init__(self, layer_dim):
        """
        layer_dim: List containing layer dimensions. 
        Code:
        Initialize weight and biases for each layer
        """
        self.loss_track = []

        self.W = []
        self.b = []
        for l in range(len(layer_dim) - 1):
            self.W.append(np.random.randn(layer_dim[l], layer_dim[l + 1]) * 0.001)
            self.b.append(np.random.randn(layer_dim[l + 1], 1))

        self.best_w = None
        self.best_b = None
        self.accuracy = 0

    def affineForward(self, A, W, b):
        affine_product = np.dot(A.T, W) + b.T
        return affine_product, (W, b)

    def relu_forward(self, X):
        X[X <= 0] = 0
        return X

    def feedforward(self, X, layer_dim):

        net_output = [X.T]
        output = []
        net_cache = []
        for i in range(len(layer_dim) - 2):
            net_i, net_cache_i = self.affineForward(net_output[i].T, self.W[i], self.b[i])
            output_i = self.relu_forward(net_i)

            net_output.append(net_i)
            output.append(output_i)
            net_cache.append(net_cache_i)

        # last layer
        last_net, last_net_cache = self.affineForward(output[-1].T, self.W[-1], self.b[-1])
        net_cache.append(last_net_cache)
        output_cache = (net_output, output, net_cache)

        return last_net, output_cache

    def softmax(self, col):
        column = col.copy()
        minimum = np.min(column)
        maximum = np.max(column)

        colsum = np.sum(np.exp(column))
        column = np.exp(column) / colsum
        return column

    def loss_one_sample(self, col, y):

        gradient = [0] * len(col)
        target = int(y)
        gradient[target] = 1
        loss = - np.log(col[target])
        gradient = col - gradient
        return loss, gradient

    def loss_function(self, At, Y, lambd):
        """
        At is the output of the last layer, returned by feedforward.
        Y contains true labels for this batch.
        this function takes softmax the last layer's output and calculates loss.
        the gradient of loss with respect to the activations of the last layer are also returned by this function.
        """
        # softmax: e^a(i) / (sum(e^a[j]) for j in all classes)
        # cross entropy loss:  -log(true class's softmax value(prediction))

        # for part2: when lambd > 0, you need to change the definition of loss accordingly

        At_softmax = np.apply_along_axis(self.softmax, 1, At)  # matrix of softmax(At)
        Loss = 0
        count = 0
        for i in range(At_softmax.shape[0]):
            loss, gradient = self.loss_one_sample(At_softmax[i], Y[0][i])
            gradient = gradient.reshape((1, At.shape[1]))
            if count == 0:
                Gradient = gradient
            else:
                Gradient = np.concatenate((Gradient, gradient), axis=0)
            count += 1
            Loss += loss

        Gradient = Gradient / count
        # Reg_loss = 0.5 * lambd * (np.sum(self.W1 * self.W1) + 0.5 * np.sum(self.W2 * self.W2))
        total_loss = Loss / count

        return total_loss, Gradient

    def affineBackward(self, dA_prev, cache):
        """
        Expected Functionality:
        Backward pass for the affine layer.
        dA_prev: gradient from the next layer.
        cache: cache returned in affineForward
        :returns dA: gradient on the input to this layer
                 dW: gradient on the weights
                 db: gradient on the bias
        """
        A = cache[0]
        W = cache[1]
        dW = np.dot(A.T, dA_prev)
        db = np.sum(dA_prev, axis=0).reshape(1, dA_prev.shape[1])
        dA = np.dot(dA_prev, W.T)

        return dA, dW, db

    def relu_backward(self, dx, cached_x):
        """
        Expected Functionality:
        backward pass for relu activation
        """
        relu = dx.copy()
        for i in range(relu.shape[0]):
            for j in range(relu.shape[1]):
                if cached_x[i, j] < 0:
                    relu[i, j] = 0
                else:
                    relu[i, j] = 1 * relu[i, j]

        return relu

    def backprop(self, loss, cache, dAct, layer_dim):
        """
        Expected Functionality: 
        returns gradients for all parameters in the network.
        dAct is the gradient of loss with respect to the output of final layer of the network.
        """

        dA = [dAct]
        dA2 = []
        dw = []
        db = []

        i = len(self.W) - 1
        j = 0
        while i > 0:
            dAi, dwi, dbi = self.affineBackward(dA[j], (cache[1][i - 1], self.W[i]))
            dAi2 = self.relu_backward(dAi, cache[0][i])
            dA.append(dAi)
            dA2.append(dAi2)
            dw.append(dwi)
            db.append(dbi)
            i -= 1
            j += 1

        dA1, dW1, db1 = self.affineBackward(dA2[-1], (cache[0][0], self.W[0]))

        dA.append(dA1)
        dw.append(dW1)
        db.append(db1)

        dw = dw[::-1]
        db = db[::-1]
        return dw, db

    def updateParameters(self, gradients, learning_rate, lambd):
        """
        Expected Functionality:
        use gradients returned by backprop to update the parameters.
        """
        # dW1 = gradients[0]
        # dW1 += lambd * self.W1
        # dW2 = gradients[1]
        # dW2 += lambd * self.W2
        # dW3 = gradients[2]
        # dW3 += lambd * self.W3

        # self.W1 -= learning_rate * dW1
        # self.W2 -= learning_rate * dW2
        # self.W3 -= learning_rate * dW3
        # self.b1 -= learning_rate * gradients[3].T
        # self.b2 -= learning_rate * gradients[4].T
        # self.b3 -= learning_rate * gradients[5].T

        dW = gradients[0]
        db = gradients[1]

        for i in range(len(self.W)):
            self.W[i] -= learning_rate * dW[i]
            self.b[i] -= learning_rate * db[i].T

    def training_one_batchtrain(self, X, Y, layer_dim, learning_rate, lambd, batch_size):
        forward_output, output_cache = self.feedforward(X, layer_dim)
        loss = self.loss_function(forward_output, Y, lambd)[0]
        self.loss_track.append(loss)
        dAct = self.loss_function(forward_output, Y, lambd)[1]

        dW, db = self.backprop(loss, output_cache, dAct, layer_dim)
        self.updateParameters((dW, db), learning_rate, lambd)

    def get_batch(self, X, Y, batch_size):
        """
        Expected Functionality: 
        given the full training data (X, Y), return batches for each iteration of forward and backward prop.
        """

        indices = np.random.randint(0, X.shape[1], batch_size)
        x = np.array([X[:, j] for j in indices]).T
        y = np.array([Y[:, j] for j in indices]).T

        return [x, y]

    def evaluate(self, X_test, Y_test, layer_dim):
        '''
        X: X_test (3472 dimensions, 10000 examples)
        Y: Y_test (1 dimension, 10000 examples)

        Expected Functionality: 
        print accuracy on test set
        '''
        forward = self.feedforward(X_test, layer_dim)[0]
        probability = np.apply_along_axis(self.softmax, 1, forward)
        self.prob = probability

        correct = 0
        for i in range(probability.shape[0]):
            if int(np.argmax(probability[i])) == int(Y_test[:, i][0]):
                correct += 1
        # print("correct: " + str(correct))
        return correct / probability.shape[0]

    def train(self, X, Y, layer_dim, max_iters=5000, batch_size=100, learning_rate=0.01, lambd=0, validate_every=200):
        """
        X: (3072 dimensions, 50000 examples) (Cifar train data)
        Y: (1 dimension, 50000 examples)
        lambd: the hyperparameter corresponding to L2 regularization

        Divide X, Y into train(80%) and val(20%), during training do evaluation on val set
        after every validate_every iterations and in the end use the parameters corresponding to the best
        val set to test on the Cifar test set. Print the accuracy that is calculated on the val set during 
        training. Also print the final test accuracy. Ensure that these printed values can be seen in the .ipynb file you
        submit.

        Expected Functionality: 
        This function will call functions feedforward, backprop and update_params. 
        Also, evaluate on the validation set for tuning the hyperparameters.
        """

        # Divide into train(80%) and val(20%)
        train_index = np.random.randint(0, X.shape[1], int(round(0.8 * X.shape[1], 0)))
        val_index = np.array(list(set(list(range(X.shape[1]))) - set(train_index)))

        train_x = np.array([X[:, j] for j in train_index]).T
        train_y = np.array([Y[:, j] for j in train_index]).T
        val_x = np.array([X[:, j] for j in val_index]).T
        val_y = np.array([Y[:, j] for j in val_index]).T

        iteration_count = 0
        while iteration_count < max_iters:

            batches = self.get_batch(train_x, train_y, batch_size)
            x_batches = batches[0]
            y_batches = batches[1]

            train_x_b = x_batches.copy()
            train_y_b = y_batches.copy().reshape(1, batch_size)

            self.training_one_batchtrain(train_x_b, train_y_b, layer_dim, learning_rate, lambd, batch_size)
            iteration_count += 1

            # for i in range(num_batch):
            if iteration_count % validate_every == 0:
                # print(iteration_count)
                # print("actual validatation y is ", str(val_y))
                accuracy = self.evaluate(val_x, val_y, layer_dim)

                if accuracy > self.accuracy:
                    self.accuracy = accuracy
                    self.best_w = self.W
                    self.best_b = self.b
                # print(np.argmax(self.prob, axis = 1))

                print("accuracy is ", str(accuracy))
                # print('------------------------------------------------------')



layer_dimensions = [3072,193,55,10]  # including the input and output layers  #193,60
# 3072 is the input feature size, 10 is the number of outputs in the final layer
FCN = FullyConnectedNetwork(layer_dimensions)
FCN.train(X_train, Y_train,layer_dimensions, max_iters= 8000, batch_size=128, learning_rate=0.01, lambd=0, validate_every=200)
# lambd, the L2 regularization penalty hyperparamter will be 0 for this part
y_predicted = FCN.evaluate(X_test, Y_test)  # print accuracy on test set
print(y_predicted)