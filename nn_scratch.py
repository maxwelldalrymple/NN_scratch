import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('train.csv') # Read training data using pandas

data = np.array(data) # convert data to a numpy array
m, n = data.shape # save the shape of the array
np.random.shuffle(data) # shuffle before splitting into dev and training sets


data_train = data[1000:m].T # transpose training matrix
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape


def init_params(): # set inital weights and biases (random)
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2


def forward_prop(W1, b1, W2, b2, X): # Forward propagation function
    Z1 = W1.dot(X) + b1 # get the dot product of w1 * x
    A1 = ReLU(Z1) # apply Relu 
    Z2 = W2.dot(A1) + b2 # get the dot product of w2 * x
    A2 = softmax(Z2) # apply softmax 
    return Z1, A1, Z2, A2


def ReLU(Z): # Relu activation function, return x = x unless x < 0
    return np.maximum(Z, 0) # create matix of max values

def ReLU_deriv(Z): 
    return Z > 0 # return true if z > 0


def softmax(Z): # softmax normalizes the probabilities of K real numbers
    return np.exp(Z) / sum(np.exp(Z)) 

def one_hot(Y): # function to assist in calculating loss 
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1 # add a 1 to the Yth position in a vecter that is Y.size large
    one_hot_Y = one_hot_Y.T # transpose the matrix
    return one_hot_Y


def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y): # Calculate the loss  
    one_hot_Y = one_hot(Y) # create one hot matrix (eg. [0,0,0,1])
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T) 
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1) # transpose weight vector and dot it with dZ2 
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2


def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha): # update weights and biases
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2


def get_predictions(A2): # determine neural networks prediction
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y): # determine accuracy of generation
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations): # initalize neural network
    W1, b1, W2, b2 = init_params() # initalize variables
    print('inital parms: ','W1: ',W1,'b1: ', b1, 'W2: ', W2, 'b2: ', b2)
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X) # establish forward step
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y) # determine loss
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha) # update variables
        print('updated parms: ','W1: ',W1,'b1: ', b1, 'W2: ', W2, 'b2: ', b2)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2) # determine neural networks prediction
            print(get_accuracy(predictions, Y)) # return neural networks accuracy rate
    return W1, b1, W2, b2

W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 1)
