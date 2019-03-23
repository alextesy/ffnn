import sys

import numpy as np
import pandas as pd
from keras.datasets import mnist

EPSILON = 0.01


def initialize_parameters(layer_dims):
    return_dict = {}
    for i, (dim, ndim) in enumerate(zip(layer_dims, layer_dims[1:])):
        return_dict['W'+str(i)] = np.random.randn(ndim, dim)
        return_dict['b'+str(i)] = np.zeros((ndim, 1))
    return return_dict


'''input: an array of the dimensions of each layer in the network (layer 0 is the size of the
flattened input, layer L is the output sigmoid)
output: a dictionary containing the initialized W and b parameters of each layer
(W1…WL, b1…bL).
Hint: Use the randn and zeros functions of numpy to initialize W and b, respectively'''


def linear_forward(A, W, b):
    linear_cache = {
        'A': A,
        'W': W,
        'b': b
    }
    # transposed_W = np.transpose(W)
    Z = np.dot(W, A) + b
    return Z, linear_cache


def softmax(z):
    """Compute softmax values for each sets of scores in x."""
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(z - np.max(z))
    return e_x / e_x.sum(axis=0), z  # only difference


def relu(Z):
    return np.maximum(0, Z), Z


def linear_activation_forward(A_prev, W, B, activation):
    Z, linear_cache = linear_forward(A_prev, W, B)
    activation_func = {
        'softmax': softmax,
        'relu': relu
    }
    A, activation_cache = activation_func[activation](Z)
    cache = {'linear_cache': linear_cache,
             'activation_cache': activation_cache}

    return A, cache


def L_model_forward(X, parameters, use_batchnorm, dropout=None):
    caches = []
    L = round(len(parameters)/2)
    A = X
    for i in range(L - 1):
        A_prev = A
        W = parameters['W'+str(i)]
        b = parameters['b'+str(i)]
        A, cache = linear_activation_forward(A_prev, W, b, 'relu')
        A = dropout_func(A, dropout) if dropout else A
        A = apply_batchnorm(A) if use_batchnorm else A
        caches.append(cache)
    W = parameters['W' + str(L-1)]
    b = parameters['b' + str(L-1)]
    AL, cache = linear_activation_forward(A, W, b, 'softmax')
    AL = apply_batchnorm(AL) if use_batchnorm else AL
    caches.append(cache)
    return AL, caches

def compute_cost(AL, Y):
    m = Y.shape[0]
    cost = 0
    for i in range(m):
        cost += np.dot(Y[0], np.log(AL[:, 0]))
    return -1 / m * cost
    ''' 
    Y = Y.T
    m = AL.shape[1]
    c = Y.shape[1]
    # loss = 0
    cost = -1 / m *(np.dot(Y, np.log(AL.T)) + np.dot(1 - Y, np.log(1 - AL).T))
    #cost *= (-1 / m)
    # cost = np.squeeze(cost)'''
    return cost


def apply_batchnorm(A):
    return A / np.linalg.norm(A)


def Linear_backward(dZ, cache):
    A_prev = cache['A']
    W = cache['W']
    b = cache['b']
    m = A_prev.shape[1]
    dW = 1 / m * np.dot(dZ, np.transpose(A_prev))
    db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(np.transpose(W), dZ)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    if activation == 'softmax':
        dZ = softmax_backward(cache['activation_cache'])
        dA_prev, dW, db = Linear_backward(dZ, cache['linear_cache'])
    else:
        dZ = relu_backward(dA, cache['activation_cache'])
        dA_prev, dW, db = Linear_backward(dZ, cache['linear_cache'])

    return dA_prev, dW, db


def relu_backward(dA, activation_cache):
    Z = activation_cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    dZ[Z > 0] = 1
    return dZ

def softmax_backward(activation_cache):
    # Reshape the 1-d softmax to 2-d so that np.dot will do the matrix multiplication
    Z = activation_cache
    soft_max = softmax(Z)[0]
    return soft_max*(1 - soft_max)


'''def sigmoid_backward(dA, activation_cache):
    # np.diag(),10
    Z = activation_cache
    return dA * sigmoid(Z)[0] * (1 - sigmoid(Z)[0])'''


def L_model_backward(AL, Y, caches):
    grads = {}
    Y = Y.reshape(AL.shape)
    cache = caches[-1]
    num_of_layers = len(caches) - 1
    dA_prev, dW, db = linear_activation_backward(None, cache, 'softmax')
    grads['dA' + str(num_of_layers)] = dA_prev
    grads['dW' + str(num_of_layers)] = dW
    grads['db' + str(num_of_layers)] = db
    caches = reversed(caches[:-1])
    for i, cache in enumerate(caches):
        i = num_of_layers - i - 1
        dA_prev, dW, db = linear_activation_backward(dA_prev, cache, 'relu')
        grads['dA' + str(i)] = dA_prev
        grads['dW' + str(i)] = dW
        grads['db' + str(i)] = db
    return grads


def update_parametrs(parameters, grad, learning_rate):
    return_params = {}
    L = round(len(parameters)/2)
    for i in range(L):
        currentDW = grad['dW' + str(i)]
        currentDb = grad['db' + str(i)]
        W = parameters['W'+str(i)]
        b = parameters['b'+str(i)]
        W = W - learning_rate * currentDW
        b = b - learning_rate * currentDb
        return_params['W'+str(i)] = W
        return_params['b'+str(i)] = b
    return return_params


def L_layer_model(X, Y, layers_dims, learning_rate, num_iterations, batch_size):
    costs = []
    parmeters = initialize_parameters(layers_dims)
    prev_cost = 100
    for iter in range(num_iterations):
        X_batches = [X[:, i:i + batch_size] for i in range(0, X.shape[1], batch_size)]
        Y_batches = [Y[i:i + batch_size] for i in range(0, len(Y), batch_size)]

        for x, y in zip(X_batches, Y_batches):
            AL, caches = L_model_forward(x, parmeters, False)
            cost = compute_cost(AL, y)
            grads = L_model_backward(AL, y, caches)
            parmeters = update_parametrs(parmeters, grads, learning_rate)

        if iter % 1 == 0 and iter > 0:
            costs.append(cost)
            print(cost)
            #if prev_cost - cost <= EPSILON:
            #    break
            prev_cost = costs[-1]

    return parmeters, costs


def Predict(X, Y, parameters):
    AL, caches = L_model_forward(X, parameters, False)
    counter = 0
    for al, y in AL, Y:
        soft_al = softmax(al)
        prediction = np.argmax(soft_al)
        if prediction == y:
            counter += 1
    accuracy = counter / len(Y)
    return accuracy


def _get_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return x_train.reshape(-1, x_train.shape[0]), \
           y_train.reshape(-1, y_train.shape[0]), \
           x_test.reshape(-1, x_test.shape[0]), \
           y_test.reshape(-1, y_test.shape[0])


def pre_preprocess(y):
    y = y.tolist()
    flattened = [val for sublist in y for val in sublist]
    return pd.get_dummies(flattened).values


def dropout_func(a, keep_prob):
    d = np.random.rand(a.shape[0], a.shape[1])
    d[d < keep_prob] = 0
    d[d >= keep_prob] = 1

    a = np.multiply(a, d)
    return a


if __name__ == "__main__":
    layer_dims = [784, 20, 7, 5, 10]
    learning_rate = 0.002
    batch_size = 30
    iterations = 2000
    x_train, y_train, x_test, y_test = _get_data()
    x_train = x_train/255
    y_train = pre_preprocess(y_train)
    y_test_hot = pre_preprocess(y_test)
    # x_train = x_train[:, :10000]
    # y_train = y_train[:10000, :]

    params, costs = L_layer_model(x_train, y_train, layer_dims, learning_rate, iterations, batch_size)
    accuracy = Predict(x_test, y_test_hot, params)
