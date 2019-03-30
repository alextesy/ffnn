import csv

from test import *
import numpy as np

from keras.datasets import mnist

def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    y_new = np.zeros((10, y_train.shape[0]))
    cnt = 0
    for label in y_train:
        y_new[label][cnt] = 1
        cnt += 1

    y_train = y_new

    y_new = np.zeros((10, y_test.shape[0]))
    cnt = 0
    for label in y_test:
        y_new[label][cnt] = 1
        cnt += 1

    y_test = y_new
    # normalize
    x_train = x_train.astype(float) / 255.
    x_test = x_test.astype(float) / 255.
    return (x_train, y_train), (x_test, y_test)

def _save_costs(costs, batch_norm=False):
    batch_str = '_batch_norm' if batch_norm else ''
    with open('costs{}.csv'.format(batch_str), 'w') as csvFile:
        for i in costs:
            csvFile.write(str(i)+'\n')

def train(x, y, layers_dims, learning_rate, iterations, batch_size, x_test, y_test, use_batchnorm=False):
    import time
    start = time.time()
    params, costs, training_steps, validation_accuracy, iter = L_layer_model(x, y, layers_dims, learning_rate, iterations, batch_size, use_batchnorm)
    end = time.time()
    total_time = (end - start)/60 # Minutes
    x_test = x_test.reshape((x_test.shape[0], 784))
    x = x.reshape((x.shape[0], 784))

    train_accuracy = Predict(x.T, y, params, use_batchnorm=use_batchnorm) * 100
    test_accuracy = Predict(x_test.T, y_test, params, use_batchnorm=use_batchnorm) * 100
    _save_costs(costs, use_batchnorm)
    print('The training has ended after {} minutes \n'
          'Iter: {}, Training Steps {}, Last Cost: {}'.format(total_time, iter, training_steps, costs[-1]))
    print('Validation accuracy: {}'.format(validation_accuracy))
    print('Training accuracy: {}'.format(train_accuracy))
    print('Test accuracy: {}'.format(test_accuracy))


(x_train, y_train), (x_test, y_test) = load_data()


layer_dims = [784, 20, 10]
learning_rate = 0.009
batch_size = 30
iterations = 5000
layers_dims = [784, 20, 7, 5, 10]
#train(x_train, y_train, layers_dims, learning_rate, iterations, batch_size,x_test, y_test, use_batchnorm=False)
print('#######################################')
train(x_train, y_train, layers_dims, learning_rate, iterations, batch_size,x_test, y_test, use_batchnorm=True)