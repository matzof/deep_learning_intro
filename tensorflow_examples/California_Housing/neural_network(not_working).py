import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import sklearn
import pandas as pd
import os
#%%
'''Load data and shuffle ''' 
ROOT_PATH = "C:/dev/Tensorflow/California_Housing"
california_housing_dataframe = pd.read_csv(os.path.join(ROOT_PATH, "california_housing_train.csv"))
# get labels
dictionary = california_housing_dataframe.to_dict()
labels = dictionary.pop('median_house_value')
# get features
feature_names = list(dictionary.keys())
features = np.delete(california_housing_dataframe.values, 8, 1)
# shuffle features and labels
features, labels = sklearn.utils.shuffle(features, labels)

#%%
'''Readapt labels to fit a losgistic regression model, and create training, validation and test sets '''
def decode_labels(labels):
    '''
    INPUT -> one dimensional array
    OUTPUT -> multidimensional array decoding the information contained in the input array
    '''
    num_labels = len(set(labels))
    num_elements = len(labels)
    new_labels = np.zeros([num_elements, num_labels])
    for index, label in enumerate(labels):
        new_labels[index, label-1] = 1
    return new_labels

#labels = np.divide(np.asarray(labels), 50000).astype(int) # dirty method, but working
num_features = len(feature_names)
#labels = decode_labels(labels)

x_train, y_train = features[:12000], labels[:12000]
x_eval, y_eval = features[12000:14000], labels[12000:14000]
x_test, y_test = features[14000:], labels[14000:]




#%%
'''Hyperparameters:
epoch -> one forward pass and one backward pass of all the training examples.
batch size -> the number of training examples in one forward/backward pass. 
              The higher the batch size, the more memory space you'll need.
iteration -> one forward pass and one backward pass of one batch of images 
             the training examples.
'''
epochs = 30             # each epoch is 550 iterations
batch_size = 1000
display_freq = 500      # Frequency of displaying the training results
learning_rate = 0.00003   # The optimization initial learning rate

h1 = 1000                # number of nodes in the 1st hidden layer
h2 = 300               # number of nodes in the 2nd hidden layer
h3 = 50                # number of nodes in the 2nd hidden layer
#%%
'''Weight and bias initializers '''
def weight_variable(name, shape):
    """
    INPUT -> weight shape
    OUTPUT -> weight variable
    """
    initializer = tf.truncated_normal_initializer(stddev=0.01)
    return tf.get_variable(name, dtype=tf.float32, shape=shape, initializer=initializer)


def bias_variable(name, shape):
    """
    INPUT -> bias shape
    OUTPUT -> weight variable
    """
    initializer = tf.constant(0., shape=shape, dtype=tf.float32)
    return tf.get_variable(name, dtype=tf.float32, initializer=initializer)
#%%
'''Fully-connected layer generator'''
def fc_layer(x, num_units, name, use_relu=True):
    """
    INPUT -> input from previous layer, number of hidden units, 
             name, boolean to add ReLU non-linarity
    OUTPUT -> fully-connected layer
    """
    num_features = x.get_shape()[1]
    W = weight_variable(name + '_W', shape=[num_features, num_units])
    b = bias_variable(name +'_b', [num_units])
    layer = tf.matmul(x, W) + b
    if use_relu:
        layer = tf.nn.relu(layer)
    return layer
#%%
'''Neural Network Model '''
tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape=[None, num_features], name='X') # (55000x784)
y = tf.placeholder(tf.float32, shape=[None, ], name='Y')

# Create a fully-connected layer with h1 nodes as hidden layer
fc1 = fc_layer(x, h1, 'FC1', use_relu=True)
# Create a fully-connected layer with h1 nodes as hidden layer
fc2 = fc_layer(fc1, h2, 'FC2', use_relu=True)
# Create a fully-connected layer with h1 nodes as hidden layer
fc3 = fc_layer(fc2, h3, 'FC3', use_relu=True)
# Create a fully-connected layer with n_classes nodes as output layer
prediction = fc_layer(fc3, 1, 'OUT', use_relu=False)

# Define the loss function, optimizer, and accuracy
loss = tf.reduce_mean(tf.square(tf.subtract(y, prediction)))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss)

accuracy = tf.reduce_mean(tf.subtract(y, prediction))

#%% 
'''Train model '''
init = tf.global_variables_initializer()
# Create an interactive session (to keep the session in the other cells)
sess = tf.InteractiveSession()
# Initialize all variables
sess.run(init)
# Number of training iterations in each epoch
num_train_iter = int(len(y_train) / batch_size)
for epoch in range(epochs):
    print('Training epoch: {}'.format(epoch + 1))
    # Randomly shuffle the training data at the beginning of each epoch 
    x_train, y_train = sklearn.utils.shuffle(x_train, y_train)
    for iteration in range(num_train_iter):
        start = iteration * batch_size
        end = (iteration + 1) * batch_size
        x_batch, y_batch = x_train[start:end], y_train[start:end]

        # Run optimization op (backprop)
        feed_dict_batch = {x: x_batch, y: y_batch}
        sess.run(optimizer, feed_dict=feed_dict_batch)

        if iteration % display_freq == 0:
            # Calculate and display the batch loss and accuracy
            loss_batch, acc_batch = sess.run([loss, accuracy],
                                             feed_dict=feed_dict_batch)

            print("iter {0:3d}:\t Loss={1:.2f},\tTraining Accuracy={2:.01%}".
                  format(iteration, loss_batch, acc_batch))

    # Run validation after every epoch
    feed_dict_valid = {x: x_eval[:1000], y: y_eval[:1000]}
    loss_valid, acc_valid = sess.run([loss, accuracy], feed_dict=feed_dict_valid)
    print('---------------------------------------------------------')
    print("Epoch: {0}, validation loss: {1:.2f}, validation accuracy: {2:.01%}".
          format(epoch + 1, loss_valid, acc_valid))
    print('---------------------------------------------------------')

#%%
'''Test the network after training'''
feed_dict_test = {x: x_test, y: y_test}
loss_test, acc_test = sess.run([loss, accuracy], feed_dict=feed_dict_test)
print('---------------------------------------------------------')
print("Test loss: {0:.2f}, test accuracy: {1:.01%}".format(loss_test, acc_test))
print('---------------------------------------------------------')

error = prediction - y
error = np.absolute(error)
errora = np.sort(error)
total_error = np.sum(np.absolute(error))

sess.close()
    





















