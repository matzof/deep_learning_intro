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

labels = np.divide(np.asarray(labels), 500000)
labels = np.reshape(np.asarray(labels), (len(labels), 1))
num_features = len(feature_names)
#labels = decode_labels(labels)

x_train, y_train = features[:12000], labels[:12000]
x_eval, y_eval = features[12000:14000], labels[12000:14000]
x_test, y_test = features[14000:], labels[14000:]

#%%
learning_rate = 0.00001
training_epochs = 1000
batch_size = 500
stall_check = 10 # number of epochs after which there will be a stall check
num_train_iter = int(len(y_train) / batch_size)
cost_history = np.empty(shape=[1],dtype=float)

h1 = 64                # number of nodes in the 1st hidden layer
h2 = 64
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

'''Fully-connected layer generator'''
def fc_layer(x, num_units, name, activation=None):
    """
    INPUT -> input from previous layer, number of hidden units, 
             name, boolean to add ReLU non-linarity
    OUTPUT -> fully-connected layer
    """
    num_features = x.get_shape()[1]
    W = weight_variable(name + '_W', shape=[num_features, num_units])
    b = bias_variable(name +'_b', [num_units])
    layer = tf.matmul(x, W) + b
    if activation == 'ReLU':
        layer = tf.nn.relu(layer)
    if activation == 'tanh':
        layer = tf.nn.tanh(layer)
    return layer
#%%
tf.reset_default_graph()
x = tf.placeholder(tf.float32,[None,num_features])
y = tf.placeholder(tf.float32,[None, 1])


# Create a fully-connected layer with h1 nodes as hidden layer
fc1 = fc_layer(x, h1, 'FC1', activation='ReLU')
# Create a fully-connected layer with h1 nodes as hidden layer
fc2 = fc_layer(fc1, h2, 'FC2', activation='ReLU')
# Create a fully-connected layer with n_classes nodes as output layer
prediction = fc_layer(fc2, 1, 'OUT', activation='tanh')

cost = tf.losses.mean_squared_error(y, prediction)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


#%%
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
previous_loss = 9999999999999
for epoch in range(training_epochs):
    print('Training epoch: {}'.format(epoch + 1)) 
    x_train, y_train = sklearn.utils.shuffle(x_train, y_train)
    for iteration in range(num_train_iter):
        start = iteration * batch_size
        end = (iteration + 1) * batch_size
        x_batch, y_batch = x_train[start:end], y_train[start:end]

        # Run optimization op (backprop)
        feed_dict_batch = {x: x_batch, y: y_batch}
        sess.run(optimizer, feed_dict=feed_dict_batch)
#        # Calculate and display the batch loss and accuracy
#        loss_batch = sess.run(cost, feed_dict=feed_dict_batch)
#
#        print("iter {0:3d}:\t Loss={1:.2f}".
#              format(iteration, loss_batch))

    # Run validation after every epoch
    feed_dict_valid = {x: x_eval[:1000], y: y_eval[:1000]}
    loss_valid = sess.run(cost, feed_dict=feed_dict_valid)
    print('---------------------------------------------------------')
    print("Epoch: {0}, validation loss: {1:.6f}".
          format(epoch + 1, loss_valid))
    print('---------------------------------------------------------')
    
#    sess.run(optimizer,feed_dict={x:x_train,y:y_train})
    cost_history = np.append(cost_history,sess.run(cost,feed_dict={x: x_train,y: y_train}))
    # check for a stall
    if epoch % stall_check == 0:
        if loss_valid >= previous_loss:
            break
        previous_loss = loss_valid
    
    
#%%
plt.plot(range(len(cost_history[1:])),cost_history[1:])
plt.show()
#%%
pred = sess.run(prediction, feed_dict={x: x_test})
error = np.abs(np.subtract(pred, np.reshape(np.asarray(y_test), pred.shape)))
mean_error = np.multiply(np.mean(error), 500000)
mse = tf.reduce_mean(tf.square(pred - y_test))

print("MSE: %.4f" % sess.run(mse)) 

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y_test, pred)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()


sess.close()





































