import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import sklearn
#%%
'''Load Data, shuffle and create evaluation set '''
def decode_labels(labels):
    '''
    INPUT -> one dimensional array
    OUTPUT -> multidimensional array decoding the information contained in the input array
    '''
    num_labels = len(set(labels))
    num_elements = len(labels)
    new_labels = np.zeros([num_elements, num_labels])
    for index, label in enumerate(labels):
        new_labels[index, label] = 1
    return new_labels
    
def load_mnist():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    
    y_train = decode_labels(y_train)
    y_test = decode_labels(y_test)
    
    
    x_train, y_train = sklearn.utils.shuffle(x_train, y_train)
    x_test, y_test = sklearn.utils.shuffle(x_test, y_test)
    
    x_eval, y_eval = x_train[55000:], y_train[55000:]
    x_train, y_train = x_train[:55000], y_train[:55000]
    return x_train, y_train, x_eval, y_eval, x_test, y_test
    
x_train, y_train, x_eval, y_eval, x_test, y_test = load_mnist()

img_size_flat = 784
n_classes = 10

#%%
'''Hyperparameters:
epoch -> one forward pass and one backward pass of all the training examples.
batch size -> the number of training examples in one forward/backward pass. 
              The higher the batch size, the more memory space you'll need.
iteration -> one forward pass and one backward pass of one batch of images 
             the training examples.
'''
epochs = 10             # each epoch is 550 iterations
batch_size = 250
display_freq = 100      # Frequency of displaying the training results
learning_rate = 0.001   # The optimization initial learning rate

#%%
'''Weight and bias initializers '''
def weight_variable(shape):
    """
    INPUT -> weight shape
    OUTPUT -> weight variable
    """
    initializer = tf.truncated_normal_initializer(stddev=0.01)
    return tf.get_variable('W', dtype=tf.float32, shape=shape, initializer=initializer)


def bias_variable(shape):
    """
    INPUT -> bias shape
    OUTPUT -> weight variable
    """
    initializer = tf.constant(0., shape=shape, dtype=tf.float32)
    return tf.get_variable('b', dtype=tf.float32, initializer=initializer)
#%%
'''Linear Regression Model '''
tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='X') # (55000x784)
y = tf.placeholder(tf.float32, shape=[None, n_classes], name='Y')

# Create weight matrix initialized randomely from N~(0, 0.01) 
W = weight_variable(shape=[img_size_flat, n_classes]) # (784x10)
# Create bias vector initialized as zero 
b = bias_variable(shape=[n_classes]) # (1x10)

# calculate output values for each class (55000x10)
output_logits = tf.matmul(x, W) + b # matmul is for matrix multiplication
# Define the loss function, optimizer, and accuracy
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=y, logits=output_logits), name='loss')
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, 
                                   name='Adam-op').minimize(loss)
correct_prediction = tf.equal(tf.argmax(output_logits, 1), tf.argmax(y, 1),
                              name='correct_pred')
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

# Model predictions
cls_prediction = tf.argmax(output_logits, axis=1, name='predictions')
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

#%%
'''Plot some results'''
def plot_images(images, cls_true, cls_pred=None, title=None):
    """
    Create figure with 3x3 sub-plots.
    :param images: array of images to be plotted, (9, img_h*img_w)
    :param cls_true: corresponding true labels (9,)
    :param cls_pred: corresponding true labels (9,)
    """
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(28, 28), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            ax_title = "True: {0}".format(cls_true[i])
        else:
            ax_title = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        ax.set_title(ax_title)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    if title:
        plt.suptitle(title, size=20)
    plt.show(block=False)


def plot_example_errors(images, cls_true, cls_pred, title=None):
    """
    Function for plotting examples of images that have been mis-classified
    :param images: array of all images, (#imgs, img_h*img_w)
    :param cls_true: corresponding true labels, (#imgs,)
    :param cls_pred: corresponding predicted labels, (#imgs,)
    """
    # Negate the boolean array.
    incorrect = np.logical_not(np.equal(cls_pred, cls_true))

    # Get the images from the test-set that have been
    # incorrectly classified.
    incorrect_images = images[incorrect]

    # Get the true and predicted classes for those images.
    cls_pred = cls_pred[incorrect]
    cls_true = cls_true[incorrect]

    # Plot the first 9 images.
    plot_images(images=incorrect_images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9],
                title=title)

# Plot some of the correct and misclassified examples
cls_pred = sess.run(cls_prediction, feed_dict=feed_dict_test)
cls_true = np.argmax(y_test, axis=1)
plot_images(x_test, cls_true, cls_pred, title='Correct Examples')
plot_example_errors(x_test, cls_true, cls_pred, title='Misclassified Examples')
plt.show()


sess.close()












































