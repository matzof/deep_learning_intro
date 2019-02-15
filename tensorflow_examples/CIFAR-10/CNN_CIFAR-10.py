import tensorflow as tf
import _pickle as pickle
import matplotlib.pyplot as plt
import numpy as np


def show_me(index, features, labels):
    label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    plt.imshow(features[index])
    plt.show()
    print('Label Id: {} Name: {}'.format(np.argmax(labels[index]), label_names[np.argmax(labels[index])]))


def shuffle_2_array(features, labels):
    s = np.arange(features.shape[0])
    np.random.shuffle(s)
    return features[s], labels[s]


def normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x - min_val) / (max_val - min_val)
    return x


def one_hot_encode(x):
    encoded = np.zeros((len(x), 10))
    for idx, val in enumerate(x):
        encoded[idx][val] = 1
    return encoded


def load_and_normalize_new_data():
    for batch_id in range(1, 6):
        with open('C:/dev/Tensorflow/CIFAR-10/cifar-10-batches-py' + '/data_batch_' + str(batch_id),
                  mode='rb') as file:
            batch = pickle.load(file, encoding='latin1')
        if batch_id == 1:
            features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
            labels = batch['labels']
        else:
            features = np.concatenate(
                (features, batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)), axis=0)
            labels = np.concatenate((labels, batch['labels']), axis=0)

    features, labels = shuffle_2_array(features, labels)
    features = normalize(features)
    labels = one_hot_encode(labels)
    pickle.dump((features), open('my_cifar10_features.p', 'wb'))
    pickle.dump((labels), open('my_cifar10_labels.p', 'wb'))
    return features, labels


# IMPORTING DATA
print('Importing Data...\n')
features, labels = load_and_normalize_new_data()
with open('my_cifar10_features.p', mode='rb') as file:
    features = pickle.load(file)
with open('my_cifar10_labels.p', mode='rb') as file:
    labels = pickle.load(file)

# CROSS VALIDATION
cross_validation = 1
print('Cross Validation, Validation Set = %d\nPreparing Sets...\n' % cross_validation)
train_features = np.concatenate((features[:(cross_validation - 1) * 10000], features[cross_validation * 10000:]))
train_labels = np.concatenate((labels[:(cross_validation - 1) * 10000], labels[cross_validation * 10000:]))
eval_features = features[(cross_validation - 1) * 10000: (cross_validation) * 10000:]
eval_labels = labels[(cross_validation - 1) * 10000: (cross_validation) * 10000:]
# accuracy_cv_5 = 0.7469
# accuracy_cv_4 = 0.7421
# accuracy_cv_3 = 0.7290
# accuracy_cv_2 = 0.7449
# accuracy_cv_1 = 0.7464

# PARAMETERS
width = 32
height = 32
layers = 3
class_output = 10

# HYPERPARAMETERS
epochs = 15
batch_size = 100
keep_probability = 0.5
beta = 0.0
learning_rate = 0.001
set = int(40000 / batch_size)

# INPUTS
x = tf.placeholder(tf.float32, shape=[None, width, height, layers], name='input_x')
y = tf.placeholder(tf.float32, shape=[None, class_output], name='input_y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

### BUILD MODEL ###
from numpy.random import seed

seed(2)
from tensorflow import set_random_seed

set_random_seed(3)
regularizer = tf.contrib.layers.l1_regularizer(scale=beta)

conv1 = tf.layers.conv2d(x, 64, 3, 1, padding='SAME', activation=tf.nn.relu, use_bias=True, name='conv1',
                         kernel_regularizer=regularizer)
conv1POOL = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
conv1_bn = tf.layers.batch_normalization(conv1POOL)

conv2 = tf.layers.conv2d(conv1_bn, 128, 3, 1, padding='SAME', activation=tf.nn.relu, use_bias=True, name='conv2',
                         kernel_regularizer=regularizer)
conv2POOL = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
conv2_bn = tf.layers.batch_normalization(conv2POOL)

conv3 = tf.layers.conv2d(conv2_bn, 256, 3, 1, padding='SAME', activation=tf.nn.relu, use_bias=True, name='conv3',
                         kernel_regularizer=regularizer)
conv3POOL = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
conv3_bn = tf.layers.batch_normalization(conv3POOL)

# conv4 = tf.layers.conv2d(conv3_bn, 521, 3, 1, padding='SAME', activation=tf.nn.relu, use_bias=True, name='conv4', kernel_regularizer=regularizer)
# conv4POOL = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# conv4_bn = tf.layers.batch_normalization(conv4POOL)

flat = tf.contrib.layers.flatten(conv3_bn)

full1 = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=128, activation_fn=tf.nn.relu,
                                          weights_regularizer=regularizer)
full1DROP = tf.nn.dropout(full1, keep_prob)
full1_bn = tf.layers.batch_normalization(full1DROP)

full2 = tf.contrib.layers.fully_connected(inputs=full1_bn, num_outputs=256, activation_fn=tf.nn.relu,
                                          weights_regularizer=regularizer)
full2DROP = tf.nn.dropout(full2, keep_prob)
full2_bn = tf.layers.batch_normalization(full2DROP)

full3 = tf.contrib.layers.fully_connected(inputs=full2_bn, num_outputs=512, activation_fn=tf.nn.relu,
                                          weights_regularizer=regularizer)
full3DROP = tf.nn.dropout(full3, keep_prob)
full3_bn = tf.layers.batch_normalization(full3DROP)

full4 = tf.contrib.layers.fully_connected(inputs=full3_bn, num_outputs=1024, activation_fn=tf.nn.relu,
                                          weights_regularizer=regularizer)
full4DROP = tf.nn.dropout(full4, keep_prob)
full4_bn = tf.layers.batch_normalization(full4DROP)

full5 = tf.contrib.layers.fully_connected(inputs=full4_bn, num_outputs=10, activation_fn=None,
                                          weights_regularizer=regularizer)

pred = tf.nn.softmax(full5)

# Define LOSS function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=full5))

# Define Optimizer
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# Define Prediction
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

# Define Accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

### TRAINING SESSION ###
#tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
with tf.Session() as sess:
    acc_list = []
    loss_list = []
    val_list = []
    print('Training...\n')
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        print('Epoch ' + str(epoch + 1) + '\n')
        for index in range(set):
            train_step.run(feed_dict={x: train_features[index * batch_size: (index + 1) * batch_size],
                                      y: train_labels[index * batch_size: (index + 1) * batch_size],
                                      keep_prob: keep_probability})
            if index % 50 == 0:
                train_loss = cross_entropy.eval(
                    feed_dict={x: train_features[index * batch_size: (index + 1) * batch_size],
                               y: train_labels[index * batch_size: (index + 1) * batch_size],
                               keep_prob: 1.0})
                train_accuracy = accuracy.eval(
                    feed_dict={x: train_features[index * batch_size: (index + 1) * batch_size],
                               y: train_labels[index * batch_size: (index + 1) * batch_size],
                               keep_prob: 1.0})
                print("\tstep %d, training accuracy %g, training loss %g" % (
                    index + (epoch) * set, float(train_accuracy), float(train_loss)))
                acc_list.append(float(train_accuracy))
                loss_list.append(float(train_loss))

        ### EVALUATION SESSION ###
        validation_accuracy = 0.0
        for e in range(200):
            validation_accuracy += accuracy.eval(
                feed_dict={x: eval_features[e * 50: (e + 1) * 50], y: eval_labels[e * 50: (e + 1) * 50],
                           keep_prob: 1.0})
        val_list.append(validation_accuracy / 200)
        print("\n\tvalidation accuracy %g\n" % (validation_accuracy / 200))

        plt.plot(acc_list)
        plt.plot(loss_list)
        plt.show()

    plt.plot(val_list)
    plt.show()

### TESTING SESSION ###
# print('Testing...\n')
# test_features, test_labels = pickle.load(open('preprocess_validation.p', mode='rb'))
# testing_accuracy = 0.0
# for e in range(200):
#    testing_accuracy += accuracy.eval(
#        feed_dict={x: test_features[e * 50: (e + 1) * 50], y: test_labels[e * 50: (e + 1) * 50], keep_prob: 1.0})
# print("testing accuracy %g\n" % (testing_accuracy / 200))
