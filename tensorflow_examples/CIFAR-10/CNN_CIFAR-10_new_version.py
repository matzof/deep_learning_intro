import tensorflow as tf
import _pickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.utils import shuffle
#%%
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

def load_and_normalize_new_data(path):
    '''Load CIFAR-10 Dataset from a folder in the path specified, 
    then shuffle it, normalize features and encode labels'''
    for batch_id in range(1, 6):
        with open(path + '/data_batch_' + str(batch_id),
                  mode='rb') as file:
            batch = pickle.load(file, encoding='latin1')
        if batch_id == 1:
            features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
            labels = batch['labels']
        else:
            features = np.concatenate(
                (features, batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)), axis=0)
            labels = np.concatenate((labels, batch['labels']), axis=0)
    features, labels = shuffle(features, labels)
    features = normalize(features)
    labels = one_hot_encode(labels)
    pickle.dump((features), open('my_cifar10_features.p', 'wb'))
    pickle.dump((labels), open('my_cifar10_labels.p', 'wb'))
    return features, labels

def show_me(index, features, labels):
    '''Show the image at the index specified together with its label'''
    label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    plt.imshow(features[index])
    plt.show()
    print('Label Id: {} Name: {}'.format(np.argmax(labels[index]), label_names[np.argmax(labels[index])]))
#%%    
'''Import data, visualize some random samples and keep some random images as test set'''
print('Importing Data...\n')
features, labels = load_and_normalize_new_data('C:/dev/Tensorflow/CIFAR-10/cifar-10-batches-py')

for index in random.sample(range(0, 10000), 10):
    show_me(index, features, labels)
    
test_size = 2500
num_images = len(features)
train_features = features[:(num_images - test_size)]
train_labels = labels[:(num_images - test_size)]
eval_features = features[(num_images - test_size):]
eval_labels = labels[(num_images - test_size):]

    
#%%
# PARAMETERS (depend on the structure of the dataset)
width = 32
height = 32
layers = 3
class_output = 10

# HYPERPARAMETERS
eval_size = 2500 # percentage of the training set used for cross-validation
train_size = num_images - test_size - eval_size
epochs = 15
batch_size = 100
keep_probability = 0.5 # dropout weight keeping probability
reg_parameter = 0.0
learning_rate = 0.001
num_batches = int(train_size / batch_size)
display_frequency = 20 # indicates how many iterations occur before showing results

#%%
'''Build tensorflow model'''
x = tf.placeholder(tf.float32, shape=[None, width, height, layers], name='input_x')
y = tf.placeholder(tf.float32, shape=[None, class_output], name='input_y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
regularizer = tf.contrib.layers.l1_regularizer(scale=reg_parameter)

# LAYER 1 (CONV + MAX POOL)
conv1 = tf.layers.conv2d(x, 64, 3, 1, padding='SAME', activation=tf.nn.relu, use_bias=True, name='conv1',
                         kernel_regularizer=regularizer)
conv1POOL = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
conv1_bn = tf.layers.batch_normalization(conv1POOL)
# LAYER 2 (CONV + MAX POOL)
conv2 = tf.layers.conv2d(conv1_bn, 128, 3, 1, padding='SAME', activation=tf.nn.relu, use_bias=True, name='conv2',
                         kernel_regularizer=regularizer)
conv2POOL = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
conv2_bn = tf.layers.batch_normalization(conv2POOL)
# LAYER 3 (CONV + MAX POOL)
conv3 = tf.layers.conv2d(conv2_bn, 256, 3, 1, padding='SAME', activation=tf.nn.relu, use_bias=True, name='conv3',
                         kernel_regularizer=regularizer)
conv3POOL = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
conv3_bn = tf.layers.batch_normalization(conv3POOL)
# LAYER 4 (CONV + MAX POOL)
# conv4 = tf.layers.conv2d(conv3_bn, 512, 3, 1, padding='SAME', activation=tf.nn.relu, use_bias=True, name='conv4', kernel_regularizer=regularizer)
# conv4POOL = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# conv4_bn = tf.layers.batch_normalization(conv4POOL)

flat = tf.contrib.layers.flatten(conv3_bn)

# LAYER 5 (FULLY CONNECTED WITH DROPOUT)
full1 = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=128, activation_fn=tf.nn.relu,
                                          weights_regularizer=regularizer)
full1DROP = tf.nn.dropout(full1, keep_prob)
full1_bn = tf.layers.batch_normalization(full1DROP)
# LAYER 6 (FULLY CONNECTED WITH DROPOUT)
full2 = tf.contrib.layers.fully_connected(inputs=full1_bn, num_outputs=256, activation_fn=tf.nn.relu,
                                          weights_regularizer=regularizer)
full2DROP = tf.nn.dropout(full2, keep_prob)
full2_bn = tf.layers.batch_normalization(full2DROP)
# LAYER 7 (FULLY CONNECTED WITH DROPOUT)
full3 = tf.contrib.layers.fully_connected(inputs=full2_bn, num_outputs=512, activation_fn=tf.nn.relu,
                                          weights_regularizer=regularizer)
full3DROP = tf.nn.dropout(full3, keep_prob)
full3_bn = tf.layers.batch_normalization(full3DROP)
# LAYER 8 (FULLY CONNECTED WITH DROPOUT)
full4 = tf.contrib.layers.fully_connected(inputs=full3_bn, num_outputs=1024, activation_fn=tf.nn.relu,
                                          weights_regularizer=regularizer)
full4DROP = tf.nn.dropout(full4, keep_prob)
full4_bn = tf.layers.batch_normalization(full4DROP)
# LAYER 9 (FULLY CONNECTED WITH DROPOUT)
full5 = tf.contrib.layers.fully_connected(inputs=full4_bn, num_outputs=10, activation_fn=None,
                                          weights_regularizer=regularizer)
# calculate predictions
pred = tf.nn.softmax(full5)

# Define LOSS function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=full5))

# Define Optimizer
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# Define Correct Prediction
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

# Define Accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#%%
'''Launch Training Session'''
#tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
with tf.Session() as sess:
    acc_list = []
    loss_list = []
    val_list = []
    print('Training...\n')
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        print('Epoch ' + str(epoch + 1) + '\n')
        # shuffle training data and select a percentage as evaluation set
        train_features, train_labels = shuffle(train_features, train_labels)
        x_train = train_features[:train_size]
        y_train = train_labels[:train_size]
        x_eval = train_features[train_size:]
        y_eval = train_labels[train_size:]
        
        for num_batch in range(num_batches):
            start_dict = num_batch * batch_size
            end_dict = start_dict + batch_size
            train_dict = {x: x_train[start_dict: end_dict],
                          y: y_train[start_dict: end_dict],
                          keep_prob: keep_probability}
            train_step.run(feed_dict=train_dict)
            # print the training accuracy obtained (only sometimes)
            if num_batch % display_frequency == 0:
                print_train_dict={x: x_train[start_dict: end_dict],
                                  y: y_train[start_dict: end_dict],
                                  keep_prob: 1.0}
                train_loss = cross_entropy.eval(feed_dict = print_train_dict)
                train_accuracy = accuracy.eval(feed_dict=print_train_dict)
                print("\tstep:", num_batch, 
                      " training accuracy:", train_accuracy,
                      " training loss:", train_loss)
                acc_list.append(train_accuracy)
                loss_list.append(train_loss)
                
        ### EVALUATION SESSION ###
        eval_dict = {x: x_eval, y: y_eval, keep_prob: 1.0}
        validation_accuracy = accuracy.eval(feed_dict=eval_dict)
        val_list.append(validation_accuracy)
        print("\n\tvalidation accuracy: ", validation_accuracy, "\n")

        plt.plot(acc_list)
        plt.plot(loss_list)
        plt.show()

    plt.plot(val_list)
    plt.show()


















