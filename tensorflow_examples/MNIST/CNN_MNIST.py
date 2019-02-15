import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Start interactive session
sess = tf.InteractiveSession()

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# DEFINES and Placeholders
width = 28 # width of the image in pixels
height = 28 # height of the image in pixels
flat = width * height # number of pixels in one image
class_output = 10 # number of possible classifications for the problem

x  = tf.placeholder(tf.float32, shape=[None, flat])
y_ = tf.placeholder(tf.float32, shape=[None, class_output])
x_image = tf.reshape(x, [-1,28,28,1]) #because x is a placeholders for an image

# LAYER I CONVOLUTION
W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32])) # need 32 biases for 32 outputs
convolve1= tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1
#--------------------------------------------------------------

# LAYER II ReLU
h_conv1 = tf.nn.relu(convolve1)
#--------------------------------------------------------------

# LAYER III MAXPOOL
conv1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #max_pool_2x2
#--------------------------------------------------------------

# LAYER IV CONVOLUTION (2)
W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64])) #need 64 biases for 64 outputs
convolve2= tf.nn.conv2d(conv1, W_conv2, strides=[1, 1, 1, 1], padding='SAME')+ b_conv2
#--------------------------------------------------------------

# LAYER V ReLU (2)
h_conv2 = tf.nn.relu(convolve2)
#--------------------------------------------------------------

# LAYER VI MAXPOOL (2)
conv2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #max_pool_2x2
#--------------------------------------------------------------

# LAYER VII FULLY-CONNECTED
layer2_matrix = tf.reshape(conv2, [-1, 7*7*64])
W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024])) # need 1024 biases for 1024 outputs
fcl=tf.matmul(layer2_matrix, W_fc1) + b_fc1
#--------------------------------------------------------------

# LAYER VIII ReLU (3)
h_fc1 = tf.nn.relu(fcl)
#--------------------------------------------------------------

# LAYER IX DROP-OUT
keep_prob = tf.placeholder(tf.float32)
layer_drop = tf.nn.dropout(h_fc1, keep_prob)
#--------------------------------------------------------------

# LAYER X FULLY-CONNECTED (2)
W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1)) #1024 neurons
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10])) # 10 possibilities for digits [0,1,2,3,4,5,6,7,8,9]
fc=tf.matmul(layer_drop, W_fc2) + b_fc2
#--------------------------------------------------------------

# LAYER XI SOFTMAX [OUTPUTS]
y_CNN= tf.nn.softmax(fc)
#--------------------------------------------------------------

# Define LOSS function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_CNN), reduction_indices=[1]))

# Define Optimizer
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

# Define Prediction
correct_prediction = tf.equal(tf.argmax(y_CNN,1), tf.argmax(y_,1))

# Define Accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# TRAINING SESSION
sess.run(tf.global_variables_initializer())

for i in range(1100):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, float(train_accuracy)))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
#--------------------------------------------------------------

# VALIDATION SESSION
validation_accuracy = 0.0
for i in range(100):
    batch = mnist.validation.next_batch(50)
    validation_accuracy += accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
print("validation accuracy %g"%(validation_accuracy/100))
#--------------------------------------------------------------

# TEST SESSION
test_accuracy = 0.0
for i in range(200):
    batch = mnist.test.next_batch(50)
    test_accuracy += accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})

print("test accuracy %g"%(test_accuracy/200))
#--------------------------------------------------------------

sess.close() #finish the session