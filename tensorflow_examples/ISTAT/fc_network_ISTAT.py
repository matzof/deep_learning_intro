import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import pandas as pd
import os
#%%
''' RUN ONLY ONCE, THEN UPLOAD DATA DROM /data/processed_population.spydata ''''''
ROOT_PATH = "C:/dev/Tensorflow/istat"
df = pd.read_csv(os.path.join(ROOT_PATH, "population.csv"), low_memory=False)
print(df.describe())
print(df.shape)

df = df.sort_values('City', ascending=True)
city_list = []
[city_list.append(element) for element in set(df.loc[:]['City'])]
city_list.sort()
# create a dictionary to store data separatedly according to city
cities = {}
for city in city_list:
    cities[city] = []
# store data in the correct key of the cities dictionary
counter = 0
for index, row in df.iterrows():
    counter += 1
    if index % 1549 == 0: # keep track of the progress
        print('Current progress: ', int((counter*100)/len(df)), '%')
    cities[row.loc['City']].append([row['Sex'], row['Age'], row['Time'], row['Population']])
for city in city_list:
    cities[city] = np.asarray(cities[city])  

'''        
#%%
'''train model for the specific city requested '''
def normalize_data(data):
    new_data = []
    for feat in range(np.size(data, axis=1)):
        min_value = np.min(data[:, feat])
        max_value = np.max(data[:, feat]) 
        new_data.append(np.divide(data[:, feat] - min_value, max_value - min_value))
    return np.transpose(np.asarray(new_data))

#%%
'''Weight, bias and fully connected layer initializers '''
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
def train_model(city):
    '''Hyperparameters'''
    learning_rate = 0.0003
    training_epochs = 500
    batch_size = 100
    stall_check = 100 # number of epochs after which there will be a stall check
    cost_history = np.empty(shape=[1],dtype=float)
    
    h1 = 8182               # number of nodes in the 1st hidden layer
    h2 = 512
    
    # separate latest data (2018) as test set and keep the rest for training set
    train_data = cities[city_request]
    max_value = np.max(train_data[:, 3])
    train_data = normalize_data(train_data)
    num_features = np.size(train_data, axis=1) - 1
    num_elements = len(train_data)
    test_index = []
    for index, point in enumerate(train_data):
        if point[2] == 1:
            test_index.append(index)
    x_test = train_data[test_index, 0:3]
    y_test = train_data[test_index, 3]
    x_train = np.delete(train_data, test_index, axis=0)[:, 0:3]
    y_train =  np.delete(train_data, test_index, axis=0)[:, 3]
    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))
    
    # build tensorflow model
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
    
    # run tensorflow model
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    previous_loss = 9999999999999
    for epoch in range(training_epochs):
        print('Training epoch: {}'.format(epoch + 1)) 
        x_train, y_train = sklearn.utils.shuffle(x_train, y_train)
        num_train_iter = int(len(y_train) / batch_size)
        for iteration in range(num_train_iter):
            start = iteration * batch_size
            end = (iteration + 1) * batch_size
            x_batch, y_batch = x_train[start:end], y_train[start:end]
            np.reshape(y_batch, (len(y_batch), 1))
    
            # Run optimization op (backprop)
            feed_dict_batch = {x: x_batch, y: y_batch}
            sess.run(optimizer, feed_dict=feed_dict_batch)
    #        # Calculate and display the batch loss and accuracy
    #        loss_batch = sess.run(cost, feed_dict=feed_dict_batch)
    #
    #        print("iter {0:3d}:\t Loss={1:.2f}".
    #              format(iteration, loss_batch))
    
        # Run validation after every epoch
        feed_dict_valid = {x: x_test, y: y_test}
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
    
    # plot cost history
    plt.plot(range(len(cost_history[1:])),cost_history[1:])
    plt.show()
    # plot test results
    pred = sess.run(prediction, feed_dict={x: x_test})
    error = np.abs(np.subtract(pred, np.reshape(np.asarray(y_test), pred.shape)))
    mean_error = np.mean(error) * max_value
    print('The mean prediction error is: ', mean_error)
    print('The relative error is: ', int(np.mean(error)*100), '%')

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, pred)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()
    
    sess.close()    

#%%
'''Ask the city of interest '''
print(city_list)
city_request = input("Please enter the name of the city for which you would like to get a population projection choosing it from the list above (case sensitive): ")
for city in city_list:
    if city_request == city:
        train_model(city)
        
#%% 
'''Approach based on dictionaries'''
## get labels
#dictionary = california_housing_dataframe.to_dict()
#labels = dictionary.pop('Population')
## get features
#city = dictionary.pop('City')
#age = dictionary.pop('Age')
#sex = dictionary.pop('Sex')
#time = dictionary.pop('Time')
#
#def rmv_lines(unuseful_elements):
#    for index in unuseful_elements:
#        city.pop(index)
#        sex.pop(index)
#        time.pop(index)
#        age.pop(index)
#        labels.pop(index)
#
## remove data relative to the total population, keeping the one differenciated by sex
#unuseful_elements = []    
#for key in sex:
#    if sex[key] == 9:
#        unuseful_elements.append(key)
#rmv_lines(unuseful_elements)
#
## generate a list of all the cities included
#cities = []
#[cities.append(element) for element in city.values()]
#cities = set(cities)
#
#def separate_data_by_city():
#%%    
    
        


