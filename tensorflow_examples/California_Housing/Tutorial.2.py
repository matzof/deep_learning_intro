'''Training and validation over California Housing Dataset'''
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import skimage

#tf.logging.set_verbosity(tf.logging.INFO)
#%%
ROOT_PATH = "C:/dev/Tensorflow/California_Housing"
california_housing_dataframe = pd.read_csv(os.path.join(ROOT_PATH, "california_housing_train.csv"))
california_housing_dataframe = california_housing_dataframe.reindex(
        np.random.permutation(california_housing_dataframe.index))

def preprocess_features(california_housing_dataframe):
    """ Preprocess features
    INPUT -> standard California Housing Dataset
    OUTPUT -> includes only the training features (creating a new one composing two of them)
    """
    selected_features = california_housing_dataframe[
            ["latitude", "longitude", "housing_median_age", "total_rooms",
             "total_bedrooms", "population", "households", "median_income"]]
    processed_features = selected_features.copy()
    # Create a synthetic feature.
    processed_features["rooms_per_person"] = (
            california_housing_dataframe["total_rooms"] / california_housing_dataframe["population"])
    return processed_features

def preprocess_labels(california_housing_dataframe):
    """ Preprocess labels
    INPUT -> standard California Housing Dataset
    OUTPUT -> dataframe with labels
    """
    output_labels = pd.DataFrame()
    # Scale the label to be in units of thousands of dollars.
    output_labels["median_house_value"] = (
            california_housing_dataframe["median_house_value"] / 1000.0)
    return output_labels

''' select 12000/17000 as training samples'''
training_examples = preprocess_features(california_housing_dataframe.head(12000))
training_labels = preprocess_labels(california_housing_dataframe.head(12000))
# training_examples.describe()
# training_labels.describe()
'''select the remaining 5000/17000 as validation samples'''
validation_examples = preprocess_features(california_housing_dataframe.tail(5000))
validation_labels = preprocess_labels(california_housing_dataframe.tail(5000))
#%%
'''Plot Latitude/Longitude vs. Median House Value'''
plt.figure(figsize=(13, 8))

ax = plt.subplot(1, 2, 1)
ax.set_title("Validation Data")
plt.scatter(validation_examples["longitude"], validation_examples["latitude"], cmap="coolwarm",
            c=validation_labels["median_house_value"] / validation_labels["median_house_value"].max())

ax = plt.subplot(1,2,2)
ax.set_title("Training Data")
plt.scatter(training_examples["longitude"], training_examples["latitude"], cmap="coolwarm",
            c=training_labels["median_house_value"] / training_labels["median_house_value"].max())
plt.plot()
#%%
def convert_dataframe_into_dict(dataframe):
    dataframe = {key:np.array(value) for key,value in dict(dataframe).items()}
    return dataframe

def my_input_fn(features, labels, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model of multiple features.
    INPUT -> features, labels, training options (batch size, num epochs)
    OUTPUT -> Tuple of (features, labels) for next data batch
    """
    
    # Convert pandas data into a dict of np arrays.
    features = convert_dataframe_into_dict(features)                                           
 
    # Construct a dataset, and configure batching/repeating.
    ds = tf.data.Dataset.from_tensor_slices((features,labels)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    # Shuffle the data, if specified.
    if shuffle:
        ds = ds.shuffle(10000)
    
    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

#training_examples, training_labels = my_input_fn(
#        training_examples, training_labels, batch_size=10, shuffle=True, num_epochs=100)
    
# Convert pandas data into a dict of np arrays.
training_examples = convert_dataframe_into_dict(training_examples)                                           
# Construct a dataset, and configure batching/repeating.
ds = tf.data.Dataset.from_tensor_slices((training_examples, training_labels)) # warning: 2GB limit
ds = ds.batch(1).repeat(100)
# Shuffle the data, if specified.
ds = ds.shuffle(10000)
# Return the next batch of data.
training_examples, training_labels = ds.make_one_shot_iterator().get_next()

































