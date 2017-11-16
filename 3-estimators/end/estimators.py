import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.logging.set_verbosity(tf.logging.ERROR)

imageSize = 28
labelSize = 10
hiddenSize = 1024

mnist = input_data.read_data_sets("resources/MNIST_Data/", one_hot=False)

def format(dataset):
    features = dataset.images
    labels = dataset.labels.astype(np.int32)
    return features, labels

featureColumns = [tf.contrib.layers.real_valued_column("", dimension=imageSize * imageSize)]
classifier = tf.contrib.learn.DNNClassifier(
    feature_columns=featureColumns,
    hidden_units=[hiddenSize], #more layers can be added here
    n_classes=labelSize,
    optimizer=tf.train.AdamOptimizer()) #not gradient descent this time

#Fit model
features, labels = format(mnist.train)
classifier.fit(x=features, y=labels, batch_size=100, steps=1000) 
#one line for all our previous code w/ more complex neural network!

#Test accuracy
features, labels = format(mnist.train)
testAccuracy = classifier.evaluate(x=features, y=labels, batch_size=100, steps=1000)['accuracy']
print("Test accuracy: %g %%"%(testAccuracy * 100))

#Evaluate model
features = mnist.validation.images[:10]
labels = mnist.validation.labels[:10].astype(np.int32)
predictions = classifier.predict(x=features)
print("Predicted labels from validation set: %s"%list(predictions))
print("Underlying values: %s"%list(labels))