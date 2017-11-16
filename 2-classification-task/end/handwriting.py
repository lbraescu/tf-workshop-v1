import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#Read data (downloaded automagically if not found) ~20000 examples
mnist = input_data.read_data_sets("resources/MNIST_Data/", one_hot=True)

#Parameters
imageSize = 28
labelSize = 10
learningRate = 0.05
steps = 1000
batchSize = 100

#Placeholders
trainingData = tf.placeholder(tf.float32, [None, imageSize * imageSize])
labels = tf.placeholder(tf.float32, [None, labelSize])

#Variables
W = tf.Variable(tf.truncated_normal([imageSize * imageSize, labelSize], stddev=1.0))
#normal distribution for weights
b = tf.Variable(tf.constant(0.1, shape=[labelSize]))
#constant value for biases

#Output layer
output = tf.matmul(trainingData, W) + b

#Loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=output))
#tf.nn.softmax_cross_entropy_with_logits that internally applies the softmax on the model's unnormalized prediction and sums across all classes
#tf.reduce_mean function takes the average over these sums

#Training step
train_step = tf.train.GradientDescentOptimizer(learningRate).minimize(loss)
#gradient descent method further optimises the loss function with each learning batch

#Accuracy calculation
correctPrediction = tf.equal(tf.argmax(output, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))
#tf.argmax return max value from tensor dimension
#tf.equal returns a list of boolean values where output matches label
#tf.cast converts booleans to float values
#tf.reduce_mean calculates the average of the above float values
#i.e. 2/10 correct predictions yields 20% accuracy

#Run the training
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())

for i in range(steps):
    inputBatch, labelBatch = mnist.train.next_batch(batchSize)
    feed = {
        trainingData: inputBatch,
        labels: labelBatch
    }
    train_step.run(feed_dict=feed)

    if i % 100 == 0:
        trainAccuracy = accuracy.eval(feed_dict=feed) #session.run
        print("Step %d, training batch accuracy %g %%"%(i, trainAccuracy * 100))

#Evaluate model
testAccuracy = accuracy.eval(feed_dict={trainingData: mnist.test.images, labels: mnist.test.labels})
print("Test accuracy: %g %%"%(testAccuracy * 100))
#not very consistent, trained on a few batches, accuracy of one specific batch
#maybe tensorflow has higher level APIs for fitting the machine learning coefficients