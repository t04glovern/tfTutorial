from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import tensorflow as tf

## a placeholder value that we'll input when
## we ask TensorFlow to run a computation.
x = tf.placeholder(tf.float32, [None, 784])

## Initialize as tensors full of zeros
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
## Notice that W is a shape of [784,10] because
## we want to multiply the 784-dimonsional image
## vectors to produce 10-dimentional vectors

y = tf.nn.softmax(tf.matmul(x, W) + b)
## multiply x by W with the expression tf.matmul(x, W)
## then add b and finally apply softmax

## Cross-entropy input placeholder
y_ = tf.placeholder(tf.float32, [None, 10])

## Apply cross-entropy
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
## tf.log computes the logarithm of each element of y.
## then multiply each element of y_ with the corresponding
## element of tf.log(y).
## Then tf.reduce_sum adds the elements in the second dimension of y
## due to the reduction_indicies=[1] parameter.
## Finally, tf.reduce_mean computes the mean over all the examples

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
## we ask TensorFlow to minimize cross_entropy using the gradient
## descent algorithm with a learning rate of 0.5

## launch a Session, and run the operation that inits the variables
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

## Lets train step 1000 times
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

## tf.argmax is a function which gives you the index of the
## highest entry in a tensor along some axis
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
## tf.argmax(y,1) is the label our model thinks is the most
## likely for each input
## tf.argmax(y_,1) is the correct label
## We use tf.equal to check if our prediction matches the truth.
## This gives us a list of booleans [True, False, True, True]
## which would become [1,0,1,1] which would become 0.75

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
## Take the mean value using the expression above

## Finally we ask for our accuracy one our test data
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))