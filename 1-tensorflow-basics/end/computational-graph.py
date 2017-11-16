import tensorflow as tf

#1. Constants
c1 = tf.constant(1.0)
c2 = tf.constant(2.0)
print(c1, c2) #tensors of type float

#2. Session
session = tf.Session()
print(session.run([c1, c2]))

#3. Add
add = tf.add(c1, c2)
print(add)
print(session.run(add));

#4. Placeholders
p1 = tf.placeholder(tf.float32)
p2 = tf.placeholder(tf.float32)
addPH = p1 + p2 #nicer
print(session.run(addPH, {p1: 3, p2: 4}))
print(session.run(addPH, {p1: 5.5, p2: 6.6}))
print(session.run(addPH, {p1: [7, 8], p2: [9, 10]}))

#5. Variables
a = tf.Variable([1], dtype=tf.float32)
b = tf.Variable([-2], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linearModel = a * x + b #f(x)=x-2
init = tf.global_variables_initializer()
session.run(init) #assigns values to variables
print(session.run(linearModel, {x: [0, 1, 2, 3, 4, 5]}))

#6. Adjust variables to fit f(x)=x/2-1
y = tf.placeholder(tf.float32)
errors = tf.square(linearModel - y)
loss = tf.reduce_sum(errors)
feed = {
    x: [0, 1, 2, 3, 4, 5],
    y: [-1, -0.5, 0, 0.5, 1, 1.5]
}
print(session.run(loss, feed))
#loss is high, would like it to be 0 or as close as possible to 0
assignA = tf.assign(a, [0.5])
assignB = tf.assign(b, [-1])
session.run([assignA, assignB])
print(session.run(loss, feed))