'''
Created on 07 set 2017

@author: davide
'''

'''
    Demonstrate definition, training, use of a single-node neural network
    using Tensor flow to learn some input data
'''

#
# 2.2 Training the NN using the whole input dataset
#

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# The data to learn from
n_observations = 1000
xs = np.linspace(-3, 3, n_observations)
ys = np.sin(xs) + np.random.uniform(-0.5, 0.5, n_observations)

fig, ax = plt.subplots(1,1)
ax.scatter(xs, ys, alpha = 0.5, marker = '+')
plt.title('Input data')
ax.set_xlabel('xs')
ax.set_ylabel('ys')
plt.show()

# Metaparameters
n_iterations = 500
precision = 1E-6

# Initial values for parameters
W = tf.Variable(tf.random_normal([1], dtype=tf.float32, stddev=0.1), name='weight')
B = tf.Variable(tf.constant([0], dtype=tf.float32), name='bias')

# Model
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')
Y_pred = X * W + B

# Cost function
def distance(y_pred, y):
    distance = tf.abs(y - y_pred)
    return distance
cost = tf.reduce_mean(distance(Y_pred, Y))

# Optimizer procedure
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

fig, ax = plt.subplots(1,1)
ax.scatter(xs, ys, alpha = 0.5, marker = '+')

overall_training_cost = []
Ws = []
Bs = []
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    prev_training_cost = 0.0
    for it_i in range(n_iterations):
        
        session.run(optimizer, feed_dict = { X: xs, Y: ys})
        training_cost = session.run(cost, feed_dict = { X: xs, Y: ys})
       
        w = session.run(W)
        Ws.append(w)
        
        b = session.run(B)
        Bs.append(b)
        
        print('Iteration: ' + str(it_i) + ', training_cost: ' + str(training_cost))
        overall_training_cost.append(training_cost)
        # Quit early if cost does not change enough
        if np.abs(prev_training_cost - training_cost) < precision:
            break
        
        # Keep track of the previous cost
        prev_training_cost = training_cost
        
        # That's all for NN parameter estimation
        # Let's show something
        if it_i % 10 == 0:
            ys_pred = Y_pred.eval(feed_dict = { X: xs }, session = session)
            ax.plot(xs, ys_pred, 'k', alpha = it_i/n_iterations, color = 'red')
            fig.show()
            plt.draw()

plt.show()

print('Learnt values W: ' + str(w) + ', B: ' + str(b))

plt.figure()
plt.plot(Ws)
plt.title('W (slope)')
plt.xlabel('Iteration no.')
plt.show()

plt.figure()
plt.plot(Bs)
plt.title('B (intercept)')
plt.xlabel('Iteration no.')
plt.show()

plt.figure()
plt.plot(range(len(overall_training_cost)), overall_training_cost)
plt.title('Training cost')
plt.xlabel('Iteration no.')
plt.ylabel('Cost/Loss')
plt.show()