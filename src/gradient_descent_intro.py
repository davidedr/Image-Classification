'''
Created on 07 set 2017

@author: davide
'''

'''
    introduction to Gradient Descent implementation in Tensor flow 
    to lean the parameters W, B of a single node NN
'''

##
## Session 2: Neural Network training w/ Tensor flor
##

#
# 2.1 Introduction
#

n_observations = 1000

import numpy as np
# Data to be learn is composed of a sin wave.
# Signal (ys) is distorted by some random noise
xs = np.linspace(-3, 3, n_observations)
ys = np.sin(xs) + np.random.uniform(-0.5, 0.5, n_observations)

import matplotlib.pyplot as plt
plt.figure()
plt.scatter(xs, ys, alpha = 0.5, marker = '+')
plt.title('Input data: xs -->> ys')
plt.show()

# Teach a network to represent a function like ys
# i.e., I give the nn a value on the x axis, the nn gives me the value on the y axis
# the sin value of the given x point

import tensorflow as tf
X = tf.placeholder(tf.float32, name = 'X')
Y = tf.placeholder(tf.float32, name = 'Y')

# Model parameters, W (weight), B (bias). Model is: Y = WX + B
# W is initialized with random values
W = tf.Variable(tf.random_normal([1], dtype = tf.float32, stddev = 0.1), name = 'weight')
# B is initialized with a constant value
B = tf.Variable(tf.constant([0], dtype = tf.float32), name = 'bias')

# The model
Y_pred = W * X + B

# Now we need to learn the parameters W and B. This could be done w/ Linear Regression.
# Let's use Gradient Descent instead.

# Cost function
def distance(y_pred, y):
    distance = tf.abs(y - y_pred)
    return distance

# For a psecific point in the x axis:
# Y_pred is the prediction; sin(x) is the ground truth
#cost = distance(Y_pred, tf.sin(xs))

# Of course we do not know that the phenomena we're at and is actually a sin wave
# We have the data, and we'll make the machine learn from those data
cost = distance(Y_pred, Y)

# Since we have entire data set (several couples (x, y) in input), the total cost (loss)
# can be defined as the average cost computed over all inputs
# (Several other definitions of cost could have been used)
cost = tf.reduce_mean(distance(Y_pred, Y))

optimizer = tf.train.GradientDescentOprimizer(learning_rate = 0.01).minimize(cost)
