'''
Created on 08 set 2017

@author: davide
'''

'''
    A function to train a neural network via Stochastic mini batches gradient descent 
    using Tensor flow
'''

#
# 2.4 NN training using the train function
#

import tensorflow as tf
import numpy as np

tf.set_random_seed(1)
np.random.seed(1)

def distance(y_pred, y):
    '''
        L1-norm distance of y_pred from y
    '''
    distance = tf.abs(y - y_pred)
    return distance

def cost_loss(y_pred, y):
    '''
        Cost/loss function as the mean of the distances of predictions 
        from actual data
    '''
    cost = tf.reduce_mean(distance(y_pred, y))
    return cost

import matplotlib.pyplot as plt

def train(X, Y, Y_pred, xs, ys, n_iterations = 100, batch_size = 200, learning_rate = 0.02):
    
    Ws = []
    Bs = []
    cost = cost_loss(Y_pred, Y)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    fig, ax = plt.subplots(1, 1)
    ax.scatter(xs, ys, alpha = 0.15, marker = '+')
    with tf.Session() as session:
        
        session.run(tf.global_variables_initializer())
        training_costs = []
        
        for it_i in range(n_iterations):
            idxs = np.random.permutation(range(len(xs)))
            n_batches = len(idxs)//batch_size
            for batch_i in range(n_batches):
                idxs_i = idxs[batch_i*batch_size: (batch_i + 1)*batch_size]
                session.run(optimizer, feed_dict = { X: xs[idxs_i], Y: ys[idxs_i] })
        
                w = session.run(W)
                Ws.append(w)

                b = session.run(B)
                Bs.append(b)

            training_cost = session.run(cost, feed_dict = { X: xs[idxs_i], Y: ys[idxs_i] })
            training_costs.append(training_cost)
            
            if it_i % 10 == 0:
                ys_pred = Y_pred.eval(feed_dict = { X: xs, Y: ys }, session = session)
                ax.plot(xs, ys_pred, 'k', alpha = it_i/n_iterations, color = 'red')
                print('it_i: ' + str(it_i) + ', training_cost: ' + str(training_cost))
    plt.show()
    
    print('Learnt values W: ' + str(w) + ', B: ' + str(b))
    # Measure variance of paramters during learning
    print('Parameter std dev, stdandard deviation of (W): ' + str(np.std(Ws)) + ', stdandard deviation of B: ' + str(np.std(Bs)))

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
    plt.plot(range(len(training_costs)), training_costs)
    plt.title('Training cost')
    plt.show()
    
if __name__ == "__main__":
        
    # Metaparameters
    n_iterations = 500
    
    # Input data to learn from
    n_observations = 1000
    xs = np.linspace(-3, 3, n_observations)
    ys = np.sin(xs) + np.random.uniform(-0.5, 0.5, n_observations)
    
    fig, ax = plt.subplots(1,1)
    ax.scatter(xs, ys, alpha = 0.5, marker = '+')
    plt.title('Input data')
    ax.set_xlabel('xs')
    ax.set_ylabel('ys')
    plt.show()
    
    # Initial values for parameters
    W = tf.Variable(tf.random_normal([1], dtype = tf.float32, stddev = 0.1), name = 'weight')
    B = tf.Variable(tf.constant([0], dtype = tf.float32), name = 'bias')
    
    # Model
    X = tf.placeholder(tf.float32, name = 'X')
    Y = tf.placeholder(tf.float32, name = 'Y')
    Y_pred = X * W + B
    batch_size = 1000
    learning_rate = 0.01
    train (X, Y, Y_pred, xs, ys, n_iterations = n_iterations, batch_size = batch_size, learning_rate = learning_rate)
