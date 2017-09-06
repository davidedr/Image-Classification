'''
Created on 05 set 2017

@author: davide
'''

'''
    Demonstrate how to divide a dataset into batches to be then fed to some learning algorithm
    and how to get the single batches chosen randomly 
'''
import numpy as np

# Dataset (or indexes into dataset array)
idxs = np.arange(100)

# Dataset is separated into batches of size batch_size
batch_size = 10
n_batches = len(idxs)//batch_size
print('idxs :' + str(idxs) + '\nn_batches: ' + str(n_batches))

for batch_i in range(n_batches):
    print(idxs[batch_i * batch_size : (batch_i + 1)*batch_size])
    
# Now batches are randomly composed
print()
print('Batches of randomly chosen indexes')
rand_idxs = np.random.permutation(idxs)

idxs = rand_idxs
for batch_i in range(n_batches):
    print(idxs[batch_i * batch_size : (batch_i + 1)*batch_size])
