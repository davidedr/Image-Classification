'''
Created on 04 set 2017

@author: davide
'''

'''
    Perform several operations on a set of images (downloading from url, reading, cropping
    rescaling, compute mean std dev, normalize, ...
    This work is done on behalf and loosely based on:
    Parag K. Mital's "Creative Applications of Deep Learning w/ Tensorflow" Kadenze Academy course  
'''
import os

# Read image links from file
print('Reading image links...')
filename = 'imagenet.synset.txt'
foldername = '.\\..\\data'
filename_pathcomplete = os.path.join(foldername, filename) 
with open(filename_pathcomplete) as f:
    content = f.readlines()
# Remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content]

# Download N images
print('Downloading images...')
import urllib.request

OVERWRITE = False

landing_foldername = foldername
N = 100
done = False
img_count = 0
img_index = 0
image_filenames = []
while not done:
    url = content[img_index]
    image_filename = 'train_000%03d.jpg' % img_count
    image_filename_pathcomplete = os.path.join(foldername, image_filename)
    if os.path.exists(image_filename_pathcomplete):
        if OVERWRITE:
            os.remove(image_filename_pathcomplete)
        else:
            img_index += 1
            img_count += 1
            print(str(img_index) + ', ' + image_filename + ', ' +  url + " don't overwrite!")
    else:
        print(str(img_index) + ', ' + image_filename + ', ' +  url)
        try: 
            urllib.request.urlretrieve(url, image_filename_pathcomplete)
        except:
            None
        else:
            if 'flick' in url and os.path.getsize(image_filename_pathcomplete) == 2051:
                os.remove(image_filename_pathcomplete)
                print('flicr empty image.')
            else:
                img_count += 1
                image_filenames.append(image_filename_pathcomplete)
            
        img_index += 1
    if img_count == N:
        done = True
    if img_index >= len(content):
        done = True
    
import matplotlib.pyplot as plt

filenames = image_filenames   
# Read every filename as an RGB image
filenames = [os.path.join(foldername, fname)
             for fname in os.listdir(foldername) if not os.path.isdir(fname) and 'train' in fname]
imgs = [plt.imread(fname)[..., :3] for fname in filenames]

# Check images: need to have all the same number of channels
# in shape (W, H, C) C must be the same for all images 
[print(img_index, ', ', img.shape) for img_index, img in enumerate(imgs)]

from libs import utils
# Crop every image to a square
imgs = [utils.imcrop_tosquare(img_i) for img_i in imgs]

# Then resize the square image to 100 x 100 pixels
from skimage.transform import resize
imgs = [resize(img_i, (100, 100)) for img_i in imgs]

# Finally make our list of 3-D images a 4-D array with the first dimension the number of images:
import numpy as np
imgs = np.array(imgs).astype(np.float32)

# Plot the resulting dataset:
# Make sure you "run" this cell after you create your `imgs` variable as a 4-D array!
# Make sure we have a 100 x 100 x 100 x 3 dimension array
assert(imgs.shape == (100, 100, 100, 3))
plt.figure(figsize=(10, 10))
plt.imshow(utils.montage(imgs, saveto='dataset.png'))
plt.show()
    
img_mean = np.mean(imgs, axis = 0)
plt.figure()
plt.imshow(img_mean)
plt.title('Mean image')
plt.show()

img_stddev = np.std(imgs, axis = 0)
plt.figure()
plt.imshow(img_stddev)
plt.title('Standard dev image')
plt.show()

'''    
#
# Use Tensor flow to compute mean image
#
import tensorflow as tf
img_mean_tf = tf.reduce_mean(imgs, axis=0, keep_dims = True)
print('img_mean_tf.shape: ' + str(img_mean_tf.shape))

s = tf.Session()
img_mean_tf = img_mean_tf.eval(session = s)
s.close()

img_mean_tf = img_mean_tf[0, :, :, :]
plt.figure()
plt.imshow(img_mean_tf)
plt.show()

#
# Compute std dev image via Tensor Flow
#

# Create a tensorflow operation to give you the standard deviation

# First compute the difference of every image with a
# 4 dimensional mean image shaped 1 x H x W x C
mean_img_4d = tf.reduce_mean(imgs, axis=0, keep_dims = True)

subtraction = imgs - mean_img_4d

# Now compute the standard deviation by calculating the
# square root of the expected squared differences
std_img_op = tf.sqrt(tf.reduce_mean(subtraction * subtraction, axis=0))

# Now calculate the standard deviation using your session
s = tf.Session()
std_img = std_img_op.eval(session = s)
s.close()

# Then plot the resulting standard deviation image:
# Make sure the std image is the right size!
assert(std_img.shape == (100, 100) or std_img.shape == (100, 100, 3))
plt.figure(figsize=(10, 10))
std_img_show = std_img / np.max(std_img)
plt.imshow(std_img_show)
plt.show()
plt.imsave(arr=std_img_show, fname='std.png')
'''

# Normalize the dataset images
imgs_normalized = (imgs - img_mean)/img_stddev
plt.figure(figsize=(10, 10))
plt.imshow(utils.montage(imgs_normalized, 'normalized.png'))
plt.show()

'''
Apply another type of normalization to 0-1 just for the purposes of plotting the image.
If we didn't do this, the range of our values would be somewhere between -1 and 1,
and matplotlib would not be able to interpret the entire range of values. By rescaling our -1 to 1 valued images to 0-1,
we can visualize it better.
'''
norm_imgs_rescaled = (imgs_normalized - np.min(imgs_normalized)) / (np.max(imgs_normalized) - np.min(imgs_normalized))
plt.figure(figsize = (10, 10))
plt.imshow(utils.montage(norm_imgs_rescaled, 'normalized.png'))
plt.show()

#
# Compute the convolution kernel
#
ksize = imgs[0].shape[0]
print('ksize: ' + str(ksize))

kernel = np.concatenate([utils.gabor(ksize)[:, :, np.newaxis] for i in range(3)], axis = 2)

# Now make the kernels into the shape: [ksize, ksize, 3, 1]:
kernel_4d = np.reshape(kernel, [ksize, ksize, 3, 1])

'''
    If yu want to do the reshape operation in Tensor flow land instead of Numpy's,
    keep in mind that in the first case kernel_4d in not actually computed until
    you request it. 
'''
'''
kernel_4d = tf.reshape(kernel, [ksize, ksize, 3, 1])
with tf.Session():
    kernel_4d = kernel_4d.eval()
'''

assert(kernel_4d.shape == (ksize, ksize, 3, 1))

# Display and save the kernel
plt.figure(figsize = (5, 5))
plt.imshow(kernel_4d[:, :, 0, 0], cmap='gray')
plt.show()
plt.imsave(arr=kernel_4d[:, :, 0, 0], fname='kernel.png', cmap='gray')

#
# Perform the convolution
#
convolved = utils.convolve(imgs, kernel_4d)

convolved_show = (convolved - np.min(convolved)) / (np.max(convolved) - np.min(convolved))

print(convolved_show.shape)
plt.figure(figsize=(10, 10))
plt.imshow(utils.montage(convolved_show[..., 0], 'convolved.png'), cmap='gray')
plt.show()

print('Done.')
