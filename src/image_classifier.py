'''
Created on 04 set 2017

@author: davide
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
            print(str(img_index) + ', ' + image_filename + ', ' +  url + " don't ovewrite!")
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

print('Done.')
    