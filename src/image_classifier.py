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

landing_foldername = foldername
N = 5
done = False
img_count = 0
img_index = 0
while not done:
    url = content[img_index]
    image_filename = 'train_000%03d.jpg' % img_index
    image_filename_pathcomplete = os.path.join(foldername, image_filename)
    print(str(img_index) + ', ' + image_filename + ', ' +  url)
    try: 
        urllib.request.urlretrieve(url, image_filename_pathcomplete)
    except:
        None
    else:
        img_count += 1
    img_index += 1
    if img_count == N:
        done = True
    if img_index >= len(content):
        done = True
        
print('Done.')
    