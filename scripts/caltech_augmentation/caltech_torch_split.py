import os, shutil

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]="0" 

import numpy as np
import cv2
from matplotlib import pyplot as plt

import keras
print("keras version: ", keras.__version__)
import tensorflow as tf
print("tensoflow version: ", tf.__version__)

# instructions from: https://michaelblogscode.wordpress.com/2017/10/10/reducing-and-profiling-gpu-memory-usage-in-keras-with-tensorflow-backend/

# TensorFlow wizardry
config = tf.ConfigProto() 
# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.5 
# Create a session with the above options specified.
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

dataset_path = '/home/javi/uzh-rpg/datasets/a-caltech101_bis/classes'
folder_names = []
folder_names = [f for f in sorted(os.listdir(dataset_path))]
print(len(folder_names)) # 101 = 100 categories + background

image_path = dataset_path + '/camera/image_0002.jpg'
image = cv2.imread(image_path)
plt.imshow(image)
plt.show()

# One row per class category. Select categories randomly.
categories_num = 9
images_number = 9 # images per class shown
categories_selected = np.random.randint(0, 101, categories_num, dtype='l')

# print categories selected
print('Selected categories:')
print([folder_names[i] for i in categories_selected])


fig, ax = plt.subplots(nrows=9, ncols=9)
fig.set_size_inches(9.5, 8.5)

#plt.subplots_adjust(top=0.85) # to include title on TOP of figure. Otherwise it overlaps due to tight_layout

fig.subplots_adjust(wspace=0.1,hspace=0.1)

for i, category in enumerate(categories_selected):
    folder_path = dataset_path + '/' + folder_names[category]
    # take the first objects
    image_names = [img for img in sorted(os.listdir(folder_path))][:images_number]
    
    for j, image_name in enumerate(image_names):
        image_path = folder_path + '/' + image_name
        image = cv2.imread(image_path)
        # resize to 100x100 for all images for this plot
        image = cv2.resize(image, (100, 100)) 
        #plt.figure()
        #plt.imshow(image)
        ax[i,j].imshow(image)
        #ax[i,j].set_axis_off()
        ax[i,j].set_xticks([])
        ax[i,j].set_yticks([])
        
        if j == 0:
            pad = 5 # in points
            #ax[i,j].set_ylabel(folder_names[category], rotation=0, size='large')
            ax[i,j].annotate(folder_names[category], xy=(0, 0.5), xytext=(-ax[i,j].yaxis.labelpad - pad, 0),
                xycoords=ax[i,j].yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center')
        
fig.tight_layout()
#plt.axis('off')
fig.show()

category_dict = {}
images_per_category_dict = {}
category_images_path_dict = {}

total_images = 0

for i, category in enumerate(folder_names):
    category_dict[i] = category
    
    folder_path = dataset_path + '/' + category
    #image_names = [os.path.join(folder_path, img) for img in sorted(os.listdir(folder_path))]
    image_names = [img for img in sorted(os.listdir(folder_path))]
    
    images_per_category_dict[i] = len(image_names)
    category_images_path_dict[i] = image_names
    
    print('%s: %d' %(category, images_per_category_dict[i]))
    total_images += images_per_category_dict[i]
    
print('Total images in dataset: %d' %(total_images))

base_path = '/home/javi/uzh-rpg/datasets/a-caltech101_bis/split'
train_dir = os.path.join(base_path, 'train')
validation_dir = os.path.join(base_path, 'valid')
test_dir = os.path.join(base_path, 'test')

# create the directories to use
os.mkdir(base_path)

train_dir = os.path.join(base_path, 'train')
os.mkdir(train_dir)

validation_dir = os.path.join(base_path, 'valid')
os.mkdir(validation_dir)

test_dir = os.path.join(base_path, 'test')
os.mkdir(test_dir)
# create the categories files in each

for directory in [train_dir, validation_dir, test_dir]:
    for category in folder_names:
        os.mkdir(os.path.join(directory, category))

# calculate the number of images to place in each train/valid/test categories folder

total_train = 0
total_validation = 0
total_test = 0

total_train_2 = 0
total_validation_2 = 0
total_test_2 = 0

f_train = open("train.txt", "w")
f_valid = open("valid.txt", "w")
f_test = open("test.txt", "w")
for i, category in enumerate(folder_names):
    train_number = int(0.6 * images_per_category_dict[i])
    validation_number = int(0.2 * images_per_category_dict[i])
    test_number = images_per_category_dict[i] - train_number - validation_number # for not exceeding maximum number
    
    # for statistics later
    total_train += train_number
    total_validation += validation_number
    total_test += test_number
    
    # now copy these images to respective folders
    fnames = category_images_path_dict[i][:train_number]
    f_train.write("%s:\n"%category)
    for fname in fnames:
        src = os.path.join(dataset_path, category, fname)
        dst = os.path.join(train_dir, category, fname)
        shutil.copyfile(src, dst)
        f_train.write("  - %s\n"%fname)
        
    total_train_2 += len(fnames)
        
    fnames = category_images_path_dict[i][train_number:train_number + validation_number]
    f_valid.write("%s:\n"%category)
    for fname in fnames:
        src = os.path.join(dataset_path, category, fname)
        dst = os.path.join(validation_dir, category, fname)
        shutil.copyfile(src, dst)
        f_valid.write("  - %s\n"%fname)
        
    total_validation_2 += len(fnames)
    
    fnames = category_images_path_dict[i][train_number + validation_number:]
    f_test.write("%s:\n"%category)
    #print("%s:\n"%category)
    for fname in fnames:
        src = os.path.join(dataset_path, category, fname)
        dst = os.path.join(test_dir, category, fname)
        shutil.copyfile(src, dst)
        f_test.write("  - %s\n"%fname)
   # print("%d\n"%len(fname))
        
    total_test_2 += len(fnames)
#close files
f_train.close()
f_valid.close()
f_test.close()
# print statistics

print('Correct train split: ', total_train == total_train_2)
print('Correct valid split: ', total_validation == total_validation_2)
print('Correct test split: ', total_test == total_test_2)
print()
print('Number of training images: ', total_train)
print('Number of valid images: ', total_validation)
print('Number of test images: ', total_test)
print()
print('Real percentage of training images: ', total_train / total_images)
print('Real percentage of valid images: ', total_validation / total_images)
print('Real percentage of test images: ', total_test / total_images)