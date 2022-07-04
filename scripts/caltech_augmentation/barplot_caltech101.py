import numpy as np
import matplotlib.pyplot as plt
import os

a_dataset_path = '/home/javi/uzh-rpg/datasets/a-caltech101_bis/classes'
folder_names = []
folder_names = [f for f in sorted(os.listdir(a_dataset_path))]

print(len(folder_names)) # 101 = 100 categories + background

a_category_dict = {}
a_images_per_category_dict = {}

a_total_images = 0

for i, category in enumerate(folder_names):
    a_category_dict[i] = category
    
    folder_path = a_dataset_path + '/' + category
    #image_names = [os.path.join(folder_path, img) for img in sorted(os.listdir(folder_path))]
    image_names = [img for img in sorted(os.listdir(folder_path))]
    
    a_images_per_category_dict[i] = len(image_names)
    
    print('%s: %d' %(category, a_images_per_category_dict[i]))
    a_total_images += a_images_per_category_dict[i]
    
print('Total images in dataset: %d' %(a_total_images))


o_dataset_path = '/home/javi/uzh-rpg/datasets/caltech101/classes'

o_category_dict = {}
o_images_per_category_dict = {}

o_total_images = 0

for i, category in enumerate(folder_names):
    o_category_dict[i] = category
    
    folder_path = o_dataset_path + '/' + category
    #image_names = [os.path.join(folder_path, img) for img in sorted(os.listdir(folder_path))]
    image_names = [img for img in sorted(os.listdir(folder_path))]
    
    o_images_per_category_dict[i] = len(image_names)
    
    print('%s: %d' %(category, o_images_per_category_dict[i]))
    o_total_images += o_images_per_category_dict[i]
    
print('Total images in dataset: %d' %(o_total_images))


#PLOT
N = len(folder_names)
ind = np.arange(N)    # the x locations for the groups
width = 0.5       # the width of the bars: can also be len(x) sequence

p1 = plt.bar(*zip(*sorted(a_images_per_category_dict.items())), width)
p2 = plt.bar(*zip(*sorted(o_images_per_category_dict.items())), width)

plt.ylabel('#samples')
plt.title('Caltech 101')
plt.xticks(ind,folder_names, rotation='vertical')
plt.yticks(np.arange(0, 801, 50))
plt.legend((p1[0], p2[0]), ('Augmented', 'Original'))
# Tweak spacing to prevent clipping of tick-labels
plt.subplots_adjust(bottom=0.15)
plt.show()

#PLOT SORTED
import operator
N = len(folder_names)
ind = np.arange(N)    # the x locations for the groups
width = 0.5       # the width of the bars: can also be len(x) sequence

sorted_augmented = sorted(a_images_per_category_dict.items(), key=operator.itemgetter(1))
sorted_original = []
sorted_names = []

for i in sorted_augmented:
    sorted_original.append((i[0], o_images_per_category_dict[i[0]]))
    sorted_names.append(folder_names[i[0]])

p1 = plt.bar(range(N), [item[1] for item in sorted_augmented])
p2 = plt.bar(range(N), [item[1] for item in sorted_original])

plt.ylabel('#samples')
plt.title('Caltech 101')
plt.xticks(ind,sorted_names, rotation='vertical')
plt.yticks(np.arange(0, 801, 50))
plt.legend((p1[0], p2[0]), ('Augmented with %d samples'%(a_total_images), 'Original with %d samples'%(o_total_images)))
# Tweak spacing to prevent clipping of tick-labels
plt.subplots_adjust(bottom=0.15)
plt.show()
