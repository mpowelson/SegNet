# -*- coding: utf-8 -*-
"""
This code is an example of using data augmentation on a semantic segmenation problem.
You should apply the same augmentation to each and then 'zip' them together

@author: Matthew
"""

import numpy as np
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator


# dimensions of our images.
img_width, img_height = 672, 672


train_data_dir = 'dataset_comb1/images_prepped_train'
train_annotation_dir = 'dataset_comb1/annotations_prepped_train'
validation_data_dir = 'dataset_comb1/images_prepped_validation'
validation_annotation_dir = 'dataset_comb1/annotations_prepped_validation'
nb_train_samples = 1
nb_validation_samples = 1
epochs = 1
batch_size = 1
seed = 1


#%%

# we create two instances with the same arguments 
data_gen_args = dict(featurewise_center=False,
                     featurewise_std_normalization=False,
                     rotation_range=10.,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.5) 

data_gen_args = dict(featurewise_center=False,
                     featurewise_std_normalization=False,
                     rotation_range=0.,
                     width_shift_range=0.,
                     height_shift_range=0.,
                     zoom_range=0.,
                     rescale=1.,
                     horizontal_flip = False) 

image_datagen = ImageDataGenerator(**data_gen_args) 
mask_datagen = ImageDataGenerator(**data_gen_args)

# Provide the same seed and keyword arguments to the fit and flow methods seed = 1 
#image_datagen.fit(images, augment=True, seed=seed) 
#mask_datagen.fit(masks, augment=True, seed=seed)

image_generator = image_datagen.flow_from_directory(
    train_data_dir,
    class_mode=None,
    batch_size=batch_size,
    seed=seed,
    target_size=(img_height,img_width))

mask_generator = mask_datagen.flow_from_directory(
    train_annotation_dir,
    class_mode=None,
    batch_size=batch_size,
    seed=seed,
    target_size=(img_height,img_width))

# combine generators into one which yields image and masks 
# If broken try itertools.izip() instead
train_generator = zip(image_generator, mask_generator)

#model.fit_generator(
#    train_generator,
#    samples_per_epoch=2000,
#    nb_epoch=50)

#%% Preview some images
i = 0
for batch in image_datagen.flow_from_directory(train_data_dir, batch_size=1, target_size=(672,672),
                          save_to_dir='preview', save_prefix='data', save_format='png'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely
        
#%%
import matplotlib.pyplot as plt

i=0
for batch in train_generator:
    i += 1
    A = batch[1]
#    print(A.shape)
#    print(batch[0].shape)
    img_mask = np.squeeze(np.uint8(A[0,:,:,:])*255)
    img = np.uint8(np.squeeze(batch[0]))
    plt.figure(figsize=(10, 8))
    plt.subplot(2,2,1)
    plt.title('Mask')
#    plt.imshow(img)
    plt.imshow(img_mask, alpha=0.5)
    plt.subplot(2,2,2)
    plt.title('Raw Image')
    plt.imshow(img)
    plt.subplot(2,2,3)
    plt.title('Correct Label')
    plt.imshow(img)
    plt.imshow(img_mask, cmap = 'jet', alpha=0.5)
    
    plt.savefig('preview/sample' + str(i) + '.png' ,dpi=800)
#    time.sleep(5)
    print('Predicting on Output: ', i)
    if i > 1:
        break  # otherwise the generator would loop indefinitely
   