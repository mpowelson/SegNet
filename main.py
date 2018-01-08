# -*- coding: utf-8 -*-
"""
Created on 1/2/2018
This code works beautifully on dataset_comb1. It fits the training set exactly and 
does really well on the CV set too. My only concern is that the data in the CV
and training set look really similar to the point that I am concerned about overfitting regardless.

@author: Matthew
"""

import numpy as np
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, ZeroPadding2D, Conv2D, BatchNormalization, UpSampling2D, Reshape, Permute, Activation, MaxPooling2D
from tensorflow.python.keras import applications
from tensorflow.python.keras.utils import get_file
from tensorflow.python.keras import optimizers
import matplotlib.pyplot as plt

# dimensions of our images.
img_width, img_height = 672,672
#img_width, img_height = 4048, 3036


train_data_dir = 'dataset_comb1/images_prepped_train'
train_annotation_dir = 'dataset_comb1/annotations_prepped_train'
validation_data_dir = 'dataset_comb1/images_prepped_validation'
validation_annotation_dir = 'dataset_comb1/annotations_prepped_validation'

batch_size = 1
seed = np.random.randint(0,1000)


#%% Start Data augmentation
print("Building Data Generators")
# we create two instances with the same arguments since Keras does not have
# good support of data augmentation w/ semantic segmentation yet.
#data_gen_args_image = dict(featurewise_center=False,
#                     featurewise_std_normalization=False,
#                     rotation_range=10.,
#                     width_shift_range=0.1,
#                     height_shift_range=0.1,
#                     zoom_range=0.5,
#                     rescale=1. / 255,
#                     horizontal_flip = True) 
data_gen_args_image = dict(featurewise_center=False,
                     featurewise_std_normalization=False,
                     rotation_range=0.,
                     width_shift_range=0.,
                     height_shift_range=0.,
                     zoom_range=0.,
                     rescale=1.,
                     horizontal_flip = True) 

# Only necessary when input image is normalized to 0-1
data_gen_args_mask = dict(data_gen_args_image)   
data_gen_args_mask['rescale'] = 1.0

image_datagen = ImageDataGenerator(**data_gen_args_image) 
mask_datagen = ImageDataGenerator(**data_gen_args_mask)

# Provide the same seed and keyword arguments to the fit and flow methods seed = 1 
# Necessary when using featurewise norm, etc.
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


image_generator_val = image_datagen.flow_from_directory(
    validation_data_dir,
    class_mode=None,
    batch_size=batch_size,
    seed=seed,
    target_size=(img_height,img_width))

mask_generator_val = mask_datagen.flow_from_directory(
    validation_annotation_dir,
    class_mode=None,
    batch_size=batch_size,
    seed=seed,
    target_size=(img_height,img_width))

# combine generators into one which yields image and masks 
# If broken try itertools.izip() instead
validation_generator = zip(image_generator_val, mask_generator_val)

## Uncomment to view data generated
#i=0
#for batch in train_generator:
#    i += 1
#    img_mask = np.squeeze(batch[1]*255)
#    img = np.uint8(np.squeeze(batch[0]*255))  #No idea why I have to multiply twice
##    plt.figure(figsize=(10, 8))
#    plt.subplot(2,1,1)
#    plt.imshow(img)
#    plt.imshow(img_mask, cmap='jet', alpha=0.5)
#    plt.subplot(2,1,2)
#    plt.imshow(img)
#    plt.savefig('preview/preprocessed' + str(i) + '.png', dpi=800)
#    plt.show()
##    time.sleep(5)
#    if i > 5:
#        break  # otherwise the generator would loop indefinitely
   
    
#%% Build Model

model = Sequential()


nClasses = 3


# Build VGG16 - Encoder
initial_model = applications.VGG16(weights="imagenet", include_top=False, input_shape=( img_height,img_width,3))
last = initial_model.output

# Build Decoder 
o = ( UpSampling2D( (2,2), data_format='channels_last'))(last)
o = ( ZeroPadding2D( (1,1) , data_format='channels_last' ))(o)
o = ( Conv2D(512, (3, 3), padding='valid', data_format='channels_last'))(o)
o = ( BatchNormalization())(o)

o = ( UpSampling2D( (2,2), data_format='channels_last'))(o)
o = ( ZeroPadding2D( (1,1) , data_format='channels_last' ))(o)
o = ( Conv2D(512, (3, 3), padding='valid', data_format='channels_last'))(o)
o = ( BatchNormalization())(o)

o = ( UpSampling2D( (2,2), data_format='channels_last'))(o)
o = ( ZeroPadding2D( (1,1), data_format='channels_last'))(o)
o = ( Conv2D( 256, (3, 3), padding='valid', data_format='channels_last'))(o)
o = ( BatchNormalization())(o)

o = ( UpSampling2D((2,2)  , data_format='channels_last' ) )(o)
o = ( ZeroPadding2D((1,1) , data_format='channels_last' ))(o)
o = ( Conv2D( 128 , (3, 3), padding='valid' , data_format='channels_last' ))(o)
o = ( BatchNormalization())(o)

o = ( UpSampling2D((2,2)  , data_format='channels_last' ))(o)
o = ( ZeroPadding2D((1,1)  , data_format='channels_last' ))(o)
o = ( Conv2D( 64 , (3, 3), padding='valid'  , data_format='channels_last' ))(o)
o = ( BatchNormalization())(o)

o =  Conv2D( nClasses , (3, 3) , padding='same', data_format='channels_last' )( o )

preds = (Activation('sigmoid'))(o)


model = Model(initial_model.input, preds)


# Display the model summary
model.summary()


# disables layers for if using pretrained model
for layer in model.layers:
    layer.trainable = False
# was 21 should be 22 probably
for layer in model.layers[-26:]:
    layer.trainable = True
    print("Layer '%s' is trainable" % layer.name)  

#%% Train model
from tensorflow.python.keras import backend as K

# Custom loss heavily weights 1's since they are rare
# Original weighting (before redialating) = 5000
def custom_binary_crossentropy(y_true, y_pred):
    return -1*(y_true*K.log(y_pred)*10 + (1-y_true)*K.log(1-y_pred))

optimizer = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(loss=custom_binary_crossentropy,
              optimizer= optimizer ,
              metrics=['accuracy'] )


hist = model.fit_generator(train_generator,
              steps_per_epoch=10, 
              epochs=5,
              verbose=1,
              callbacks=None,
              validation_data=validation_generator,
              validation_steps=1,
              class_weight=None, 
              max_queue_size=10,
              workers=1,
              use_multiprocessing=False, 
              initial_epoch=0)
print("Training Complete")
model.save('preview/tempModel.h5')  #save in case something happens

# Load a model from a file (eg. the one saved above)
#from keras.models import load_model
#model = load_model('my_model.h5')

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
        
#%% View some of the results

import time

i=0
#for batch in train_generator:
for batch in validation_generator:
    i += 1
    timer = time.time()
    A = model.predict(batch[0])
    print(time.time()-timer)  
    img_mask = np.squeeze(np.uint8(A[0,:,:,:]>0.5)*255)
    img = np.uint8(np.squeeze(batch[0]))
#    plt.figure(figsize=(10, 8))
    plt.subplot(2,2,1)
    plt.title('Output')
    plt.imshow(img)
    plt.imshow(img_mask, cmap='jet', alpha=0.5)
    plt.subplot(2,2,2)
    plt.title('Raw Image')
    plt.imshow(img)
    plt.subplot(2,2,3)
    plt.title('Correct Label')

    plt.imshow(img)
    plt.imshow(np.squeeze(np.uint8(batch[1])*255), alpha=0.5)
    
    plt.savefig('preview/output' + str(i) + '.png' ,dpi=800)
#    time.sleep(5)
    print('Predicting on Output: ', i)
    if i > 4:
        break  # otherwise the generator would loop indefinitely
    
# Plot Loss Curve
plt.figure(figsize=(10, 8))
plt.subplot(2,1,1)
plt.imshow(img)

plt.subplot(2,1,2)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.savefig('preview/lossCurve' + '.png' ,dpi=800)
print("Total Pixels that are 1s: " , np.sum(np.uint8(A>0.5)))


