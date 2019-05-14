#!/usr/bin/env python
# coding: utf-8

# # **Quick Draw: Catch Doodle!**

# “Quick, Draw!” is a game created by Google. It's a game where one player is prompted to draw a picture of an object, and the other player needs to guess what it is. More details can be found [this post](https://towardsdatascience.com/quick-draw-the-worlds-largest-doodle-dataset-823c22ffce6b). 
# 
# This project is for building an image classifier model that can handle noisy and sometimes incomplete drawings and perform well on classifying 50 different animals.

# Table of contents:
# - [1. Load data](# 1. Load data)
# - [2. Let's draw doodle](# 2. Let's draw doodle)
# - [3. From strokes to Image](# 3. From strokes to Image)
# - [4. Let's call all friends here!](# 4. Let's call all friends here!)
# - [5. Modeling- CNN](# 5. Modeling- CNN)
# - [6. Plot the result](# 6. Plot the result)
# - [7. Modeling with ResNet50](# 7. Modeling with ResNet50)

# In[ ]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white', context='notebook')

np.random.seed(36)


# In[ ]:


import ast
import cv2
import dask.bag as db

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau 

from keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg19 import VGG19


# # 1. Load data <a></a>

# I'm going to use only the train_simplified.zip file for training. And we will use only the animal drawings among them. Cause...they are soooo cute :-) To load all 50 animals files automatically, I'll make a list for filenames. 

# In[ ]:


# list of animals 
animals = ['ant', 'bat', 'bear', 'bee', 'bird', 'butterfly', 'camel', 'cat', 'cow',
           'crab', 'crocodile', 'dog', 'dolphin', 'dragon', 'duck', 'elephant', 'fish',
           'flamingo', 'frog', 'giraffe', 'hedgehog', 'horse', 'kangaroo', 'lion',
           'lobster', 'monkey', 'mosquito', 'mouse', 'octopus', 'owl', 'panda',
           'parrot', 'penguin', 'pig', 'rabbit', 'raccoon', 'rhinoceros', 'scorpion',
           'sea turtle', 'shark', 'sheep', 'snail', 'snake', 'spider', 'squirrel',
           'swan', 'teddy-bear', 'tiger', 'whale', 'zebra']


# Before uploading all the files at once, let's take a test with .csv first.

# In[ ]:


dir_path = '../input/train_simplified/'
df = pd.read_csv(dir_path + animals[0] + '.csv')
df.head()


# `drawing` is the stroke values, which is telling the drawing of animals. We need to exchage this data into image data later. `word` indicates the result of drawings or animals. `recognized` means whether the drawing was understood as a certain object or not. Let's take only the 10 rows per animals and filter the unrecogizable drawing out. 

# In[ ]:


am = pd.DataFrame(columns = df.columns)

for i in range(len(animals)):
    filename = dir_path + animals[i] + '.csv'
    df = pd.read_csv(filename, nrows = 100)
    df = df[df.recognized == True]
    am = am.append(df)


# In[ ]:


# Check
am.word.nunique()


# # 2. Let's draw doodle <a></a>

# Before data proprocessing and modeling, let's see how people drew animals. The image information can be found at `drawing` but in order to make it visual, we need some steps of processing. Let's take only 100 data for an example.

# In[ ]:


# Sampling only 100 examples
ex = am.sample(100)
ex.head()


# In[ ]:


ex.drawing.head(1).values   # -> strings


# Take a note at the front of the result. 
# <br>
# array(**[ ' [[ 34, 41  ... ]]]) **
# <br>
# This indicates this data is strings, not list.

# In[ ]:


ex.drawing.head(1).map(ast.literal_eval).values   # -> list


# We can see that the dypes changed. Now we are able to use it for plotting the strokes. Let's convert the other data as well.

# In[ ]:


# Convert to list
ex['drawing'] = ex.drawing.map(ast.literal_eval)


# Now we are going to meet our lovely cats, dogs and pandas. Take a look at the code below.

# In[ ]:


# Plot the strokes 
# fig, axs = plt.subplots(nrows = 10, ncols = 10, figsize = (10, 8))

# for index, col in enumerate(ex.drawing):
#     ax = axs[index//10, index%10]
#     for x, y in col:
#         ax.plot(x,-np.array(y), lw = 3)
#     ax.axis('off')
    
# plt.show()


# The concept of visualization can be seen complex at the first sight but it's not true. First we will get 100 grids and put the drawing in the grids one by one. `enumerate()` will return the index and column values. Let's take one example to understand step by step. 

# In[ ]:


# Understanding enumerate
for index, col in enumerate(ex.drawing[:12]):
    print('The index is ', index)
    print('Position will be ({}, {})'.format(index//10, index%10))
    print('The strokes are ', col)
    print('===========')


# As you can see above `enumerate()` brings us the index and the values one by one. So we are going to plot the values at the given values by col. 

# In[ ]:


for index, col in enumerate(ex.drawing[:2]):
    print('==================================')
    for x, y in col:
        print('X is {}'.format(x))
        print('Y is {}'.format(y))
        print('-----------------------')


# So what we are going to do is ploting these x, y values just like what we've been doing with graphs. Now let's apply all this into one shot and finally meet our lovely friends.

# In[ ]:


# Plot the strokes 
fig, axs = plt.subplots(nrows = 10, ncols = 10, figsize = (10, 8))

for index, col in enumerate(ex.drawing):
    ax = axs[index//10, index%10]
    for x, y in col:
        ax.plot(x,-np.array(y), lw = 3)
    ax.axis('off')
    
plt.show()


# OMG 🙉🙈😍.....This is so funny.

# # 3. From strokes to Image <a></a>

# Now the next step is transforming all these drawings into image data. Like I said above, the data isn't in the form of image data. We have to covert it into numpy array format. I'm going to make a function for this. 

# In[ ]:


im_size = 64
n_class = len(animals)


# In[ ]:


# define a function converting drawing to image data
def draw_to_img(strokes, im_size = im_size):

    fig, ax = plt.subplots()                        # plot the drawing as we did above
    for x, y in strokes:
        ax.plot(x, -np.array(y), lw = 10)
    ax.axis('off')
    
    fig.canvas.draw()                               # update a figure that has been altered
    A = np.array(fig.canvas.renderer._renderer)     # converting them into array
    
    plt.close('all')
    plt.clf()
    
    A = (cv2.resize(A, (im_size, im_size)) / 255.)  # image resizing to uniform format
    
    return A


# All the things we discussed at the second section are put inside the `draw_to_img()` function.  Let's try it with an example.

# In[ ]:


X = ex.drawing.values
image = draw_to_img(X[1])
plt.imshow(image)


# In[ ]:


image.shape


# The image has 4 channels and we can also check each channels separately.

# In[ ]:


# Channel selection 
fig, axs = plt.subplots(nrows = 1, ncols = 4, figsize = (10, 10))

for i in range(4):
    ax = axs[i]
    ax.imshow(image[:, :, i])


# We will make the input image shape as `(im_size, im_size, 3)`, which means it has only one channels. Therefore we''ll take only the last channel here.

# In[ ]:


# redefine
def draw_to_img(strokes, im_size = im_size):
    fig, ax = plt.subplots()                        # plot the drawing as we did above
    for x, y in strokes:
        ax.plot(x, -np.array(y), lw = 10)
    ax.axis('off')
    
    fig.canvas.draw()                               # update a figure that has been altered
    A = np.array(fig.canvas.renderer._renderer)     # converting them into array
    
    plt.close('all')
    plt.clf()
    
    A = (cv2.resize(A, (im_size, im_size)) / 255.)  # image resizing to uniform format

    return A[:, :, :3]                               # drop the last one 


# In[ ]:


image = draw_to_img(X[1])
plt.imshow(image)
print(image.shape)


# # 4. Let's call all friends here! <a></a>

# Now we are ready to apply what we've been doing so far into the entire dataset.

# In[ ]:


n_samples = 500
X_train = np.zeros((1, im_size, im_size, 3))
y = []

for a in animals:
    print(a)
    filename = dir_path + a + '.csv'
    df = pd.read_csv(filename, usecols=['drawing', 'word'], nrows=n_samples)  # import the data in chunks
    df['drawing'] = df.drawing.map(ast.literal_eval)                          # convert strings into list
    X = df.drawing.values
    
    img_bag = db.from_sequence(X).map(draw_to_img)                            # covert strokes into array
    X = np.array(img_bag.compute())  
    X_train = np.vstack((X_train, X))                                         # concatenate to get X_train  
    
    y.append(df.word)


# As I just stack the array, the dimension of `X_train` has one more values than it's expected. Therefore we'll drop the first layer.  

# In[ ]:


# The dimension of X_train
X_train.shape


# In[ ]:


# Drop the first layer
X_train = X_train[1:, :, :, :]
X_train.shape


# Don't forget to encoding the categorical data before modeling fitting

# In[ ]:


# Encoding 
y = pd.DataFrame(y)
y = pd.get_dummies(y)
y_train = np.array(y).transpose()


# In[ ]:


# Check the result
print("The input shape is {}".format(X_train.shape))
print("The output shape is {}".format(y_train.shape))


# Now let's combine the X_train and y_train again. This is for splitting the data into the trainning set and validation set.  

# In[ ]:


# Reshape X_train
X_train_2 = X_train.reshape((X_train.shape[0], im_size*im_size*3))

# Concatenate X_train and y_train
X_y_train = np.hstack((X_train_2, y_train))


# ![resize](https://github.com/jjone36/Doodle/blob/master/resize.png)

# In[ ]:


# Random shuffle
np.random.shuffle(X_y_train)
a = im_size*im_size*3
cut = int(len(X_y_train) * .1)
X_val = X_y_train[:cut, :a]
y_val = X_y_train[:cut, a:]
X_train = X_y_train[cut:, :a]
y_train = X_y_train[cut:, a:]

# Reshape X_train back to (64, 64)
X_train = X_train.reshape((X_train.shape[0], im_size, im_size, 3))
X_val = X_val.reshape((X_val.shape[0], im_size, im_size, 3))


# Check the final shape of train and validation set.

# In[ ]:


# Check the result
print("The input shape of train set is {}".format(X_train.shape))
print("The input shape of validation set is {}".format(X_val.shape))
print("The output shape of train set is {}".format(y_train.shape))
print("The output shape of validation set is {}".format(y_val.shape))


# # 5. Modeling- CNN <a></a>

# I'm going to start with the basic CNN model as a baseline. And then compare the results with ResNet and VGG19

# In[ ]:


n_epochs = 10
batch_size = 500

# Initialize
model = Sequential()

# ConvNet_1
model.add(Conv2D(32, kernel_size = 3, input_shape = (im_size, im_size, 3), padding = 'same', activation = 'relu'))
model.add(MaxPool2D(2, strides = 2))
# Dropout
model.add(Dropout(.2))

# ConvNet_2
model.add(Conv2D(64, kernel_size = 3, activation = 'relu'))
model.add(MaxPool2D(2, strides = 2))
# Dropout
model.add(Dropout(.2))

# ConvNet_3
model.add(Conv2D(64, kernel_size = 3, activation = 'relu'))
model.add(MaxPool2D(2, strides = 2))
# Dropout
model.add(Dropout(.2))

# Flattening
model.add(Flatten())

# Fully connected
model.add(Dense(680, activation = 'relu'))

# Dropout
model.add(Dropout(.5))

# Final layer
model.add(Dense(n_class, activation = 'softmax'))

# Compile
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# In[ ]:


model.summary()


# I'll also add callbacks not to get overfitting. 

# In[ ]:


# Early stopper
stopper = EarlyStopping(monitor='val_top_3_accuracy', mode='max', patience = 3)

# Learning rate reducer
reducer = ReduceLROnPlateau(monitor = 'val_acc',
                           patience = 3,
                           verbose = 1,
                           factor = .5,
                           min_lr = 0.00001)

callbacks = [stopper, reducer]


# In[ ]:


# Fitting baseline
history = model.fit(X_train, y_train, epochs = n_epochs, batch_size = batch_size, 
                    validation_split = .2, verbose = True)


# # 6. Plot the result <a></a>

# Let's see how well our model is trained.

# In[ ]:


# Train and validation curves
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(history.history['loss'], color = 'b', label = 'Train Loss')
ax1.plot(history.history['val_loss'], color = 'm', label = 'Valid Loss')
ax1.legend(loc = 'best')

ax2.plot(history.history['acc'], color = 'b', label = 'Train Accuracy')
ax2.plot(history.history['val_acc'], color = 'm', label = 'Valid Accuracy')
ax2.legend(loc = 'best')


# # 7. Modeling with ResNet50

# It's seem not good. Let's try other pre-trained model.

# In[ ]:


# ResNet50 Application 
model_r = ResNet50(include_top = True, weights= None, input_shape=(im_size, im_size, 3), classes = n_class)


# In[ ]:


model_r.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model_r.summary()


# In[ ]:


n_epochs = 5
batch_size = 50


# In[ ]:


# Fitting ResNet50
history_r = model_r.fit(X_train, y_train, epochs = n_epochs, batch_size = batch_size, 
                      validation_split = .2, verbose = True)


# In[ ]:


# Train and validation curves with ResNet50
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(history_r.history['loss'], color = 'b', label = 'Train Loss')
ax1.plot(history_r.history['val_loss'], color = 'm', label = 'Valid Loss')
ax1.legend(loc = 'best')

ax2.plot(history_r.history['acc'], color = 'b', label = 'Train Accuracy')
ax2.plot(history_r.history['val_acc'], color = 'm', label = 'Valid Accuracy')
ax2.legend(loc = 'best')


# In[ ]:





# In[ ]:




