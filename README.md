# LE-NET-DEEP-NETWORK-FOR-TRAFFIC-SIGN-CLASSIFICATION
Traffic sign classification using LE-NET Architecture 
# Traffic sign classification using LE-NET Architecture

# Importing Libraries and Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from sklearn.metrics import confusion_matrix



# Loading the dataset
with open(r'C:\Users\chukw\PycharmProjects\8_Practical_Project\traffic-signs-data\train.p', mode= 'rb') as training_data:
    train = pickle.load(training_data)

with open(r'C:\Users\chukw\PycharmProjects\8_Practical_Project\traffic-signs-data\valid.p', mode= 'rb') as validation_data:
    valid = pickle.load(validation_data)

with open(r'C:\Users\chukw\PycharmProjects\8_Practical_Project\traffic-signs-data\test.p', mode= 'rb') as testing_data:
    test = pickle.load(testing_data)

# Exploring the dataset
X_train, y_train = train['features'], train['labels']
X_validation,y_validation = valid['features'],valid['labels']
X_test, y_test = test['features'], test['labels']

print(X_train.shape)
print(y_train.shape)

print(X_validation.shape)
print(y_validation.shape)

print(X_test.shape)
print(y_train.shape)

# Image Exploration

# Creating an image matrix with the training data
nrows = 15
ncols = 15

fig,axes = plt.subplots(nrows,ncols, figsize=(25,25))
axes = axes.ravel() # To flatten the 15x15 matrix to give 225
n_training = len(X_train) # 34799
for i in np.arange(0,nrows*ncols): # i ranges from 0 to 244

    index = np.random.randint(0,n_training) # pick a random number from the training data
    axes[i].imshow(X_train[index])
    axes[i].set_title(y_train[index],size=5 )
    axes[i].axis('off')

plt.subplots_adjust(hspace=1)
plt.show()

# creating an image matrix with the testing data
ncols = 15
nrows = 15

fig,axes = plt.subplots(nrows,ncols, figsize=(30,30))
axes= axes.ravel()
n_testing = len(X_test)

for i in np.arange(0,nrows*ncols):
    index = np.random.randint(0,n_testing)
    axes[i].imshow(X_test[index])
    axes[i].set_title(y_test[index], size = 5)
    axes[i].axis('off')

plt.subplots_adjust(hspace=1)

plt.show()

# creating an image matrix for the validation data

nrows =15
ncols = 15

fig,axes = plt.subplots(nrows,ncols,figsize=(30,30))
axes = axes.ravel()
n_validation = len(X_validation)

for i in np.arange(0,nrows*ncols):
    index = np.random.randint(0,n_validation)
    axes[i].imshow(X_validation[index])
    axes[i].set_title(y_validation[index], size =5)
    axes[i].axis('off')

plt.subplots_adjust(hspace=1)
plt.show()

# Data Preparation
X_train,y_train = shuffle(X_train,y_train) # To shuffle the images

# Converting the image from coloured(RGB) to a gray-scale image
X_train_gray =  np.sum(X_train/3, axis=3,keepdims=True) # it means we are averaging the pixel, that is summing the pixel and dividing by 3(RGB)
X_test_gray = np.sum(X_test/3, axis = 3,keepdims=True)
X_validation_gray = np.sum(X_validation/3, axis = 3, keepdims=True)

print('X_train Gray:\n', X_train_gray)
print('X_test gray: \n', X_test_gray)
print('X_Validation gray \n', X_validation_gray)

# Data Normalization
# To normalize this data, we will take the pixel and substrate 128 and divide by 128
# This is to obtain the central values within the pixel and we are going to obtain pixel that ranges
# from -1 to 1

X_train_gray_norm = (X_train_gray-128)/128
X_test_gray_norm = (X_test_gray-128)/128
X_validation_gray_norm = (X_validation_gray-128)/128

# to visualise the train image and we are going the squeeze function to remove the 1 (32,32,1)
# removing the one will remove the invalid shape error
i = 610
plt.imshow(X_train_gray[i].squeeze(), cmap = 'gray') # the gray image
plt.show()

plt.figure()
plt.imshow(X_train[i]) # the original image
plt.show()

plt.figure()
plt.imshow(X_train_gray_norm[i].squeeze(), cmap='gray') # to show the normalized image
plt.show()

# to visualize the test image
i = 200
plt.figure()
plt.imshow(X_test_gray[i].squeeze(), cmap='gray') # the gray image
plt.show()

plt.figure()
plt.imshow(X_test[i]) # the original image
plt.show()

plt.figure()
plt.imshow(X_test_gray_norm[i].squeeze(), cmap='gray')
plt.show()

# to visualize the validation data
i =333
plt.figure()
plt.imshow(X_validation_gray[i].squeeze(), cmap='gray')
plt.show()

plt.figure()
plt.imshow(X_validation[i]) # the original image
plt.show()

plt.figure()
plt.imshow(X_validation_gray_norm[i].squeeze(), cmap='gray')
plt.show()

"""
The LE-NET Deep network consists of the following layers:
STEP 1 : THE FIRST CONVOLUTIONAL LAYER #1
Input = 32x32x1
Output = 28x28x6
Output = (Input-filter+1)/Stride ==> (32-5+1)/1 =28
Used a 5x5 filter with input depth of 3 and output depth of 6
Apply a RELU Activation function to the output
pooling for input, Input =28x28x6 and Output = 14x14x6

Stride is the amount by which the kernel is shifted when the kernel is passed over the 
image

STEP 2: THE SECOND CONVOLUTIONAL LAYERS #2
Input = 14x14x6
Output = 10x10x16
Layer 2: Convolutional layer with Output = 10x10x16
Output = (Input-filter+1)/strides ==> 10 =14-5+1/1
Apply a RELU Activation function to the output
Pooling with input = 10x10x16 and output = 5x5x16

STEP 3 : FLATTENING THE NETWORK
Flatten the network with input = 5x5x16 and output  = 400

STEP 4: FULLY CONNECTED LAYER
Layer 3: Fully connected layer with input = 400 and output = 120

STEP 4 : FULLY CONNECTED LAYER
Layer 3 : Fully connected layer with input = 400 and output = 120
Apply a RELU activation to the output

STEP 5 : ANOTHER FULLY CONNECTED LAYER
Layer 4 : Fully connected layer with input = 120 and output = 84
Apply a RELU activation function to the output


STEP 6 : FULLY CONNECTED LAYER
Layer 5 : Fully connected layer with input = 84 and Output = 43

"""

cnn_model = Sequential()
cnn_model.add(Conv2D(filters=6, kernel_size=(5,5), activation= 'relu', input_shape = (32,32,1)))
cnn_model.add(AveragePooling2D())

cnn_model.add(Conv2D(filters = 16, kernel_size=(5,5), activation='relu'))
cnn_model.add(AveragePooling2D())

cnn_model.add(Flatten())

cnn_model.add(Dense(units=120, activation='relu'))

cnn_model.add(Dense(units=44, activation='relu'))

cnn_model.add(Dense(units=43, activation='softmax'))

cnn_model.compile(loss= 'sparse_categorical_crossentropy', optimizer = Adam(lr=0.001), metrics= ['acc'])

history = cnn_model.fit(X_train_gray_norm, y_train, batch_size= 500, nb_epoch= 5, verbose=1,
              validation_data = (X_validation_gray_norm, y_validation))

# Model Evaluation
score = cnn_model.evaluate(X_test_gray_norm, y_test)
print('Test Accuracy: {}'.format(score[1]))

print(history.history.keys())

accuracy = history.history['acc']
val_accuracy = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

# plotting the above variables versus number of epochs
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label = 'Training Accuracy')
plt.plot(epochs, val_accuracy, 'r', label = 'Validation Accuracy')
plt.title('Training and validation Accuracy')
plt.legend()
plt.show()

plt.figure()
plt.plot(epochs, loss, 'bo', label = 'Loss')
plt.plot(epochs, val_loss, 'r', label = 'Validation Loss')
plt.title('Loss and validation loss')
plt.legend()
plt.show()

# Making predictions
predicted_classes = cnn_model.predict_classes(X_test_gray_norm) # predicted class
y_true = y_test # True class

# using a Confusion Matrix to show the predicted and true classes
cm = confusion_matrix(y_true,predicted_classes)
plt.figure(figsize=(25,25))
sns.heatmap(cm, annot= True)
plt.show()


# plotting the actual image, true label and the predicted label

L = 15
W = 15

fig,axes = plt.subplots(L,W, figsize=(25,25))
axes = axes.ravel() # To flatten the 15x15 matrix to give 225

for i in np.arange(0,L*W): # i ranges from 0 to 244
    axes[i].imshow(X_test[i])
    axes[i].set_title('Prediction = {}\n True ={}'.format(predicted_classes[i],y_true[i]))
    axes[i].axis('off')
plt.subplots_adjust(wspace=1)
plt.show()

