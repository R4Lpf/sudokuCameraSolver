# CNN_TRAINING
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

import pickle
###################################################################################
path = "myData"
testRatio = 0.2
valRatio = 0.2
imageDimensions = (32,32,3)

batchSizeVal = 50
epochsVal = 20
stepsForEpochVal = 2000

###################################################################################
images = []
classN = []
myList = os.listdir(path)
nOfClasses = len(myList)

for x in range(nOfClasses):
    myPicList = os.listdir(path+"/"+str(x))
    for y in myPicList:
        curImg = cv2.imread(path+"/"+str(x)+"/"+y)
        curImg = cv2.resize(curImg, (imageDimensions[0],imageDimensions[1]))
        images.append(curImg)
        classN.append(x)
    print(x,end = " ")
print(" ")
print("Total images in Images List = " + str(len(images)))
print("Total IDs in ClassN List = " + str(len(classN)))

images = np.array(images)
classN = np.array(classN)

print(images.shape)

# SPLITTING THE DATA
X_train, X_test, y_train, y_test  = train_test_split(images, classN, test_size = testRatio) 
X_train, X_validation, y_train, y_validation  = train_test_split(X_train, y_train, test_size = valRatio) 

print(np.where(y_train==0))

print(X_train.shape)
print(X_test.shape)
print(X_validation.shape)

nOfSamples =  []

for x in range(nOfClasses):
    #print(len(np.where(y_train==0)[0]))
    nOfSamples.append(len(np.where(y_train==x)[0]))

print(nOfSamples)

plt.figure(figsize=(10,5))
plt.bar(range(0,nOfClasses), nOfSamples)
plt.title("N of images for each Class")
plt.xlabel("ClassID")
plt.ylabel("Number of Images")
plt.show()


def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

# =============================================================================
# img = preProcessing(X_train[30])
# img = cv2.resize(img,(300,300))
# cv2.imshow("PreProcessed",img)
# cv2.waitKey(0)
# =============================================================================
    
X_train = np.array(list((map(preProcessing,X_train))))
X_test = np.array(list((map(preProcessing,X_test))))
X_validation = np.array(list((map(preProcessing,X_validation))))


X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)
X_validation = X_validation.reshape(X_validation.shape[0],X_validation.shape[1],X_validation.shape[2],1)


dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)


dataGen.fit(X_train)


y_train = to_categorical(y_train,nOfClasses)
y_test = to_categorical(y_test,nOfClasses)
y_validation = to_categorical(y_validation,nOfClasses)

def myModel():
    nOfFilters = 60
    sizeOfFilter1 = (5,5)
    sizeOfFilter2 = (3,3)
    sizeOfPool = (2,2)
    nOfNodes = 500
    
    model = Sequential()
    model.add((Conv2D(nOfFilters,
                       sizeOfFilter1,
                       input_shape=(imageDimensions[0],imageDimensions[1],1),
                       activation="relu")))
    model.add((Conv2D(nOfFilters,sizeOfFilter1, activation="relu")))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add((Conv2D(nOfFilters//2,sizeOfFilter2, activation="relu")))
    model.add((Conv2D(nOfFilters//2,sizeOfFilter2, activation="relu")))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(nOfNodes, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(nOfClasses, activation = "softmax"))
    model.compile(Adam(learning_rate=0.001),loss = "categorical_crossentropy", metrics = ["accuracy"])
    
    return model

model = myModel()
print(model.summary())

history = model.fit_generator(dataGen.flow(X_train,y_train,
                                batch_size=batchSizeVal),
                                steps_per_epoch=len(X_train)//batchSizeVal, #changed to this to make the epoch training work
                                epochs=epochsVal,
                                validation_data = (X_validation, y_validation),
                                shuffle = 1)

plt.figure(1)
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.legend(["training","validation"])
plt.title("Loss")
plt.xlabel("epochs")
plt.figure(2)
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.legend(["training","validation"])
plt.title("Accuracy")
plt.xlabel("epochs")
plt.show()
score = model.evaluate(X_test,y_test,verbose = 0)
print("Test Score = ",score[0])
print("Test Accuracy = ",score[1])


model = myModel()
model.save("myModel.h5",save_format='h5')
# =============================================================================
# pickle_out = open("model_trained.p","wb")
# pickle.dump(model,pickle_out)
# pickle_out.close()
# =============================================================================
