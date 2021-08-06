import cv2
import numpy as np
from tensorflow.keras.models import load_model


# 1 PREPARE THE IMAGE
def preProcess(img):
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurImg = cv2.GaussianBlur(grayImg, (5,5), 1)
    imgThreshold = cv2.adaptiveThreshold(blurImg, 255, 1, 1, 11, 2)
    return imgThreshold
    

# 3 FIND BIGGEST CONTOUR
def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area > 50:
            parameter = cv2.arcLength(c, True)
            corners = cv2.approxPolyDP(c, 0.02 * parameter , True)
            if area > max_area and len(corners) == 4: #if there are 4 conrners then we found a rectangle or square
                biggest = corners
                max_area = area
    return biggest, max_area


# 3 REORDER POINTS FOR WARP PERSPECTIVE
def reorder(myPoints):
    myPoints = myPoints.reshape((4,2))
    myPointsNew = np.zeros((4,1,2), dtype = np.int32)
    
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    
    diff = np.diff(myPoints, axis = 1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    
    return myPointsNew
  
# 4 TO SPLIT THE IMAGE  
def splitBoxes(img):
    rows = np.vsplit(img,9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r,9)
        for box in cols:
            boxes.append(box)
    return boxes

def getPrediction(boxes, model):
    result = []
    for image in boxes:
        img = np.asarray(image)
        img = img [4:img.shape[0]-4,4:img.shape[1]-4]
        img = cv2.resize(img, (28,28))
        img = img/255
        img = img.reshape(1,28,28,1)
        ## GET PREDICTION
        predictions = model.predict(img)
        classIndex = np.argmax(predictions,axis=-1)
        probabilityValue = np.amax(predictions)
        print(classIndex, probabilityValue)
        ## SAVE TO RESULT
        if probabilityValue > 0.5:
            result.append(classIndex[0])
        else:
            result.append(0)
    return result

    
def initializePredictionModel():
    model = load_model("digitTesseract/myModel.h5")
    return model

def stackImages(imgArray, scale):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver