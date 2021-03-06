import cv2
import numpy as np
from tensorflow.keras.models import load_model


# 1 PREPARE THE IMAGE
def pre_process_img(img):
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurImg = cv2.GaussianBlur(grayImg, (5,5), 1)
    imgThreshold = cv2.adaptiveThreshold(blurImg, 255, 1, 1, 11, 2)
    return imgThreshold
    

# 3 FIND BIGGEST CONTOUR
def find_biggest_contour(contours):
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
def split_sudoku_sections(img):
    rows = np.vsplit(img,9)
    print(rows)
    boxes = []
    for r in rows:
        cols = np.hsplit(r,9)
        print(cols)
        for box in cols:
            boxes.append(box)
    return boxes

def get_prediction(boxes, model):
    result = []
    for image in boxes:
        img = np.asarray(image)
        img = img [4:img.shape[0]-4,4:img.shape[1]-4]
        img = cv2.resize(img, (32,32))
        img = img/255
        img = img.reshape(1,32,32,1)
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

    
def initialize_prediction_model(modelPath):
    model = load_model(modelPath)
    return model


def display_numbers(img, numbers, color = (255,100,0)):
    secW = int(img.shape[1]/9)
    secH = int(img.shape[1]/9)
    for x in range(9):
        for y in range(9):
            if numbers[(y*9)+x] != 0:
                cv2.putText(img, str(numbers[(y*9)+x]),
                            (x*secW+int(secW/2)-10, int((y+0.8)*secH)),
                            cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    
    return img

def draw_grid(img):
    secW = int(img.shape[1]/9)
    secH = int(img.shape[0]/9)
    for i in range(9):
        pt1 = (0,secH*i)
        pt2 = (img.shape[1], secH*i)
        pt3 = (secW*i,0)
        pt4 = (secW*i, img.shape[0])
        cv2.line(img, pt1,pt2,(230,230,0),2)
        cv2.line(img, pt3,pt4,(230,230,0),2)
    return img
    
    
def stack_images(imgArray, scale):
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