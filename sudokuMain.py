import os
from utilities import *
import sudokuSolver

pathImg = "assets/4.jpg"
heightImg = 330
widthImg = 330


#PREPARE THE IMAGE
img = cv2.imread(pathImg)
img = cv2.resize(img,(widthImg,heightImg))
imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)
imgThreshold = preProcess(img)


# FIND CORNERS
imgContours = img.copy()
imgBigContour = img.copy()
contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3)
 
# FIND THE BIGGEST COUNTOUR
biggest, maxArea = biggestContour(contours)
if biggest.size != 0:
    biggest = reorder(biggest)
    cv2.drawContours(imgBigContour, biggest, -1, (0, 0, 255), 10)
    points1 = np.float32(biggest)
    points2 = np.float32([[0,0], [widthImg,0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(points1, points2)
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    imgDetectedDigits = imgBlank.copy()
    imgWarpColored = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)

# SPLITTING THE IMAGE AND GET EACH BOX WITH DIGITS









imgArray = ([img, imgThreshold, imgContours, imgBigContour ],[imgWarpColored, imgBlank, imgBlank, imgBlank])

stackedImage = stackImages (imgArray, 1)
cv2.imshow("Stacked Images", stackedImage)

cv2.waitKey(0)

