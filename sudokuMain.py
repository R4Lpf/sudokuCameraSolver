import os
from utilities import *
import sudokuSolver

pathImg = "assets/4.jpg"
heightImg = 450 # MUST BE A MULTIPLE OF 9 FOR THE VSPLIT IN PART 4
widthImg = 450 # MUST BE A MULTIPLE OF 9 FOR THE VSPLIT IN PART 4
model = initializePredictionModel()

# 1 PREPARE THE IMAGE
img = cv2.imread(pathImg)
img = cv2.resize(img,(widthImg,heightImg))
imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)
imgThreshold = preProcess(img)


# 2 FIND CORNERS
imgContours = img.copy()
imgBigContour = img.copy()
contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3)
 
# 3 FIND THE BIGGEST COUNTOUR
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

    # 4 SPLITTING THE IMAGE AND GET EACH BOX WITH DIGITS
    
    imgSolvedDigits = imgBlank.copy()
    boxes = splitBoxes(imgWarpColored)
    print(len(boxes))
    #cv2.imshow("sample",boxes[0])
    numbers = getPrediction(boxes,model)
    print(numbers)
    imgDetectedDigits = displayNumbers(imgDetectedDigits, numbers, color = (0,255,0))
    numbers = np.asarray(numbers)
    posArray = np.where(numbers > 0, 0, 1)
    
    # 5 FIND SOLUTION
    
    grid = np.array_split(numbers,9)
    try:
        sudokuSolver.solve(0,0,grid)
    except:
        pass
    flatList = []
    for sublist in grid:
        for item in sublist:
            flatList.append(item)
    solvedNumbers = flatList*posArray
    imgSolvedDigits = displayNumbers(imgSolvedDigits,solvedNumbers)
    
    print(grid)
    
    
    # 6 OVERLAY SOLUTION
    points2 = np.float32(biggest)
    points1 = np.float32([[0,0], [widthImg,0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(points1,points2)
    imgInvWarpColored = img.copy()
    imgInvWarpColored = cv2.warpPerspective(imgSolvedDigits, matrix, (widthImg, heightImg))
    inv_perspective = cv2.addWeighted(imgInvWarpColored, 1, img, 0.5, 1)
    imgDetectedDigits = drawGrid(imgDetectedDigits)
    imgSolvedDigits = drawGrid(imgSolvedDigits)
    
    imgArray = ([img, imgThreshold, imgContours, imgBigContour ],[imgDetectedDigits, imgSolvedDigits, imgInvWarpColored, inv_perspective])
    
    stackedImage = stackImages (imgArray, 1)
    cv2.imshow("Stacked Images", stackedImage)

else:
    print("No Sudoku Found")
    
cv2.waitKey(0)

