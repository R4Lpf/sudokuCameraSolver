import numpy as np
import cv2
import pickle
from tensorflow.keras.models import load_model


########### PARAMETERS ##############
width = 640
height = 480
threshold = 0.8 # MINIMUM PROBABILITY TO CLASSIFY
cameraNo = 1
#####################################

cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

model = load_model("myModel10.h5")

def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

while(True):
    success, imgOriginal = cap.read()
    img = np.asarray(imgOriginal)
    img = cv2.resize(img,(32,32))
    img = preProcessing(img)
    #cv2.imshow("Processed Image",img)
    img = img.reshape(1,32,32,1)
    #Predict
    classIndex = int(model.predict_classes(img))
    #print(classIndex)
    predictions = model.predict(img)
    probVal = np.amax(predictions)
    print(classIndex,probVal)
    
    if probVal>threshold:
        cv2.putText(imgOriginal,str(classIndex) + " " +str(probVal),
                    (50,50), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,1, (0,255,0),1)
    
    cv2.imshow("Original image", imgOriginal)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()