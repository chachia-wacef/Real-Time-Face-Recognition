import cv2
import time
from keras.models import load_model
from keras.preprocessing import image
import numpy as np


target_size = (300,300)   #####because of the trained model
confThreshold = 0.7

#Model_prediction
model = load_model('C:/Users/hp/Desktop/Travail_Stage/Res10SSD_VGG16/best_model.h5')

#Res10 SSD (Single Shot Detectors)
detector = cv2.dnn.readNetFromCaffe("C:/Users/hp/Desktop/Travail_Stage/Res10_SSD/deploy.prototxt","C:/Users/hp/Desktop/Travail_Stage/Res10_SSD/res10_300x300_ssd_iter_140000.caffemodel")

#cam = cv2.VideoCapture('rtsp://192.168.1.110:554/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream')
cam = cv2.VideoCapture(0)

if (cam.isOpened() == True):   
    print('Yes')
else: 
    print('No')   

pTime = 0

while True:
    ret, frame = cam.read()
    
    cols = frame.shape[1]
    rows = frame.shape[0]  
    img = cv2.resize(frame, target_size)
    

    imageBlob = cv2.dnn.blobFromImage(image = img)
    detector.setInput(imageBlob)
    detections = detector.forward()


    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confThreshold:
            xLeftBottom = int(detections[0, 0, i, 3] * cols)
            yLeftBottom = int(detections[0, 0, i, 4] * rows)
            xRightTop = int(detections[0, 0, i, 5] * cols)
            yRightTop = int(detections[0, 0, i, 6] * rows)
            cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop), (0, 0, 255))

            ##
            w = xRightTop - xLeftBottom
            h = yRightTop - yLeftBottom
            timage = frame[yLeftBottom:yLeftBottom+h,xLeftBottom:xLeftBottom+w]
            if timage is None:
                continue
            else:
                timage = cv2.resize(timage, (224,224))
                timage= image.img_to_array(timage)
                timg= np.expand_dims(timage,axis=0)
                result= model.predict(timg)
                print(result)
                cv2.putText(frame, str(result), (200, 100), cv2.FONT_HERSHEY_PLAIN,3, (0, 255, 0), 3)
                ##


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN,3, (255, 0, 0), 3)
    


    cv2.imshow('image',frame)    
    k = cv2.waitKey(1)
    if k == ord('q'):
        print("Closing the window")
        break
    

cam.release()

cv2.destroyAllWindows()