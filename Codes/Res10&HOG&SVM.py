import cv2
import time
from keras.preprocessing import image
import numpy as np
import joblib
from skimage.feature import hog

####  HOG parameters  ####
ppc =8  #Size (in pixels) of a cell
cb=4    #Number of cells in each block.

target_size = (300,300)   #####because of the trained model
confThreshold = 0.6

#Model_prediction
svm_from_joblib = joblib.load('C:/Users/hp/Desktop/Travail_Stage/HOG_SVM_Model/svm_model1.pkl')

#Res10 SSD (Single Shot Detectors) ((  DNN Face Detector in OpenCV  ))
detector = cv2.dnn.readNetFromCaffe("C:/Users/hp/Desktop/Travail_Stage/Res10_SSD/deploy.prototxt","C:/Users/hp/Desktop/Travail_Stage/Res10_SSD/res10_300x300_ssd_iter_140000.caffemodel")

#cam = cv2.VideoCapture('rtsp://192.168.1.110:554/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream')
cam = cv2.VideoCapture(0)

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
                image = cv2.cvtColor(timage , cv2.COLOR_BGR2GRAY)
                image= np.array(image)
                
                fd , hogim = hog(image , orientations=9 , pixels_per_cell=(ppc , ppc) , block_norm='L2' , cells_per_block=(cb,cb) , visualize=True)

                fd = np.reshape(fd,(1, -1))
                    #print(fd.shape)  ==>expected (1,90000)

                res = svm_from_joblib.predict(fd)
                
                cv2.putText(frame, str(res), (200, 100), cv2.FONT_HERSHEY_PLAIN,3, (0, 255, 0), 3)
                
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