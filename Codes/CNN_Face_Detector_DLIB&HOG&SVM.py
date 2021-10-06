
##########################################

#   Very slow on CPU

##########################################

import cv2
import time
import dlib
import os
from keras.preprocessing import image
import numpy as np
import joblib
from skimage.feature import hog

####  HOG parameters  ####
ppc =8  #Size (in pixels) of a cell
cb=4    #Number of cells in each block.

#### List of names  
list_names = ['unknown','wacef']


target_size = (300,300)   #####because of the trained model
confThreshold = 0.6

#Model_prediction
svm_from_joblib = joblib.load('C:/Users/hp/Desktop/Travail_Stage/HOG_SVM_Model/svm_model.pkl')

#Detector
dnnFaceDetector = dlib.cnn_face_detection_model_v1("./mmod_human_face_detector.dat")

################# This variable determine if we are saving images or not  
taked = False
################
path_bd = 'C:/Users/hp/Desktop/Travail_Stage/HOG_SVM_Model/BD_Faces'


cam = cv2.VideoCapture(0)
pTime = 0


while True:
    ret, frame = cam.read()
    
    cols = frame.shape[1]
    rows = frame.shape[0]  
    img = cv2.resize(frame, target_size)
    
    faceRects = dnnFaceDetector(img, 0)

    for faceRect in faceRects:
        x1 = faceRect.rect.left()
        y1 = faceRect.rect.top()
        x2 = faceRect.rect.right()
        y2 = faceRect.rect.bottom()
        xLeftBottom = int(x1 * cols)
        yLeftBottom = int(y1 * rows)
        xRightTop = int(x2 * cols)
        yRightTop = int(y2 * rows)
        cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop), (0, 0, 255))

        ##
        w = xRightTop - xLeftBottom
        h = yRightTop - yLeftBottom
        timage = frame[yLeftBottom:yLeftBottom+h,xLeftBottom:xLeftBottom+w]
        if timage is None:
            continue
        else:
            timage = cv2.resize(timage, (224,224))
            #############  Saving Faces  ################## 
            if taked : 
                path_img = path_bd + '/' + str(sv) 
                sv -= 1 
                cv2.imwrite(path_img, timage)
                if sv == 1 :
                    taked = False
            ####################################
            else:
                image = cv2.cvtColor(timage , cv2.COLOR_BGR2GRAY)
                image= np.array(image)
                
                fd , hogim = hog(image , orientations=9 , pixels_per_cell=(ppc , ppc) , block_norm='L2' , cells_per_block=(cb,cb) , visualize=True)

                fd = np.reshape(fd,(1, -1))
                #print(fd.shape)  ==>expected (1,90000)

                res = svm_from_joblib.predict(fd)
                kn = res[0][0]
                if kn != 0 : 
                    namep = list_names[kn]
                    cv2.putText(frame, namep, (200, 100), cv2.FONT_HERSHEY_PLAIN,3, (0, 255, 0), 3)
                elif kn == 0 : 
                    sv = input('Unknown person , would you save this person? if yes enter the number of images else 0')
                    if sv != 0 :
                        namee = input('Enter the name of this person to save him')
                        list_names.append(namee)
                        nb = len(list_names) 
                        taked = True
                        path1 = path_bd + '/' + nb 
                        os.mkdir(path1)

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