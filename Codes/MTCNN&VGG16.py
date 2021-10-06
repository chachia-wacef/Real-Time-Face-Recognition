import cv2
import time
from mtcnn import MTCNN
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

#Parameters 
#target_size = (300,300)

#face detector
detector = MTCNN()

#Model_prediction
model = load_model('C:/Users/hp/Desktop/Travail_Stage/Res10SSD_VGG16/best_model.h5')

cam = cv2.VideoCapture(0)

pTime = 0

while True:
    ret, frame = cam.read()
    #cols = frame.shape[1]
    #rows = frame.shape[0]  
    #img = cv2.resize(frame, target_size)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    faces = detector.detect_faces(img, 1.1, 4)
    for (x,y,w,h) in faces:
        timage = frame[y:y+h,x:x+w]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        if timage is None:
            continue
        else:
            timage = cv2.resize(timage, (224,224))
            timage= image.img_to_array(timage)
            timg= np.expand_dims(timage,axis=0)
            result= model.predict(timg)
            print(result)
            cv2.putText(frame, str(result), (200, 100), cv2.FONT_HERSHEY_PLAIN,3, (0, 255, 0), 3)
        
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN,3, (255, 0, 0), 3)
    
    cv2.imshow('image',frame)    

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Closing it")
        break
    

cam.release()

cv2.destroyAllWindows()