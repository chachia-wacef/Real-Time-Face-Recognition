import dlib
import face_recognition
import cv2
from PIL import Image, ImageDraw
import numpy as np

# Load a sample picture and learn how to recognize it.
wacef_image = face_recognition.load_image_file("C:/Users/hp/Desktop/Travail_Stage/Faces_data/faces/F1/F16.jpg")
wacef_face_encoding = face_recognition.face_encodings(wacef_image)[0]
# Create arrays of known face encodings and their names
known_face_encodings = [
    wacef_face_encoding
    ]
known_face_names = [
    "Wacef Chachia"
]

cam = cv2.VideoCapture(0)
while True:
    ret, frame = cam.read()
  

    # Load an image with an unknown face
    #unknown_image = face_recognition.load_image_file("C:/Users/hp/Desktop/Travail_Stage/Faces_data/faces/F1/F24.jpg")

    # Find all the faces and face encodings in the unknown image
    
    face_locations = face_recognition.face_locations(frame) ######## model='cnn' with GPU #######
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
    # See http://pillow.readthedocs.io/ for more about PIL/Pillow
    pil_image = Image.fromarray(frame)
    # Create a Pillow ImageDraw Draw instance to draw with
    draw = ImageDraw.Draw(pil_image)

    # Loop through each face found in the unknown image
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding) #tolerance = 0.6 per default

        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            #print(name)
        else:
            #show the face which we talk about
            draw1 = draw.copy()
            draw1.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
            cv2.imshow('Face detected',draw1)
            #
            rep = input('Unknown person !. Do you want to save it ? y : yes or n : no')
            while rep not in ['y','n']:
                rep = input('Unknown person !. Do you want to save it ? y : yes or n : no')
            if rep == 'y':
                name = input('So enter his name here : ')
                known_face_names.append(name)
                known_face_encodings.append(face_encoding)
            cv2.destroyWindow('Face detected')
        

        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))
    


    # Remove the drawing library from memory as per the Pillow docs
    del draw

    # Display the resulting image
    pil_image.show()

    # You can also save a copy of the new image to disk if you want by uncommenting this line
    if name != "Unknown":
        img_name = name + '.jpg'
        pil_image.save(img_name)
        
    #Show final result
    cv2.imshow('image',draw)    
    ##Close window
    k = cv2.waitKey(1)
    if k == ord('q'):
        print("Closing the window")
        break

cam.release()

cv2.destroyAllWindows()
