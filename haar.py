#source activate myenv
import numpy as np
import cv2
def algo1(inp,mask):
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    #face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_profileface.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
    print inp
    img = cv2.imread(inp)   #input image variable
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('grayscale',gray)

    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    #cv2.imshow('grayscale2',gray)
    #cv2.imwrite('T1515.jpg',gray)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    print(len(faces))
    mask = np.zeros(img.shape,np.uint8)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        mask[y:y+h,x:x+w] = img[y:y+h,x:x+w]
        lower_black = np.array([1,1,1], dtype = "uint16")
        upper_black = np.array([255,255,255], dtype = "uint16")
        black_mask = cv2.inRange(mask, lower_black, upper_black)
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    #cv2.imshow('img', img)
    #cv2.imshow('roi',roi_color)

    #masking
    #cv2.imshow('mask_final',black_mask)
    cv2.imwrite(mask,black_mask)


    #cv2.waitKey(0)
    #cv2.destroyAllWindows()