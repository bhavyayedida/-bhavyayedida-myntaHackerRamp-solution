import pandas as pd
import cv2
import numpy as np
import dlib


img = cv2.imread('static/image1.jpg')
img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
imgOriginal = img.copy()

color = "Gemstone"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def createBox(img, points, scale = 5, masked = False, cropped = False):
    if masked:
        mask = np.zeros_like(img)
        mask = cv2.fillPoly(mask, [points], (255, 255, 255))
        img = cv2.bitwise_and(img, mask)
        #cv2.imshow("mask", img)
    if cropped:
        bbox = cv2.boundingRect(points)
        x, y, w, h = bbox
        imgCrop = img[y:y+h, x:x+w]
        imgCrop = cv2.resize(imgCrop, (0, 0), None, scale, scale)
        return imgCrop
    else:
        return mask


imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
faces = detector(imgGray)

excel_file = 'myntracolours_dataset.xls'
df = pd.read_excel(excel_file)
r = (df['R'].where(df['color'] == color)).dropna()
g = (df['G'].where(df['color'] == color)).dropna()
b = (df['B'].where(df['color'] == color)).dropna()
r = r.to_string(index=False)
b = b.to_string(index=False)
g = g.to_string(index=False)
red = float(r)
green = float(g)
blue = float(b)
print(red, green, blue)


for face in faces:
    x1, y1 = face.left(), face.top()
    x2, y2 = face.right(), face.bottom()
    #imgOriginal = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    landmarks = predictor(imgGray, face)
    myPoints = []
    for n in range(68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        myPoints.append([x, y])

    myPoints = np.array(myPoints) 
    imglips = createBox(img, myPoints[48:61], 3, masked = True, cropped = False)  


    imgColorlips = np.zeros_like(imglips)
    imgColorlips[:] = blue, green, red #BGR 255,192,203
    imgColorlips = cv2.bitwise_and(imgColorlips, imglips)
    imgColorlips = cv2.GaussianBlur(imgColorlips, (7, 7), 10)
    imgColorlips = cv2.addWeighted(imgOriginal, 1, imgColorlips, 0.4, 1)
    cv2.imshow('Colored', imgColorlips)
    cv2.imwrite('colored.jpg', imgColorlips)
    

cv2.imshow("Original", imgOriginal)
cv2.waitKey(0)