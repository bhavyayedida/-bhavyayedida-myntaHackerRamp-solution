from re import I
import numpy as np
import cv2
from sklearn.cluster import KMeans
from collections import Counter
import imutils
import pprint
from matplotlib import pyplot as plt
import math
import pandas as pd
from functools import wraps
import sqlite3
import dlib

from flask import *
import os
app = Flask(__name__)
@app.route("/")
def home():  
    return render_template("mainpage.html")
@app.route("/first")
def first():
    return render_template("mainpage.html")
@app.route("/second")
def second():
    return render_template("module1.html")  
@app.route("/lipstick")
def lipstick():
    return render_template("lipstick.html")  
@app.route("/blush")
def blush():
    return render_template("blush.html")  
@app.route("/eyeshadow")
def eyeshadow():
    return render_template("eyeshadow.html")  
@app.route("/virtual")
def virtual():
    return render_template("virtual.html")  


uploads_dir = os.path.join(app.instance_path, 'C:/Users/madhuri/Desktop/flaskapp/static')
app.config['UPLOAD_FOLDER'] = uploads_dir

@app.route("/success",methods = ['POST',"GET"])

        
def success():
    
    output = request.form.to_dict()
    event = output["name"]
    f = request.files["file"] 
    f.filename = "temp.jpg"  #some custom file name that you want
    f.save(os.path.join(uploads_dir, f.filename))
    excel_file = 'data.xlsx'
    df = pd.read_excel(excel_file,engine='openpyxl')
    def extractSkin(image):
        # Taking a copy of the image
        img = image.copy()
        # Converting from BGR Colours Space to HSV
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Defining HSV Threadholds
        lower_threshold = np.array([0, 48, 80], dtype=np.uint8)         #dark
        upper_threshold = np.array([187, 207, 255], dtype=np.uint8)      #fair

        # Single Channel mask,denoting presence of colours in the about threshold
        skinMask = cv2.inRange(img, lower_threshold, upper_threshold)

        # Cleaning up mask using Gaussian Filter
        skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)

        # Extracting skin from the threshold mask
        skin = cv2.bitwise_and(img, img, mask=skinMask)

        # Return the Skin image
        return cv2.cvtColor(skin, cv2.COLOR_HSV2BGR)


    def removeBlack(estimator_labels, estimator_cluster):

        # Check for black
        hasBlack = False

        # Get the total number of occurance for each color
        occurance_counter = Counter(estimator_labels)

        # Quick lambda function to compare to lists
        def compare(x, y): return Counter(x) == Counter(y)

        # Loop through the most common occuring color
        for x in occurance_counter.most_common(len(estimator_cluster)):

            # Quick List comprehension to convert each of RBG Numbers to int
            color = [int(i) for i in estimator_cluster[x[0]].tolist()]

            # Check if the color is [0,0,0] that if it is black
            if compare(color, [0, 0, 0]) == True:
                # delete the occurance
                del occurance_counter[x[0]]
                # remove the cluster
                hasBlack = True
                estimator_cluster = np.delete(estimator_cluster, x[0], 0)
                break

        return (occurance_counter, estimator_cluster, hasBlack)


    def getColorInformation(estimator_labels, estimator_cluster, hasThresholding=False):

        # Variable to keep count of the occurance of each color predicted
        occurance_counter = None

        # Output list variable to return
        colorInformation = []

        # Check for Black
        hasBlack = False

        # If a mask has be applied, remove th black
        if hasThresholding == True:

            (occurance, cluster, black) = removeBlack(
                estimator_labels, estimator_cluster)
            occurance_counter = occurance
            estimator_cluster = cluster
            hasBlack = black

        else:
            occurance_counter = Counter(estimator_labels)

        # Get the total sum of all the predicted occurances
        totalOccurance = sum(occurance_counter.values())

        # Loop through all the predicted colors
        for x in occurance_counter.most_common(len(estimator_cluster)):

            index = (int(x[0]))

            # Quick fix for index out of bound when there is no threshold
            index = (index-1) if ((hasThresholding & hasBlack)
                                & (int(index) != 0)) else index

            # Get the color number into a list
            color = estimator_cluster[index].tolist()

            # Get the percentage of each color
            color_percentage = (x[1]/totalOccurance)

            # make the dictionay of the information
            colorInfo = {"cluster_index": index, "color": color,
                        "color_percentage": color_percentage}

            # Add the dictionary to the list
            colorInformation.append(colorInfo)

        return colorInformation


    def extractDominantColor(image, number_of_colors=5, hasThresholding=False):

        # Quick Fix Increase cluster counter to neglect the black(Read Article)
        if hasThresholding == True:
            number_of_colors += 1

        # Taking Copy of the image
        img = image.copy()

        # Convert Image into RGB Colours Space
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Reshape Image
        img = img.reshape((img.shape[0]*img.shape[1]), 3)

        # Initiate KMeans Object
        estimator = KMeans(n_clusters=number_of_colors, random_state=0)

        # Fit the image
        estimator.fit(img)

        # Get Colour Information
        colorInformation = getColorInformation(
            estimator.labels_, estimator.cluster_centers_, hasThresholding)
        return colorInformation

    """## Section Two.4.2 : Putting it All together: Pretty Print

    The function makes print out the color information in a readable manner
    """


    def prety_print_data(color_info):
        global link
        r = color_info[0]['color'][0]
        tone =""
        if(r>217 and r<=255):
            tone = "fair"
        elif(r>190 and r<=217):
            tone="medium"
        elif(r>120 and r<=190):
            tone="brown"
        elif(r>=80 and r<=120):
            tone="dark"
        print(tone)
        makeup = df['suggestions'].where(df['Tone']==tone).where(df['occasion']==event)
        makeup = makeup.dropna()
        link = makeup.to_string(index=False)
        print(link)

        #mainpath = r'{}'.format(link)
     


    # Get Image from URL. If you want to upload an image file and use that comment the below code and replace with  image=cv2.imread("FILE_NAME")

    path = r'C:/Users/madhuri/Desktop/flaskapp/static/temp.jpg'
    img_src = cv2.imread(path)
    skin = extractSkin(img_src)

    # Find the dominant color. Default is 1 , pass the parameter 'number_of_colors=N' where N is the specified number of colors
    dominantColors = extractDominantColor(skin, hasThresholding=True)
    # Show in the dominant color information
    print("Color Information")
    prety_print_data(dominantColors)
    print("link=",link)
    return render_template("success.html",name = event,image=f.filename, link=link)

@app.route("/virtualtryon",methods = ['POST',"GET"])
def virtualtryon():
    output = request.form.to_dict()
    color = output["name1"]
    f = request.files["file1"] 
    f.filename = "temp1.jpg"  #some custom file name that you want
    f.save(os.path.join(uploads_dir, f.filename))
    img = cv2.imread('C:/Users/madhuri/Desktop/flaskapp/static/temp1.jpg')
    img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
    imgOriginal = img.copy()
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
    print(red,green,blue)

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
        path1 = 'C:/Users/madhuri/Desktop/flaskapp/static'
        cv2.imwrite(os.path.join(path1,'colored.jpg'), imgColorlips)
        return render_template("virtualtryon.html",color = color,link="C:/Users/madhuri/Desktop/flaskapp/static/colored.jpg")


if __name__=='__main__':
    app.run(debug= True,port=5001)