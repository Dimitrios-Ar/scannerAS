# USAGE
# **to use already saved image**
# python scan_image_or_android.py image
# **to use already saved image and see the steps**
# python scan_image_or_android.py image steps

# **to use android phone camera**
# python scan_image_or_android.py camera
# **to use android phone camera and see the steps**
# python scan_image_or_android.py camera steps

from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import cv2
import imutils
import sys
import requests

stop = False
imageDone = False
imageType = str(sys.argv[1])
imageNaming = 0
showSteps = False
correctArgument = False

#Checking if the steps will be shown or not
while correctArgument == False:
    if len(sys.argv) == 3:
        if sys.argv[2] == "steps":
            showSteps = True
            correctArgument = True
        else:
            stepsAnswer = str(input('Do you want to see the steps? (y/n)'))
            if stepsAnswer == 'y':
                showSteps = True
                correctArgument = True
            elif stepsAnswer == 'n':
                showSteps = False
                correctArgument = True
    else:
        correctArgument = True

while stop == False:
    if imageType == "camera": #Usage for android phone
        url = "http://192.168.43.1:8080/shot.jpg"#URL FOR HOTSPOT
        #url = "http://192.168.1.5:8080//shot.jpg"#URL FOR WIFI
        while imageDone == False:
            img_resp = requests.get(url)
            img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
            image = cv2.imdecode(img_arr, -1)
            image_copy = imutils.resize(image, height = 800)
            cv2.putText(image_copy, "Press 's' when ready to scan", (250, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(image_copy, "Press 'q' to quit", (250, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("androidCam", image_copy)
            k = cv2.waitKey(1)
            if k%256 == 115: #only proceed when hitting "s"
                imageDone = True
                break
            elif k%256 == 113: #if "q" then program stops
                sys.exit()
    elif imageType == "image": #usage for image
        askForImage = str(input('Please select image or type "q" to quit:'))
        if askForImage == "q":
            sys.exit()
        else:
            try:
                image = open(askForImage, 'r')
                image = cv2.imread(askForImage)
                imageDone = True
            except FileNotFoundError:
                imageDone = False
                print('The image does not exist. Aborting..')

    #After finding if an image of the android cameram will be used then we continue
    if imageDone == True:
        ratio = image.shape[0] / 500
        orig = image.copy()
        image = imutils.resize(image, height = 500)

        #STEP 1: Edge Detection
        # convert the image to grayscale, blur it, and find edges in the image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray_blurred, 75, 200)

        if showSteps == True:
            # show the original image and the edge detected image
            print("STEP 1: Edge Detection")
            cv2.imshow("Original Image", image)
            cv2.imshow("Edged Image", edged)
            cv2.imwrite('edged.png',edged)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        #STEP 2: Corner detection
        # find the contours using the edged image
        _, cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #By RETR_EXTERNAL we keep only the outer contours.
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
        temp_image = image.copy()
        # for every contour (object)
        for c in cnts:
            imageNaming +=1
            ## compute the center of the contour
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # draw the contour and center of the shape on the image
            cv2.drawContours(temp_image, [c], -1, (0, 0, 255), 2)
            # set epsilon(ε) and approximate the contour using Ramer-Douglas-Peucker algorithm
            epsilon = 0.02 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c,epsilon, True)

            # if the approximated contour has four points, then we assume it's a document
            if len(approx) == 4:
                print('Number of points: ', len(c))
                # draw the corners of the document
                cv2.drawContours(temp_image, approx, -1, (0, 255, 0), 5)
                if showSteps == True:
                    #(a) show all the contours (objects) with red outline and (b) define those that are documents
                    print("STEP 2: Find contours of paper")
                    cv2.circle(temp_image, (cX, cY), 3, (0, 0, 0), -1)
                    cv2.putText(temp_image, "Document "+ str(imageNaming), (cX - 20, cY - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    cv2.imshow("Image With Contour and Corners", temp_image)
                    cv2.imwrite('Contour.png',temp_image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                #STEP 3: Applying transform to obtain a top-down view of the original image
                transformedImage = four_point_transform(orig, approx.reshape(4, 2) * ratio)
                cv2.imwrite('transformed.png',transformedImage)
                # convert the transformed image to grayscale, then perform adaptive threshold
                grayTransformedImage = cv2.cvtColor(transformedImage, cv2.COLOR_BGR2GRAY)
                cv2.imwrite('grayTransformed.png',grayTransformedImage)
                T = threshold_local(grayTransformedImage, 11, offset = 10, method = "gaussian")
                cv2.imwrite('threshold.png',T)
                scannedImage = (grayTransformedImage > T).astype("uint8") * 255

                # show the transformed and scanned images
                if showSteps == True:
                    print("STEP 3: Applying transform")
                    cv2.imshow("Transformed Image", imutils.resize(transformedImage, height = 500))
                    cv2.imshow("Scanned Image", imutils.resize(scannedImage, height = 500))
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                cv2.imwrite('scanned'+str(imageNaming)+'.png', scannedImage)

    imageDone = False #recheck for new image
