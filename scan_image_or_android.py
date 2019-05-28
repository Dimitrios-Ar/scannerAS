# USAGE
# **to use already saved image**
# python scan_image_or_android.py image
# **to use already saved image and see the steps**
# python scan_image_or_android.py image steps


# **to use android phone camera**
# python scan_image_or_android.py camera
# python scan_image_or_android.py camera steps


# import the necessary packages
from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import cv2
import imutils
import sys
import requests

stop = False
imageDone = False
type = str(sys.argv[1])
counter = 0
showSteps = False
correctArgument = False

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
    if type == "camera":
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
    elif type == "image":
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

    if imageDone == True:
        ratio = image.shape[0] / 500
        orig = image.copy()
        image = imutils.resize(image, height = 500)

        # convert the image to grayscale, blur it, and find edges in the image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 75, 200)


        # show the original image and the edge detected image
        if showSteps == True:
            print("STEP 1: Edge Detection")
            cv2.imshow("Image", image)
            cv2.imshow("Edged", edged)
            cv2.imwrite('Edged.png',edged)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # find the contours in the edged image, keeping only the
        # largest ones, and initialize the screen contour
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
        #print(cnts)
        # loop over the contours
        for c in cnts:
            counter +=1
            temp_image = image.copy()
            ## compute the center of the contour
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # draw the contour and center of the shape on the image
            cv2.drawContours(temp_image, [c], -1, (0, 0, 255), 2)
            cv2.circle(temp_image, (cX, cY), 3, (0, 0, 0), -1)
            cv2.putText(temp_image, "Document "+ str(counter), (cX - 20, cY - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            # approximate the contour
            peri = 0.02 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c,peri, True)
            print(len(approx))
            # 	# if our approximated contour has four points, then we
            # 	# can assume that we have found our screen
            if len(approx) == 4:
                screenCnt = approx
                #print(c)
                # show the contour (outline) of the piece of paper
                if showSteps == True:
                    print("STEP 2: Find contours of paper")
                    cv2.drawContours(temp_image, screenCnt, -1, (0, 255, 0), 2)
                    cv2.imshow("Outline", temp_image)
                    cv2.imwrite('Outline.png',temp_image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()


                # apply the four point transform to obtain a top-down
                # view of the original image
                warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
                #print('warped1', warped)
                # convert the warped image to grayscale, then threshold it
                # to give it that 'black and white' paper effect
                warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                T = threshold_local(warped, 19, offset = 10, method = "gaussian")
                warped = (warped > T).astype("uint8") * 255


                # show the original and scanned images
                if showSteps == True:
                    print("STEP 3: Apply perspective transform")
                    cv2.imshow("Original", imutils.resize(orig, height = 650))
                    cv2.imshow("Scanned1", imutils.resize(warped, height = 650))
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                cv2.imwrite('scanned'+str(counter)+'.png', warped)
    imageDone = False

