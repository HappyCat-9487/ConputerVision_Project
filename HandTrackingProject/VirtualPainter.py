import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm

#####################################
brushThickness = 15
eraserThickness = 100
#####################################

folderPath = "HeaderPro"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))
header = overlayList[0]
drawColor = (0, 0, 255)

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

detector = htm.handDetector(detectionCon=0.85)
xp, yp = 0, 0
imgCanvas = np.zeros((480, 640, 3), np.uint8)

while True:
    # 1.Import image
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # 2.Find Hand Landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw= False)

    if len(lmList) != 0:
        # print(lmList)

        # tip of index and middle fingers
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # 3.Check which fingers are up
        fingers = detector.fingerUp()
        #print(fingers)

        # 4.If selection meode - Two fingers are up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            print("Selection Mode")

            #checking for the click
            if y1 < 75:
                if 139 < x1 < 235:
                    header = overlayList[0]
                    drawColor = (0, 0, 255)
                elif 282 < x1 < 378:
                    header = overlayList[1]
                    drawColor = (255, 200, 110)
                elif 416 < x1 < 512:
                    header = overlayList[2]
                    drawColor = (34, 139, 34)
                elif 529 < x1 < 625:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)


        # 5.If drawing mode - Index finger is up
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            print("Drawing Mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # Setting the header image
    img[0:75, 0:640] = header
    # img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    cv2.imshow("Image", img)
    # cv2.imshow("Canvas", imgCanvas)
    cv2.waitKey(1)
