# check our webcam
import cv2
import os
from cvzone.HandTrackingModule import HandDetector
import numpy as np

# variables
width, height = 1000, 680
folderPath = "presentation"
# Camera setup

cap = cv2.VideoCapture(1)
cap.set(3, width)
cap.set(4, height)

# get the list of the presentation images
pathImages = sorted(os.listdir(folderPath), key=len)
# problem is if our slide' name is 10( more than 10 slides), then it will show at second (1,10,3,4,5) which doesnt
# make sense solution is sorted print(pathImages)


# Variables
imgNumber = 1
# smaller size of webcam pic
heightSmall, weightSmall = int(120 * 1), int(213 * 1)
gestureThreshold = 250  # value below this, the gesture detection wouldnt works. ideally is center of ur face
buttonPressed = False
buttonCounter = 0
buttonDelay = 30  # buttonDelay is the num of Frame
annotations = [[]]  # list inside the list to create breakpoint between annotation and annotation
annotationNumber = 0
annotationStart = False  # to trace the starting point of annotation

# Hand Detector
# detection Confident = 0.8
detector = HandDetector(detectionCon=0.8, maxHands=1)

while True:
    # Import Images(presentation slides)
    success, img = cap.read()
    # need to flip the image so that when we move to right, it is move to right and vice versa
    img = cv2.flip(img, 1)  # flip in horizontal direction
    # join the dir name
    pathFullImage = os.path.join(folderPath, pathImages[imgNumber])  # imgNumber is easier when we change slide
    # imgNumber will change when we change the hand gesture
    imgCurrent = cv2.imread(pathFullImage)
    # since we flip the image, then the hand detection also terbalik(right hand detect as left, left detect as right)
    # but the rest ok no prob
    # Adding webcam image on the slide
    imgSmall = cv2.resize(img, (weightSmall, heightSmall))
    h, w, _ = imgCurrent.shape

    # this built in libray also not that perfect, but ok la
    hands, img = detector.findHands(img)

    # define the line of detection
    cv2.line(img, (0, gestureThreshold), (width, gestureThreshold), (0, 255, 0), 8)

    # if we detect dao hands, get the landmarks and number of finger
    if hands and buttonPressed is False:
        hand = hands[0]  # get the first hand but max number of hands is 1 only, so just take the first hand.
        fingers = detector.fingersUp(hand)  # detect  the finger up
        centerx, centery = hand['center']
        lmList = hand['lmList']

        # Constraint values for easier drawing

        xVal = int(np.interp(lmList[8][0], [width // 2, w], [0, width]))  # convert the range , w is the slide width
        # change lmList[8][0] to range(width  divided by 2, w) and change it to [0,width]
        yVal = int(
            np.interp(lmList[8][1], [150, height - 150], [0, height]))  # remove 150 from top, and 150 from bottom
        indexFinger = xVal, yVal  # landmark of index finger
        # print(fingers)
        # detect the gesture using the finger up
        # condition: the gesture must be above ur face so avoid unnecessary detection, center of hand is above line
        if centery <= gestureThreshold:  # if hand is at the height of the face ( top is 0, bottom is max)
            annotationStart = False
            # Gesture 1 - Left (thumb out)
            if fingers == [1, 0, 0, 0, 0]:
                annotationStart = False  # to trace the starting point of annotation
                print("left")

                if imgNumber > 0:
                    buttonPressed = True
                    annotations = [[]]  # list inside the list to create breakpoint between annotation and annotation
                    annotationNumber = 0
                    imgNumber -= 1  # go to left slide, if not first slide , until now have one prob, too fast(need to have button to slow it down)
            # Gesture 2 - Right (pinky out)
            if fingers == [0, 0, 0, 0, 1]:
                annotationStart = False  # to trace the starting point of annotation
                print("right")

                if imgNumber < len(pathImages) - 1:
                    buttonPressed = True
                    annotations = [[]]  # list inside the list to create breakpoint between annotation and annotation
                    annotationNumber = 0

                    imgNumber += 1  # go to right slide if not last slide
        # outside the borderline
        # Gesture 3 - Show Pointer
        # We want to limit the pic size(finger movement) because we dont want to move our finger at the whole screen
        # we limit it at the half of the screen (scale up) (constraint the region)
        if fingers == [0, 1, 1, 0, 0]:
            cv2.circle(imgCurrent, indexFinger, 10, (0, 0, 255), cv2.FILLED)
            annotationStart = False

        # Gesture 4 - Draw Pointer
        if fingers == [0, 1, 0, 0, 0]:
            if annotationStart is False:
                annotationStart = True
                annotationNumber += 1
                annotations.append([])  # append empty list
            cv2.circle(imgCurrent, indexFinger, 10, (0, 0, 255), cv2.FILLED)
            annotations[annotationNumber].append(indexFinger)  # put inside the list of list, breakpoint of each anno
        else:
            annotationStart = False

        # Gesture 5= Erase
        if fingers == [0, 1, 1, 1, 0]:
            if annotations:
                if annotationNumber > -1:
                    annotations.pop(-1)
                    annotationNumber -= 1
                    buttonPressed = True
    else:
        annotationStart = False  # when hand is lost
    # outside the if statement
    # Button Pressed Iterations
    if (buttonPressed):
        buttonCounter += 1
        if buttonCounter > buttonDelay:  # buttonDelay is the num of Frame
            buttonCounter = 0
            buttonPressed = False

    # annotation
    for i in range(len(annotations)):
        for j in range(len(annotations[i])):
            if j != 0:
                cv2.line(imgCurrent, annotations[i][j - 1], annotations[i][j], (0, 0, 200), 12)

    # put the webcam in the top right corner
    # starting point of height is 0, ending is heightSmall
    # starting point of weight is totalweight - heightSmall , ending is totalweight
    imgCurrent[0:heightSmall, w - weightSmall:w] = imgSmall
    cv2.imshow("Image", img)  # later can remove it
    cv2.imshow("Slides", imgCurrent)
    key = cv2.waitKey(1)
    # quit if we press q key
    if key == ord('q'):
        break
