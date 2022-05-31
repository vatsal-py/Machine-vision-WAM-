import cv2
import numpy as np


cap = cv2.VideoCapture('data/F1_r.MOV')

Centerframe = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2
while True:
    ret, frame = cap.read()
    width = int(cap.get(3))
    height = int(cap.get(4))

    blueFrame = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blueFrame, cv2.COLOR_BGR2HSV)
    # lower boundary RED color range values; Hue (0 - 10)
    lower1 = np.array([160, 50, 50])
    upper1 = np.array([10, 255, 255])

    # upper boundary RED color range values; Hue (160 - 180)
    lower2 = np.array([160, 50, 50])
    upper2 = np.array([179, 255, 255])

    lower_mask = cv2.inRange(hsv, lower1, upper1)
    upper_mask = cv2.inRange(hsv, lower2, upper2)

    full_mask = lower_mask + upper_mask;

    result = cv2.bitwise_and(frame, frame, mask=full_mask)

    kernel = np.ones((13, 13), np.uint8)
    mask_without_noise = cv2.morphologyEx(full_mask, cv2.MORPH_OPEN, kernel)
    mask_closed = cv2.morphologyEx(mask_without_noise, cv2.MORPH_CLOSE, kernel)

    counters, _ = cv2.findContours(mask_closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    MaxCont = counters[0]
    for counters in counters:
        if cv2.contourArea(counters) <= cv2.contourArea(MaxCont):
            continue
        MaxCont = counters

    counters = MaxCont
    approx = cv2.approxPolyDP(counters, 0.01 * cv2.arcLength(counters, True), True)

    x, y, w, h = cv2.boundingRect(approx)

    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    M = cv2.moments(counters)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    x_end = cx - Centerframe
    cv2.line(frame, (Centerframe, 50), (Centerframe + x_end, 50), (0, 255, 0), 3)
    cv2.circle(frame, (Centerframe, 50), 3, (0, 0, 0), -1)

    print(cx, cy)
    print(approx)
    print(MaxCont)

    cv2.imshow('frame', frame)
    cv2.imshow('new', mask_closed)

    if cv2.waitKey(25) == ord('q'):
        break

cv2.release()
cv2.destroyAllWindows()
