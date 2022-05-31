
import numpy as np
import cv2 as cv

img = cv.imread('data/tray4.jpg',0)
img = cv.medianBlur(img,5)

cimg = cv.cvtColor(img,cv.COLOR_GRAY2BGR)
cimg=cv.GaussianBlur(img,(21,21),cv.BORDER_DEFAULT)
circles = cv.HoughCircles(img,cv.HOUGH_GRADIENT,0.9,10, param1=110,param2=49,minRadius=0,maxRadius=0)
circles = np.uint16(np.around(circles))

for i in circles[0,:]:
    # draw the outer circle
    cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

img = cv.imread('data/tray4.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray, 80, 150, apertureSize=3)
lines = cv.HoughLinesP(edges, 1, np.pi / 180, 150, minLineLength=100, maxLineGap=100)
print(lines.shape)
print(edges)

x_hig, y_low, x_low, y_hig = lines[0][0]
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if x1 > x_hig:
        x_hig = x1
    if x2 > x_hig:
        x_hig = x2
    if y1 > y_hig:
        y_hig = y1
    if y2 > y_hig:
        y_hig = y2
    if x1 < x_low:
        x_low = x1
    if x2 < x_low:
        x_low = x2
    if y1 < y_low:
        y_low = y1
    if y2 < y_low:
        y_low = y2

cv.rectangle(img, (x_low, y_low), (x_hig, y_hig), (225, 0, 0), 3)
# area=(x_hig+y_hig)*(x_low+x_hig)
# print(area)

coinin=0
coinout=0
for i in circles[0, :]:
    cv.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
    cv.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
    if i[0] > x_low and i[0] < x_hig and i[1] > y_low and i[1] < y_hig:
        coinin += 1
        print(coinin)
    if i[0] < x_low and i[0] > x_hig and i[1] < y_low and i[1] > y_hig:
        coinout +=1
        print(coinout)

cv.imshow("Linie", img)
cv.imshow('detected circles',cimg )
cv.waitKey(0)
cv.destroyAllWindows()
