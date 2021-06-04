import cv2
import numpy as np

img_name = input("Please enter the image name you want to be skeletonized : ")
img_color = cv2.imread(img_name)
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
# copy gray image to implementation
img = img_gray.copy()

#########################################
########## Threshold the image ##########
#########################################
histr_gray = cv2.calcHist([img], [0], None, [256], [0, 256])

# initialize values
prob = np.zeros(256)
total = img.size
img_shape = img.shape
hight = img_shape[0]
width = img_shape[1]

# Calculate the probability of each gray level
for i in range(256):
    prob[i] = histr_gray[i] / total

# Calculate a(t), b(t), m, m(t) 
a = np.zeros(256)
b = np.zeros(256)
m_a = np.zeros(256)
m = 0
for i in range(256):
    m += i * prob[i]
for t in range(256):
    for i in range(t+1):
        a[t] += prob[i]
    for i in range(t+1, 256):
        b[t] += prob[i]
    for i in range(t+1):
        m_a[t] += i * prob[i]

# find max_t s.t. the func value is the maximum
max_t = 0
max_value = 0
for t in range(256):
    if(a[t]*b[t]>0):
        func_value = (m_a[t] - m*a[t])**2 / (a[t]*b[t])
        if max_value < func_value:
            max_value = func_value
            max_t = t

# Reset gray values of the gray image
for i in range(hight):
    for j in range(width):
        if(img[i][j] >= max_t):
            img[i][j] = 255
        else:
            img[i][j] = 0

#########################################
####### Create an empty skeleton ########
#########################################
size = np.size(img)
img_skel = np.zeros(img.shape, np.uint8)

#########################################
####### Get a Cross Shaped Kernel #######
#########################################
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

#########################################
########## Repeat skeletonize ###########
#########################################
while True:
    # Open the image
    open = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
    # Substract open from the original image
    temp = cv2.subtract(img, open)
    # Erode the original image and refine the skeleton
    img_eroded = cv2.erode(img, element)
    img_skel = cv2.bitwise_or(img_skel,temp)
    img = img_eroded.copy()
    # If there are no white pixels left ie.. the image has been completely eroded, quit the loop
    if cv2.countNonZero(img)==0:
        break

##########################################
#### Saving and displaying the result ####
##########################################
cv2.imwrite('Skeletonized-' + img_name, img_skel)

cv2.imshow("Skeletonized Image",img_skel)
cv2.imshow("Gray Image",img_gray)
cv2.imshow("Original Image",img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()