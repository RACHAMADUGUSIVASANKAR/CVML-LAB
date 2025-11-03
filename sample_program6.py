#6. write a program of Histragram Equalization
import cv2
import numpy as np

# Read the image in grayscale mode
img = cv2.imread('images/car.jpg', 0)

# Apply histogram equalization
equ = cv2.equalizeHist(img)

# Stack original and equalized images side by side for comparison
res = np.hstack((img, equ))

# Display the result
cv2.imshow('Histogram Equalization', res)

# Wait for a key press and close windows
cv2.waitKey(0)
cv2.destroyAllWindows()
