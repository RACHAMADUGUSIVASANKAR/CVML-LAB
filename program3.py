#EXPERIMENT 3 : Object detection using contour detection and bounding boxes?
import cv2

image = cv2.imread("images/car.jpg") 
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
blurred = cv2.GaussianBlur(gray, (5, 5), 0)   # Apply Gaussian blur to reduce noise
ret, threshold = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)    # Thresholding the image (binary inverse)
contours, ret = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)          # Find contours
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, "Object", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
cv2.imshow("Original with Bounding Boxes", image)
cv2.imshow("Threshold", threshold)
cv2.waitKey(0)
