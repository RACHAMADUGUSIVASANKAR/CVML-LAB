import cv2
# Load the Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Load image
image = cv2.imread('men.jpg')  # Replace with your image path
if image is None:
    print("Error: Could not load image.")
    exit()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
# Draw rectangles around faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
# Show result
cv2.imshow('Face Detection - Haar Cascade', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
