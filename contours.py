import cv2
import numpy as np

# Load the image
image = cv2.imread('frame_img.png')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply a binary threshold
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find contours
contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Optionally draw contours on the original image
for contour in contours:
    cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)  # Draw in green

# Show the original image with contours (for visualization)
cv2.imshow('Contours', image)
cv2.waitKey(0)
cv2.destroyAllWindows()