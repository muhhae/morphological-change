import cv2
import numpy as np
import copy

image_path = "./images.jpg"
image = cv2.imread(image_path)
img_cpy = copy.deepcopy(image)

lower_skin = np.array([0, 20, 70], dtype=np.uint8)
upper_skin = np.array([20, 255, 255], dtype=np.uint8)

hsv_image = cv2.cvtColor(img_cpy, cv2.COLOR_BGR2HSV)

skin_mask = cv2.inRange(hsv_image, lower_skin, upper_skin)

img_cpy[skin_mask > 0] = [0, 255, 0]

shirt_area = (img_cpy[..., 1] < 100) & (img_cpy[..., 0] < 100)
img_cpy[shirt_area] = [0, 0, 255]

pants_area = (img_cpy[..., 2] < 100) & (img_cpy[..., 1] < 100)
img_cpy[pants_area] = [255, 0, 0]

kernel = np.ones((5, 5), np.uint8)
morphed_image = cv2.dilate(img_cpy, kernel, iterations=1)

cv2.imshow("Original Image", image)
cv2.imshow("Morphological Transformation", morphed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
