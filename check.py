import cv2

image_path = "./images.jpg"
image = cv2.imread(image_path)
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def get_hsv_value(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv_value = hsv_image[y, x]
        print("HSV Value at ({}, {}): {}".format(x, y, hsv_value))


cv2.imshow("Image", image)
cv2.setMouseCallback("Image", get_hsv_value)

cv2.waitKey(0)
cv2.destroyAllWindows()
