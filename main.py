import cv2
import numpy as np

# Shirt
# HSV Value at (73, 92): [127 127  44]
# HSV Value at (73, 92): [127 127  44]
# HSV Value at (77, 94): [124  85  63]
# HSV Value at (85, 96): [125  83  52]
# HSV Value at (86, 96): [125  83  52]
# HSV Value at (95, 91): [125  96  45]
# HSV Value at (104, 82): [129  85  42]
# HSV Value at (110, 80): [128  93  41]
# HSV Value at (110, 78): [125  96  45]
# HSV Value at (104, 83): [129  83  43]
# HSV Value at (100, 94): [129  87  41]
# HSV Value at (90, 107): [125  87  50]
# HSV Value at (82, 124): [127  87  50]
# HSV Value at (80, 131): [124  92  58]
# HSV Value at (82, 128): [129  71  50]
# HSV Value at (84, 119): [127  83  52]
# HSV Value at (80, 110): [128  93  55]
# HSV Value at (82, 99): [125  83  52]
# HSV Value at (85, 96): [125  83  52]
# HSV Value at (76, 79): [127  97  58]
# HSV Value at (75, 75): [128  68  56]
# HSV Value at (92, 89): [125  90  54]
# HSV Value at (105, 85): [129 108  33]
# HSV Value at (112, 78): [128 100  51]
# HSV Value at (111, 73): [124  79  45]
# HSV Value at (108, 71): [125  77  43]

# Pants
# HSV Value at (74, 142): [ 12  69 193]
# HSV Value at (94, 157): [ 13  65 188]
# HSV Value at (99, 171): [ 12  73 178]
# HSV Value at (81, 207): [ 14 101 156]
# HSV Value at (81, 223): [ 14  92 161]
# HSV Value at (106, 215): [ 13 100 161]
# HSV Value at (103, 194): [ 13  97 160]
# HSV Value at (104, 193): [ 13  91 159]
# HSV Value at (100, 174): [  9  70 174]
# HSV Value at (100, 159): [ 16  67 175]
# HSV Value at (73, 144): [ 13  76 192]
# HSV Value at (76, 167): [ 13  69 181]
# HSV Value at (75, 180): [ 13  76 182]
# HSV Value at (76, 194): [ 13  87 168]
# HSV Value at (78, 217): [ 14  94 168]
# HSV Value at (76, 230): [ 12  77 172]

# Skin
# HSV Value at (55, 145): [  6 124 161]
# HSV Value at (60, 134): [  9 101 177]
# HSV Value at (64, 122): [  7  97 226]
# HSV Value at (66, 111): [ 10 114 202]
# HSV Value at (70, 104): [ 11 112 226]
# HSV Value at (115, 101): [ 13 124 231]
# HSV Value at (114, 111): [ 12 133 217]
# HSV Value at (112, 118): [ 11 148 202]
# HSV Value at (108, 123): [ 11 137 209]
# HSV Value at (99, 131): [  8  90 199]
# HSV Value at (94, 142): [  8 124 227]
# HSV Value at (91, 56): [  9 140 138]
# HSV Value at (88, 47): [ 10  96 231]
# HSV Value at (93, 38): [ 10  94 218]
# HSV Value at (92, 33): [  9 101 197]


# Background
# HSV Value at (154, 185): [ 21  16 209]
# HSV Value at (151, 117): [ 11   9 222]
# HSV Value at (156, 62): [ 11   9 224]
# HSV Value at (146, 27): [ 11   9 222]
# HSV Value at (127, 16): [ 11   9 223]
# HSV Value at (102, 11): [ 11   9 224]
# HSV Value at (70, 9): [ 11   9 224]
# HSV Value at (43, 20): [ 11   9 224]
# HSV Value at (37, 74): [ 11   9 227]
# HSV Value at (27, 118): [ 11   9 228]
# HSV Value at (20, 191): [ 11   9 227]
# HSV Value at (25, 229): [ 11   9 227]
# HSV Value at (81, 262): [ 11   9 229]
# HSV Value at (125, 259): [ 11   9 228]
# HSV Value at (145, 243): [ 11   9 227]
# HSV Value at (147, 158): [ 16  13 215


def perform_color_detection(image: cv2.UMat) -> cv2.UMat:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    shirt_min = np.array([120, 65, 30])
    shirt_max = np.array([135, 110, 60])

    pants_min = np.array([8, 60, 150])
    pants_max = np.array([18, 110, 200])

    skin_min = np.array([5, 90, 130])
    skin_max = np.array([15, 150, 230])

    background_min = np.array([10, 0, 200])
    background_max = np.array([20, 20, 255])

    shirt = cv2.inRange(hsv, shirt_min, shirt_max)
    pants = cv2.inRange(hsv, pants_min, pants_max)
    skin = cv2.inRange(hsv, skin_min, skin_max)
    background = cv2.inRange(hsv, background_min, background_max)

    background = cv2.bitwise_and(background, cv2.bitwise_not(skin))

    black_background = np.zeros_like(image)

    red_shirt = cv2.bitwise_and(black_background, black_background, mask=shirt)
    red_shirt[:, :, 2] = cv2.bitwise_or(red_shirt[:, :, 2], shirt)

    blue_pants = cv2.bitwise_and(black_background, black_background, mask=pants)
    blue_pants[:, :, 0] = cv2.bitwise_or(blue_pants[:, :, 0], pants)

    green_skin = cv2.bitwise_and(black_background, black_background, mask=skin)
    green_skin[:, :, 1] = cv2.bitwise_or(green_skin[:, :, 1], skin)

    image = cv2.addWeighted(red_shirt, 1, blue_pants, 1, 0)
    image = cv2.addWeighted(image, 1, green_skin, 1, 0)

    image[background > 0] = [0, 0, 0]

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def main():
    image = cv2.imread("./images.jpg")
    img_hasil = perform_color_detection(image)

    cv2.imshow("win", image)
    cv2.imshow("win 1", img_hasil)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
