import cv2
import numpy as np


dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

marker = cv2.aruco.generateImageMarker(
    dictionary,
    0,
    600,
)

# White quiet zone around marker
image = np.full(
    (800, 800),
    255,
    dtype=np.uint8,
)

image[100:700, 100:700] = marker

cv2.imwrite(
    r"..\robotis_mujoco_menagerie"
    r"\robotis_tb3"
    r"\aruco_0.png",
    image,
)

print("ArUco marker saved.")