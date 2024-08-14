import rawpy
import cv2
import os
import logging

logging.basicConfig()
logger = logging.getLogger("Image")
logger.setLevel(logging.DEBUG)

def read_image(file):

    with rawpy.imread(file) as raw:
        rgb = raw.postprocess()
    

    return rgb

def save_image(image, file):

    # Create output directory
    path = os.path.dirname(file)
    if not os.path.exists(path):
        os.makedirs(path)

    cv2.imwrite(file, image)

