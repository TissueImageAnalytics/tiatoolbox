"""
This file contains miscellaneous small functions repeatedly required and used in the repo
"""
import cv2


def cv2_imread(image_path):
    """
    Read an image to a numpy array using OpenCV

    Args:
        image_path: Input file path

    Returns:
        img: image as numpy array
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def hello(count, name):
    """Simple program that greets NAME for a total of COUNT times."""
    for x in range(count):
        print("Hello %s!" % name)
