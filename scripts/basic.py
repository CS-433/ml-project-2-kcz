import numpy as np
from PIL import Image

def rescale_image(img):
    """
    Rescale the image's grayscale values to
    the 0-1 range.
    """
    result = img.copy()
    minval, maxval = np.min(img), np.max(img)
    if (maxval != minval):
        result = (result - minval)/(maxval - minval)
    return result


def separate(img, factor):
    """
    Separate the images by making bright pixels
    brighter and dim pixels dimmer
    """
    result = img.copy()
    result = ((img + 0.1)*(img - 1.1)) ** factor
    return np.abs(rescale_image(result))


def threshold(img, alpha):
    """
    Make all pixels with values above of equal
    to `alpha` be 1, and all others be 0.
    """
    result = img.copy()
    idx = result >= alpha
    result[idx] = 1
    result[~idx] = 0

    return result


def pad_img(img, frame):
    """
    Add black padding to an image. The amount of vertical padding
    is specified by `frame[0]`, whereas the amount of horizontal
    padding is specified by `frame[1]`.
    """
    pad_x = img.shape[0] + frame[0] * 2
    pad_y = img.shape[1] + frame[1] * 2
    padding = np.zeros(shape=(pad_x, pad_y))
    padding[frame[0]:-frame[0], frame[1]:-frame[1]] = img

    return padding


def save_img(img, path):
    saved = Image.fromarray(np.uint8(img * 255)).convert('RGB')
    saved.save(path)