import numpy as np
from skimage.util import img_as_float
import cv2

def brisque_reward(brisque_metrics, previous_image, current_image):
    if len(previous_image.shape) == 4:
        previous_image = np.transpose(previous_image[0], [1,2,0]) * 255
        current_image = np.transpose(current_image[0], [1,2,0]) * 255
    else:
        previous_image = np.transpose(previous_image, [1,2,0]) * 255
        current_image = np.transpose(current_image, [1,2,0]) * 255
    reward = brisque_metrics.score(img_as_float(cv2.cvtColor(previous_image, cv2.COLOR_BGR2RGB))) - brisque_metrics.score(img_as_float(cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)))
    return reward