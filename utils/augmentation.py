import numpy as np
import cv2

def aug_mask(image, mask, mask_percentage=0.6):
    # return the random aug mask
    original_mask = mask.copy()
    aug_mask = np.zeros_like(mask)
    h, w, _ = image.shape
    total_pixels = h * w
    iters = 0
    while iters < 100:
        iters += 1
        masked_pixels = np.count_nonzero(aug_mask)
        if masked_pixels / total_pixels > mask_percentage:
            break
        random_translation_x = np.random.randint(-w//2, w//2)
        random_translation_y = np.random.randint(-h//2, h//2)
        translation_matrix = np.array([[1, 0, random_translation_x], [0, 1, random_translation_y]], dtype=np.float32)
        tmp_mask = cv2.warpAffine(mask, translation_matrix, (mask.shape[1], mask.shape[0]))
        tmp_mask = cv2.resize(tmp_mask, (image.shape[1], image.shape[0]))
        aug_mask = np.where((original_mask == 0) & (tmp_mask > 0), 1, aug_mask)
    return aug_mask