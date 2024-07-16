import argparse
import cv2
import numpy as np
import os
import time
from utils.bfilter2 import bfilter2
from sklearn.svm import LinearSVC
import random

def compute_mean_and_std(img, mask):
    list = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if mask[i][j] != 0:
                list.append(img[i][j])
    list = np.array(list)
    mean = np.mean(list)
    std = np.std(list)
    return mean, std

def sample_possible_rain_pixels(B_nonrain, G_nonrain, R_nonrain, B_mean, G_mean, R_mean, B_std, G_std, R_std, svm_predict_mask):
    sample_mask = np.zeros_like(B_nonrain)
    
    count = 0
    coord_list = []
    for i in range(B_nonrain.shape[0]):
        for j in range(B_nonrain.shape[1]):
            if (R_nonrain[i][j] <= R_mean + R_std) and (G_nonrain[i][j] <= G_mean + G_std) and (B_nonrain[i][j] <= B_mean + B_std) and\
                (R_nonrain[i][j] >= R_mean - R_std) and (G_nonrain[i][j] >= G_mean - G_std) and (B_nonrain[i][j] >= B_mean - B_std):
                coord_list.append((i, j))
                count += 1
    
    sample_index = np.random.choice(len(coord_list), size=count, replace=False)
    sample_list = [coord_list[i] for i in sample_index]
    for (i, j) in sample_list:
        sample_mask[i][j] = 255
    
    filtered_sample_mask = np.zeros_like(sample_mask)
    filtered_sample_mask = np.where(svm_predict_mask == 255, sample_mask, 0)
    
    return filtered_sample_mask

def sample_process(HF_rainy, mask, svm_predict_mask):
    B, G, R = cv2.split(HF_rainy)
    B_rain = np.where(mask != 0, B, 0)
    G_rain = np.where(mask != 0, G, 0)
    R_rain = np.where(mask != 0, R, 0)
            
    B_nonrain = np.where(mask == 0, B, 0)
    G_nonrain = np.where(mask == 0, G, 0)
    R_nonrain = np.where(mask == 0, R, 0)

    B_rain_mean, B_rain_std = compute_mean_and_std(B_rain, mask)
    G_rain_mean, G_rain_std = compute_mean_and_std(G_rain, mask)
    R_rain_mean, R_rain_std = compute_mean_and_std(R_rain, mask)
    sample_mask = sample_possible_rain_pixels(B_nonrain, G_nonrain, R_nonrain, B_rain_mean, G_rain_mean, R_rain_mean, B_rain_std, G_rain_std, R_rain_std, svm_predict_mask)

    new_rain_mask = np.where(sample_mask!=0, 255, mask)
    
    return new_rain_mask

def svm_process(image, rain_mask):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    normalized_image = image_gray / 255.0
    # extract 9x9 window
    window_size = 4
    features = []
    labels = []
    for i in range(window_size, normalized_image.shape[0] - window_size):
        for j in range(window_size, normalized_image.shape[1] - window_size):
            window = normalized_image[i - window_size:i + window_size + 1, j - window_size:j + window_size + 1].ravel()
            features.append(window)
            labels.append(rain_mask[i, j] > 0)
    features = np.array(features)
    labels = np.array(labels)

    # compute number of rain pixels and non-rain pixels
    num_rain_samples = np.sum(labels)
    if num_rain_samples == 0:
        return rain_mask
    num_non_rain_samples = len(labels) - num_rain_samples
    # random sample the non-rain pixels to be same number of rain pixels 
    random_non_rain_indices = random.sample(range(num_non_rain_samples), num_rain_samples)
    random_non_rain_samples = features[labels == 0][random_non_rain_indices]
    # combine 
    balanced_features = np.vstack([features[labels == 1], random_non_rain_samples])
    balanced_labels = np.concatenate([np.ones(num_rain_samples), np.zeros(num_rain_samples)])
    # train svm
    linear_svc_model = LinearSVC(dual=True, max_iter=5000)
    linear_svc_model.fit(balanced_features, balanced_labels)
    # predict svm
    padded_image = cv2.copyMakeBorder(normalized_image, window_size, window_size, window_size, window_size, cv2.BORDER_REFLECT)
    predicted_mask = np.zeros_like(image_gray, dtype=np.uint8)
    pad_height, pad_width = padded_image.shape
    for i in range(window_size, pad_height - window_size):
        for j in range(window_size, pad_width - window_size):
            window = padded_image[i - window_size:i + window_size + 1, j - window_size:j + window_size + 1].ravel()
            prediction = linear_svc_model.predict([window])
            predicted_mask[i-window_size, j-window_size] = 255 if prediction else 0
    
    return predicted_mask

def main(args):
    os.makedirs(args.save_dir_path, exist_ok=True)
    img_list = os.listdir(args.rainy_dir_path)
    img_list.sort()
    print(img_list)
    print(len(img_list))
    
    total_time = 0
    for img_name in img_list:
        print(img_name)
        beg = time.time()        
        rainy = cv2.imread(os.path.join(args.rainy_dir_path, img_name))
        mask = cv2.imread(os.path.join(args.mask_dir_path, img_name[:-4]+'.png'), 0)
        mask = np.where(mask!=0, 255, 0)
        
        if np.sum(mask) == 0:
            cv2.imwrite(os.path.join(args.save_dir_path, img_name[:-4]+'.png'), mask)
            continue
        
        svm_predict_mask = svm_process(rainy, mask)
        
        rainy = (rainy / 255).astype(np.float32)
        ILF = bfilter2(rainy, 5, (6, 0.2))
        IHF = rainy - ILF
        ILF = np.clip(ILF * 255, 0, 255)
        IHF = np.clip((IHF) * 255, 0, 255)
        
        new_rain_mask = sample_process(IHF, mask, svm_predict_mask)
        cv2.imwrite(os.path.join(args.save_dir_path, img_name[:-4]+'.png'), new_rain_mask)
        end = time.time()
        
        process_time = end - beg
        print("process_time:", process_time)
        total_time += process_time
        
    print("average process time", total_time/len(img_list))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--rainy_dir_path",  type=str, default='dataset/Rain100L/test/input')
    parser.add_argument('--mask_dir_path', type=str, default='dataset/Rain100L/test/RDP')
    parser.add_argument('--save_dir_path', type=str, default='dataset/Rain100L/test/RDP_w_sampling')
    args = parser.parse_args()

    main(args)