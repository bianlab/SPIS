
import time
import scipy.io as scio
import cv2
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
from SPIS_simulation import SPIS_detect
import matplotlib.pyplot as plt
import torch



if __name__ == "__main__":
    detect = SPIS_detect()

    mode = "predict"

    crop            = False

    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0

    test_interval   = 100

    dir_origin_path = "test/"
    dir_save_path   = "img_out/"

    if mode == "predict":

        #load the measurements
        data=scio.loadmat('./features/features.mat')
        feature=data['data']
        print(feature.shape) #(5, 1, 96, 8, 8)
        img_names = os.listdir(dir_origin_path)
        i = 0
        plt.figure()
        for img_name in tqdm(img_names):
            image_path  = os.path.join(dir_origin_path, img_name)
            image       = Image.open(image_path)
            r_image = detect.detect_image(feature[i],image, crop = crop)
            plt.subplot(2,3,i+1)
            plt.imshow(r_image)
            i += 1
        plt.show()

