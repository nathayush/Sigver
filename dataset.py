import pandas as pd
import numpy as np
import cv2
import torch
from scipy import ndimage
import os
from torch.utils.data import Dataset

class CNNData(Dataset):
    def __init__(self, path):
        x_data = []
        y_data = []
        z_data = []

        for filename in os.listdir(path):
            x_data.append(self.process_img(os.path.join(path, filename)))
            y_data.append(float(filename[0:3]))
            forg = 1 if filename[-5] == 'f' else 0
            z_data.append(forg)

        self.x_data = torch.from_numpy(np.asarray(x_data)).type(torch.FloatTensor) # 220 x 150 tensors of images
        self.y_data = torch.from_numpy(np.asarray(y_data)).type(torch.FloatTensor) # 1-d 2-tensors of target_user, forgery_bool
        self.z_data = torch.from_numpy(np.asarray(z_data)).type(torch.FloatTensor)

        assert len(self.x_data)==len(self.y_data) and len(self.y_data)==len(self.z_data)
        self.len = len(self.x_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index], self.z_data[index]

    def __len__(self):
        return self.len

    def process_img(self, filepath, sizeX=220, sizeY=150):
        canvas_width, canvas_height = 1616, 1132
        blank_image = np.zeros((canvas_height,canvas_width), np.uint8) # blank canvas

        image = cv2.imread(filepath)

        img = cv2.bitwise_not(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)) # white on black
        blur = cv2.GaussianBlur(img, (3,3), 0) # optional
        _, img = cv2.threshold(img, 125, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) # otsu thresholding
        img_height, img_width = img.shape

        Y_COM_img, X_COM_img  = ndimage.center_of_mass(img) # center of mass

        X_COI_canvas = canvas_width / 2 # canter of canvas
        Y_COI_canvas = canvas_height / 2
        imgX = (canvas_width) / 2
        imgY = (canvas_height) / 2

        pip_h = imgY - int(Y_COM_img) # offset
        pip_w = imgX - int(X_COM_img)

        blank_image[int(pip_h):int(pip_h+img_height), int(pip_w):int(pip_w+img_width)] = img # place image over canvas

        cropY = int(Y_COI_canvas) - sizeY / 2
        cropX = int(X_COI_canvas) - sizeX / 2
        crop_img = blank_image[int(cropY):int(cropY+sizeY), int(cropX):int(cropX+sizeX)] # crop to needed resolution

        return crop_img

testData = CNNData("data/cnn/test")
