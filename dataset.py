import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from sklearn import preprocessing

class SigData(Dataset):
    def __init__(self):
        df = pd.read_csv("data/dataset.csv")
        self.len = len(df)

        # convert string names to ints
        name_encoder = preprocessing.LabelEncoder()
        name_encoder.fit(df['target'])
        df['target_encoded'] = name_encoder.transform(df['target'])

        x_data = []
        y_data = []
        for index, row in df.iterrows():
            img = cv2.resize(cv2.imread("data/images/{}".format(row["image"]), 0), (220, 150)) # PREPROCESSING YET TO BE DONE
            x_data.append(img)
            y_data.append((row["target_encoded"], row["forgery"]))
        self.x_data = torch.from_numpy(np.asarray(x_data)) # 220 x 150 tensors of images
        self.y_data = torch.from_numpy(np.asarray(y_data)) # 1-d 2-tensors of target_user, forgery_bool

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len
