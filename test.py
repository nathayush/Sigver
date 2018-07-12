from math import log
import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from sklearn import preprocessing
from scipy import ndimage

# outputs[0] 18x2 P(y_j|X_i)
# outputs[1] 18x1 P(f|X_i)
# labels[0] 18x1 y_j, f_i


# def main(outputs, labels, lambda=0.95, num_classes=2):
#     loss = 0
#     for i in range(0, labels.size):
#         # temp = (labels[i][1] - 1)*(1 - lambda)
#         temp2 = 0
#         for j in range(0, num_classes):
#             if j == int(labels[i][0]):
#                 temp2 += log(outputs[0][i][j])
#         # temp3 = labels[i][1]*log(outputs[1][i][0])
#         # temp4 = (1 - labels[i][1])*log(1-outputs[1][i][0])
#         # loss += temp*temp2 - lambda*(temp3 + temp4)
#         loss += (labels[i][1] - 1)*(1 - lambda)*temp2 - lambda*((labels[i][1]*log(outputs[1][i][0])) + ((1 - labels[i][1])*log(1-outputs[1][i][0])))

def process_img(sizeX=220, sizeY=150):
    canvas_width, canvas_height = 1360, 952
    blank_image = np.zeros((canvas_height,canvas_width), np.uint8)

    image = cv2.imread("data/images/{}".format("john-11.png"))

    img = cv2.bitwise_not(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    blur = cv2.GaussianBlur(img, (3,3), 0)
    _, img = cv2.threshold(img, 125, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_height, img_width = img.shape

    Y_COM_img, X_COM_img  = ndimage.center_of_mass(img)

    X_COI_canvas = canvas_width / 2
    Y_COI_canvas = canvas_height / 2
    imgX = (canvas_width) / 2
    imgY = (canvas_height) / 2

    pip_h = imgY - int(Y_COM_img)
    pip_w = imgX - int(X_COM_img)

    blank_image[int(pip_h):int(pip_h+img_height), int(pip_w):int(pip_w+img_width)] = img

    cropY = int(Y_COI_canvas) - sizeY / 2
    cropX = int(X_COI_canvas) - sizeX / 2
    crop_img = blank_image[int(cropY):int(cropY+sizeY), int(cropX):int(cropX+sizeX)]

    cv2.imwrite(r'.\dump_pip.png',crop_img)

process_img()
# outputs = torch.Tensor(
#     np.array([[ 0.2193,  0.1821,  0.1034,  0.3726,  0.1227],
#             [ 0.1974,  0.1802,  0.0860,  0.4037,  0.1328],
#             [ 0.2028,  0.1979,  0.0928,  0.3843,  0.1222],
#             [ 0.2069,  0.1789,  0.0885,  0.3966,  0.1291],
#             [ 0.2173,  0.1862,  0.0960,  0.3793,  0.1213],
#             [ 0.1924,  0.1944,  0.0896,  0.3857,  0.1379],
#             [ 0.2122,  0.1672,  0.0947,  0.4071,  0.1187]]),
#     np.array([0.1,
#             0.2,
#             0.3,
#             0.4,
#             0.5,
#             0.6,
#             0.7]))
# labels = [[0, 1]
#         [0, 1]
#         [0, 1]
#         [0, 1]
#         [0, 0]
#         [0, 0]
#         [0, 0]
#         [0, 0]]

# name_encoder = preprocessing.LabelEncoder()
# df = pd.read_csv("data/dataset.csv")
# name_encoder.fit(df['target'])
# df['target_encoded'] = name_encoder.transform(df['target'])
# x_data = []
# y_data = []
# for index, row in df.iterrows():
#     img = cv2.resize(cv2.imread("data/images/{}".format(row["image"]), 0), (220, 150))
#     x_data.append(img)
#     y_data.append((row["target_encoded"], row["forgery"]))
# x_data = torch.from_numpy(np.asarray(x_data))
# y_data = torch.from_numpy(np.asarray(y_data))
# print(y_data.numpy())
