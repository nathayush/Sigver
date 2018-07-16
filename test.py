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


def main(label_pred, forg_pred, labels, forgeries):
    num_classes = 21
    lamb = 0.95
    out = 0
    for i in range(0, forgeries.shape[0]):
        temp = (forgeries[i] - 1)*(1 - lamb)
        temp2 = 0
        for j in range(0, num_classes):
            temp2 += labels[i][j]*log(label_pred[i][j])
        temp3 = forgeries[i]*log(forg_pred[i][0])
        temp4 = (1 - forgeries[i])*log(1-forg_pred[i][0])
        out += temp*temp2 - lamb*(temp3 + temp4)
    return out

# outputs = (
#     torch.Tensor(np.array([
#             [ 0.2193,  0.1821,  0.1034,  0.3726,  0.1227],
#             [ 0.1974,  0.1802,  0.0860,  0.4037,  0.1328],
#             [ 0.2028,  0.1979,  0.0928,  0.3843,  0.1222],
#             [ 0.2069,  0.1789,  0.0885,  0.3966,  0.1291],
#             [ 0.2173,  0.1862,  0.0960,  0.3793,  0.1213],
#             [ 0.1924,  0.1944,  0.0896,  0.3857,  0.1379],
#             [ 0.2122,  0.1672,  0.0947,  0.4071,  0.1187]])),
#     torch.Tensor(np.array([
#             [0.1],
#             [0.2],
#             [0.3],
#             [0.4],
#             [0.5],
#             [0.6],
#             [0.7]])))
# #
# labels = torch.Tensor(np.array(
#         [[0, 1],
#         [0, 1],
#         [0, 1],
#         [0, 1],
#         [0, 0],
#         [0, 0],
#         [0, 0],
#         [0, 0]]))

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
