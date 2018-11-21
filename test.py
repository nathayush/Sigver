import numpy as np
import torch

t1 = torch.from_numpy(np.asarray([37, 49, 12, 47, 68, 38, 26, 63, 36, 15, 9, 50, 65, 31, 31, 59, 33, 66, 53, 28, 27, 59, 60, 30, 39, 15, 30, 16, 53, 29, 68, 37])).view(-1,1)
one_hot = torch.zeros(32, 44)
one_hot.scatter(2, t1, 1)
print(torch.eye(44).index(1, indicies))
