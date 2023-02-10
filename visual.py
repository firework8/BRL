import torch
import torch.nn as nn
import numpy as np
import tsneutil
import sys

from sklearn.manifold import TSNE

get_data = np.load("./data_1.npy")
get_label = np.load("./label_1.npy")
save_name = './data_1.jpg'

the_choosed_label = [6,7,13,17,18,43,48,54,55,58]
changed_label = range(10)

choose_data = []
choose_label = []
for idx in range(len(get_label)):
    if get_label[idx] in the_choosed_label:
        choose_data.append(get_data[idx])
        choose_label.append(changed_label[the_choosed_label.index(get_label[idx])])

data = np.array(choose_data)
data = np.reshape(data,(data.shape[0],data.shape[2]))

X_embedded = TSNE(n_components=2, init='pca', perplexity=10).fit_transform(data)
print(X_embedded.shape)

tsneutil.plot(X_embedded, choose_label, colors=tsneutil.MOUSE_10X_COLORS, save_name=save_name)





