import numpy as np
import os
os.environ["XDG_SESSION_TYPE"] = "xcb"
import matplotlib
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.util import random_noise
from skimage import feature



from skimage import io

# image = io.imread('computer_vision/79690227.jpg')

import cv2
 
# Load image in grayscale
img = cv2.imread("/home/luuk/mavlab/paparazzi/computer_vision/86623673.jpg", cv2.IMREAD_GRAYSCALE)

# print(type(image))21190489352849371
# print(image.dtype)
# print(image.shape)
# print(image.min(), image.max())
 
# Apply Gaussian Blur to reduce noise
blur = cv2.GaussianBlur(img, (5, 5), 1.4)
 
# Apply Canny Edge Detector
edges = cv2.Canny(blur, threshold1=25, threshold2=70)

print(type(edges))
print(edges.dtype)
print(edges.shape)
print(edges.min(), edges.max())
#print(edges)

X = []
x = []
y = []
for i in range(int(0.40*edges.shape[0]), int(0.6*edges.shape[0])):
    for j in range(int(0.4*edges.shape[1]), int( 0.6*edges.shape[1])):
        if edges[i][j] == 255:
            X.append([j,i])
            x.append(i)
            y.append(j)

print(X)
print(len(X))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn import datasets

# X, y_true = make_blobs(n_samples=300, centers=4,
#                        cluster_std=0.50, random_state=0)

# print(X)
# print(y_true)


db = DBSCAN(eps=2.5, min_samples=7).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
print(labels)
print(len(labels))
print(min(labels), max(labels))

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

unique_labels = set(labels)
colors = ['y', 'b', 'g', 'r', 'm']
print(colors)
# for k, col in zip(unique_labels, colors):
#     if k == -1:
       
#         col = 'k'

#     class_member_mask = (labels == k)

#     xy = X[class_member_mask & core_samples_mask]
#     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
#              markeredgecolor='k',
#              markersize=6)

#     xy = X[class_member_mask & ~core_samples_mask]
#     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
#              markeredgecolor='k',
#              markersize=6)

print('number of clusters: %d' % n_clusters_)
import random
xs = np.array(x)
ys = np.array(y)
lbls = np.array(labels)
colormap = []
# can use named colors or HTML codes
for i in range(n_clusters_):
    colormap.append("#%06x" % random.randint(0, 0xFFFFFF))

print(colormap)
print(len(colormap))
#colormap = np.array(['#dce9f8', '#6f93cd', '#91965f', '#4a9f6e', '#e0f48a', '#44d1ee', '#2a594c', '#e946f1', '#a29325'])
colormaps = np.array(colormap)
plt.scatter(xs, ys, s=5, c=colormaps[lbls])
plt.show()



# # # Display result
# cv2.imshow("Canny Edge Detection", edges)

 
# cv2.waitKey(0)
# cv2.destroyAllWindows()






# plt.imshow(image)
# plt.plot()


# # Generate noisy image of a square
# image = np.zeros((128, 128), dtype=float)
# image[32:-32, 32:-32] = 1

# image = ndi.rotate(image, 15, mode='constant')
# image = ndi.gaussian_filter(image, 4)
# image = random_noise(image, mode='speckle', mean=0.1)

# # Compute the Canny filter for two values of sigma
# edges1 = feature.canny(image)
# edges2 = feature.canny(image, sigma=3)

# # display results
# fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 3))

# ax[0].imshow(image, cmap='gray')
# ax[0].set_title('noisy image', fontsize=20)

# ax[1].imshow(edges1, cmap='gray')
# ax[1].set_title(r'Canny filter, $\sigma=1$', fontsize=20)

# ax[2].imshow(edges2, cmap='gray')
# ax[2].set_title(r'Canny filter, $\sigma=3$', fontsize=20)

# for a in ax:
#     a.axis('off')

# fig.tight_layout()
# plt.show()