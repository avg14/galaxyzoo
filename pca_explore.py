from sklearn.decomposition import PCA, FastICA
from pylab import *
from skimage import data, io, color

img = io.imread('images_training_rev1/100023.jpg')

orig_shape = np.shape(img)
img1 = img[:,:,0]

num_components = []
variance_retained = []
compression_ratio = []

for i in range(1, 100):
	n_comp = 2 * i
	pca = PCA(n_components=n_comp, whiten=False)
	img_pca1 = pca.fit_transform(img1)
	num_components.append(n_comp)
	variance_retained.append(1 - sum(pca.explained_variance_ratio_) / size(pca.explained_variance_ratio_))
	compression_ratio.append(1 - (float(size(img_pca1)) / size(img1)))
plot(num_components, variance_retained, color='r')
plot(num_components, compression_ratio, color='b')
show()
