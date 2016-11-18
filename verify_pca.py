import pickle
import numpy as np
from pylab import *
from skimage import data, io, color
from mypca import MyPCA

pca = MyPCA('trained_model.out')
with open('transformed_set.out', 'r') as f:
	X = pickle.load(f)
with open('transformed_id_set.out', 'r') as f:
	ids = pickle.load(f)


for idx in range(0,10):
	img = color.rgb2grey(io.imread('images_training_rev1/' + str(ids[idx]) + '.jpg'))
	subplot(2,10,idx+1)
	xlabel("original" + str(ids[idx]))
	imshow(img)

for idx in range(0,10):
	img = pca.model.inverse_transform(X[idx]).reshape(424, 424)
	subplot(2,10,idx+11)
	xlabel("pca" + str(ids[idx]))
	imshow(img)

show()


