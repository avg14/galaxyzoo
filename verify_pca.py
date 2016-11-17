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
	img = color.rgb2grey(io.imread('images_training_rev1/' + str(ids[100 + idx]) + '.jpg'))
	subplot(2,10,idx+1)
	imshow(img)

for idx in range(0,10):
	img = pca.model.inverse_transform(X[100 + idx]).reshape(424, 424)
	subplot(2,10,idx+11)
	imshow(img)

show()


