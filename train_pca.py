import pickle
import numpy as np

from skimage.io import imread
from skimage.color import rgb2grey
from sklearn.decomposition import IncrementalPCA

from mypca import MyPCA, NUM_COMP


BASE_PATH = './images_training_rev1/'
SELECTED_PATH = './selected_files'
BATCH_SIZE = 200
IMAGE_SIZE = (424, 424)
TRAINED_MODEL_FILENAME = 'trained_model'
TRANSFORMED_FILENAME = 'transformed_set'
TRAINING_SIZE = 500
TESTING_SIZE = 1000


def get_data_batch(selected_files, prev_i, i):
	X = np.empty((BATCH_SIZE, IMAGE_SIZE[0]*IMAGE_SIZE[1]))
	x_idx = 0
	print "Processing:", prev_i, i
	for x_i in range(prev_i, i):
		# Converting to grey for first pass - need to evaluate all three channels
		# separately for color
		img = rgb2grey(imread(BASE_PATH + selected_files[x_i])).reshape((1, IMAGE_SIZE[0]*IMAGE_SIZE[1]))
		X[x_idx,:] = img
		x_idx += 1
	return X, (x_i+1)


def train():
	pca = MyPCA()
	file = open(SELECTED_PATH, 'r')
	selected_files = pickle.load(file)[:TRAINING_SIZE]
	m = len(selected_files)
	prev_i = 0
	for i in range(BATCH_SIZE, m+1, BATCH_SIZE):
		X, prev_i = get_data_batch(selected_files, prev_i, i)
		pca.train(X)
	pca.dump(TRAINED_MODEL_FILENAME)



def transform():
	pca = MyPCA(TRAINED_MODEL_FILENAME)
	file = open(SELECTED_PATH, 'r')
	selected_files = pickle.load(file)[:TESTING_SIZE]
	m = len(selected_files)
	prev_i = 0
	transformed = np.empty((0, NUM_COMP))
	for i in range(BATCH_SIZE, m+1, BATCH_SIZE):
		X, prev_i = get_data_batch(selected_files, prev_i, i)
		transformed = np.vstack([transformed, pca.transform(X)])
	with open(TRANSFORMED_FILENAME, 'w') as f:
		pickle.dump(transformed, f)


if __name__ == '__main__':
	train()
	transform()