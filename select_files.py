import pickle
import random

from os import listdir
from os.path import isfile, join


DIR = './images_training_rev1/'
NUM_SELECTED = 15000
FILENAME = 'selected_files.out'


if __name__ == '__main__':
	files = [f for f in listdir(DIR) if isfile(join(DIR, f))]
	random.shuffle(files)
	files = files[:NUM_SELECTED]

	with open(FILENAME, 'w') as f:
		pickle.dump(files, f)
