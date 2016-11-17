import pickle
import numpy as np

from sklearn.decomposition import IncrementalPCA

NUM_COMP = 125

class MyPCA:

	def __init__(self, filename=None):
		if not filename:
			self.model = IncrementalPCA(NUM_COMP)
		else:
			with open(filename, 'r') as f:
				self.model = pickle.load(f)

	def train(self, X):
		self.model.partial_fit(X)

	def transform(self, X):
		return self.model.transform(X)	

	def dump(self, filename):
		with open(filename, 'w') as f:
			pickle.dump(self.model, f)
