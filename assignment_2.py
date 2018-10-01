import os
import csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import math
from time import time
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
from sklearn.manifold import MDS
from sklearn.manifold import LocallyLinearEmbedding


def read():
	path = os.path.abspath("data")
	file = open(path + '/5000.txt','r')

	df = pd.read_csv(file, delim_whitespace=True, low_memory=False)

	values = list(df.columns.values)

	#REPLACING STRING TYPE DATA WITH RELATED NUMBERS
	df = df.replace('CLR', 0)
	df = df.replace('SCT', 1)
	df = df.replace('BKN', 2)
	df = df.replace('OVC', 3)
	df = df.replace('OBS', 4)
	df = df.replace('POB', 5)

	"""
	ENCODE LABELS WITH VALUE BETWEEN 0 AND N_CLASSES-1
	"""
	le = preprocessing.LabelEncoder()
	for i in values:
		df[i] = le.fit_transform(df[i].values.astype('U'))

	return df

def pca(df):

	t0 = time()

	#STANDARDIZING THE FEATURES
	x = StandardScaler().fit_transform(df)

	#PROJECTS THE ORIGINAL DATA INTO 2 DIMENSIONS AND TRANSFORMS IT USING PCA MODEL
	pca = PCA(n_components = 2)
	Y = pca.fit_transform(x)

	t1 = time()

	plt.scatter(Y[:, 0], Y[:, 1],edgecolor = 'none',c = Y[:,0],cmap = plt.cm.Spectral)
	plt.title("PCA (%.2g sec)" % (t1 - t0))
	plt.xlabel('Component 1')
	plt.ylabel('Component 2')
	plt.show()

def isomap(df):

	t0 = time()

	#PROJECTS THE ORIGINAL DATA INTO 2 DIMENSIONS AND TRANSFORMS IT USING ISOMAP MODEL
	isomap = Isomap(n_components = 2, eigen_solver = 'auto')
	Y = isomap.fit_transform(df)

	t1 = time()

	plt.scatter(Y[:, 0], Y[:, 1], c = Y[:,0] ,cmap = plt.cm.Spectral)
	plt.title("Isomap (%.2g sec)" % (t1 - t0))
	plt.xlabel('Component 1')
	plt.ylabel('Component 2')
	plt.show()

def mds(df):

	t0 = time()
	#PROJECTS THE ORIGINAL DATA INTO 2 DIMENSIONS AND TRANSFORMS IT USING MDS MODEL
	mds = MDS(n_components = 2, max_iter = 100, n_init = 1)
	Y = mds.fit_transform(df)

	t1 = time()

	plt.scatter(Y[:, 0], Y[:, 1], c = Y[:,0] , cmap = plt.cm.Spectral)
	plt.title("MDS (%.2g sec)" % (t1 - t0))
	plt.xlabel('Component 1')
	plt.ylabel('Component 2')
	plt.show()

def lle(df):

	t0 = time()
	"""
	PROJECTS THE ORIGINAL DATA INTO 2 DIMENSIONS AND TRANSFORMS IT USING LLE MODEL.
	NUMBER OF NEIGHBORS DEFINED ACCORDING TO LECTURE PDFS.
	"""
	lle = LocallyLinearEmbedding(n_neighbors = 7, n_components = 2,eigen_solver = 'auto')
	Y = lle.fit_transform(df)

	t1 = time()

	plt.scatter(Y[:, 0], Y[:, 1], c = Y[:,0] , cmap = plt.cm.Spectral)
	plt.title("LocallyLinearEmbedding (%.2g sec)" % (t1 - t0))
	plt.xlabel('Component 1')
	plt.ylabel('Component 2')
	plt.show()

def main(): 
	df = read()
	
	pca(df)
	isomap(df)
	mds(df)
	lle(df)

if __name__ == "__main__":
    main()