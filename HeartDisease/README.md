# MNIST Image classification

This folder contains projects involving the MNIST handwritten digit classification dataset.

*TDA : 
	1. We approach this dataset using topological data analysis (TDA) techniques. The idea is to convert each image to a pointcloud
	in $\bR^2$, and then use the stable rank invariant to produce a signature for each digit. We first develop a characteristic signature for each 
	digit by averaging the signature of a large number of images of that digit. To classify an image we then us a k-nearest neighbours algorithm.
	
	2. We reproduce the reults of [cite].

 