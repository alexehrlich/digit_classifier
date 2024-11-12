import numpy as np

class Network(object):

	#sizes is an array of the layers. Each digit containing the number of nodes
	# sizes[0] must be the input nodes, no bias and no weights
	# [3, 5, 2] is the layer structure. 
	# :-1 creates [3, 5] (w/o the last element)
	# :1 creates [5, 1] (w/o the first element)
	# These arrays are elementwise zipped to the tupel ([3, 5], [5,2 ])
	# For matrix mult we must pass (y, x) and not (x, y). To match dimensions.
	# This is the shape of the needed weight matrix.
	# randn picks random standard distributed vals (SD = 1, mean = 0)
	def __init__(self, sizes) -> None:
		self.num_nodes = len(sizes)
		self.sizes = sizes
		self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
		self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

	# zip creates a tupel of a bias vector and a weight matrix for each layer.
	# we iterate over all tuples. one tupel represent one layer
	# the outputvector of each layer is caculated by the dot product of the
	#	weight-matrix and the current input vector. The corrosponding bias vector is then added.
	# The outputvector is elementwise activated by sigmoid and is the input vector
	#	for the next tupel/layer
	def feedforward(self, input):
		for b, w in zip(self.biases, self.weights):
			input = sigmoid(np.dot(w, input) + b)
		return input

#if z is a np array, the sigmoid is applied elementwise
def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))




net = Network([3, 5, 2])

