# import numpy as np
# import random as rd

# class Network(object):

# 	#sizes is an array of the layers. Each digit containing the number of nodes
# 	# sizes[0] must be the input nodes, no bias and no weights
# 	# [3, 5, 2] is the layer structure. 
# 	# :-1 creates [3, 5] (w/o the last element)
# 	# :1 creates [5, 1] (w/o the first element)
# 	# These arrays are elementwise zipped to the tupel ([3, 5], [5,2 ])
# 	# For matrix mult we must pass (y, x) and not (x, y). To match dimensions.
# 	# This is the shape of the needed weight matrix.
# 	# randn picks random standard distributed vals (SD = 1, mean = 0)
# 	def __init__(self, sizes) -> None:
# 		self.num_layers = len(sizes)
# 		self.sizes = sizes
# 		self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
# 		self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

# 	# zip creates a tupel of a bias vector and a weight matrix for each layer.
# 	# we iterate over all tuples. one tupel represent one layer
# 	# the outputvector of each layer is caculated by the dot product of the
# 	#	weight-matrix and the current input vector. The corrosponding bias vector is then added.
# 	# The outputvector is elementwise activated by sigmoid and is the input vector
# 	#	for the next tupel/layer
# 	def feedforward(self, input):
# 		for b, w in zip(self.biases, self.weights):
# 			input = sigmoid(np.dot(w, input) + b)
# 		return input

# 	#training_data is a tupel of the input vector and the target vector
# 	# eta is the learning rate for gradient descent
# 	def SGD(self, training_data, batch_size, epochs, eta, test_data=None):
# 		n_training_data = len(training_data)
# 		for j in range(epochs):
# 			#shuffle the whole training data each epoch, to achieve more generalization
# 			rd.shuffle(training_data)
			
# 			#create a list of mini_batches
# 			mini_batches = [training_data[k:k+batch_size] for k in range(0, n_training_data, batch_size)]

# 			#update every mini batch in this epoch
# 			for mini_batch in mini_batches:
# 				self.update_mini_batch(mini_batch, eta)
# 			if test_data:
# 				print("Epoch {0}: {1} / {2}".format(
# 					j, self.evaluate(test_data), n_training_data))
# 			else:
# 				print("Epoch {0} complete".format(j))


# 	def update_mini_batch(self, mini_batch, eta):
# 		#create an accumulator for thre gradients of the cost function with respect to every weight and bias
# 		# creates all bias vectors and weight matrices but with zeros.
# 		nabla_b = [np.zeros(b.shape) for b in self.biases]
# 		nabla_w = [np.zeros(w.shape) for w in self.weights]

# 		#sum up all gradients (dC/dparamter) and store the in the accumulator
# 		for x, y in mini_batch:
# 			#self.backprop returns a list of all gradients for that mini batch
# 			delta_nabla_b, delta_nabla_w = self.backprop(x, y)
# 			#accumulate all the returned gradients from backpropagation
# 			nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
# 			nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

# 			#average the summed up gradients with the batch size and update
# 			# and update all weights and biases
# 			self.biases = [b - (eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]
# 			self.weights = [w - (eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
	
# 	def backprop(self, x, y):
# 		"""Return a tuple ``(nabla_b, nabla_w)`` representing the
# 		gradient for the cost function C_x.  ``nabla_b`` and
# 		``nabla_w`` are layer-by-layer lists of numpy arrays, similar
# 		to ``self.biases`` and ``self.weights``."""
# 		nabla_b = [np.zeros(b.shape) for b in self.biases]
# 		nabla_w = [np.zeros(w.shape) for w in self.weights]
# 		# feedforward
# 		activation = x
# 		activations = [x] # list to store all the activations, layer by layer
# 		zs = [] # list to store all the z vectors, layer by layer
# 		for b, w in zip(self.biases, self.weights):
# 			z = np.dot(w, activation)+b
# 			zs.append(z)
# 			activation = sigmoid(z)
# 			activations.append(activation)
# 		# backward pass
# 		delta = self.cost_derivative(activations[-1], y) * \
# 			sigmoid_prime(zs[-1])
# 		nabla_b[-1] = delta
# 		nabla_w[-1] = np.dot(delta, activations[-2].transpose())
# 		# Note that the variable l in the loop below is used a little
# 		# differently to the notation in Chapter 2 of the book.  Here,
# 		# l = 1 means the last layer of neurons, l = 2 is the
# 		# second-last layer, and so on.  It's a renumbering of the
# 		# scheme in the book, used here to take advantage of the fact
# 		# that Python can use negative indices in lists.
# 		for l in range(2, self.num_layers):
# 			z = zs[-l]
# 			sp = sigmoid_prime(z)
# 			delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
# 			nabla_b[-l] = delta
# 			nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
# 		return (nabla_b, nabla_w)
	
# 	def evaluate(self, test_data):
# 		"""Return the number of test inputs for which the neural
# 		network outputs the correct result. Note that the neural
# 		network's output is assumed to be the index of whichever
# 		neuron in the final layer has the highest activation."""
# 		test_results = [(np.argmax(self.feedforward(x)), y)
# 						for (x, y) in test_data]
# 		return sum(int(x == y) for (x, y) in test_results)

# 	def cost_derivative(self, output_activations, y):
# 		"""Return the vector of partial derivatives \partial C_x /
# 		\partial a for the output activations."""
# 		return (output_activations-y)


# #if z is a np array, the sigmoid is applied elementwise
# def sigmoid(z):
# 	return 1.0/(1.0+np.exp(-z))

# def sigmoid_prime(z):
# 	"""Derivative of the sigmoid function."""
# 	return sigmoid(z)*(1-sigmoid(z))

"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

class Network(object):

	def __init__(self, sizes):
		"""The list ``sizes`` contains the number of neurons in the
		respective layers of the network.  For example, if the list
		was [2, 3, 1] then it would be a three-layer network, with the
		first layer containing 2 neurons, the second layer 3 neurons,
		and the third layer 1 neuron.  The biases and weights for the
		network are initialized randomly, using a Gaussian
		distribution with mean 0, and variance 1.  Note that the first
		layer is assumed to be an input layer, and by convention we
		won't set any biases for those neurons, since biases are only
		ever used in computing the outputs from later layers."""
		self.num_layers = len(sizes)
		self.sizes = sizes
		self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
		self.weights = [np.random.randn(y, x)
						for x, y in zip(sizes[:-1], sizes[1:])]

	def feedforward(self, a):
		"""Return the output of the network if ``a`` is input."""
		for b, w in zip(self.biases, self.weights):
			a = sigmoid(np.dot(w, a)+b)
		return a

	def SGD(self, training_data, epochs, mini_batch_size, eta,
			test_data=None):
		"""Train the neural network using mini-batch stochastic
		gradient descent.  The ``training_data`` is a list of tuples
		``(x, y)`` representing the training inputs and the desired
		outputs.  The other non-optional parameters are
		self-explanatory.  If ``test_data`` is provided then the
		network will be evaluated against the test data after each
		epoch, and partial progress printed out.  This is useful for
		tracking progress, but slows things down substantially."""
		if test_data: n_test = len(test_data)
		n = len(training_data)
		for j in range(epochs):
			random.shuffle(training_data)
			mini_batches = [
				training_data[k:k+mini_batch_size]
				for k in range(0, n, mini_batch_size)]
			for mini_batch in mini_batches:
				self.update_mini_batch(mini_batch, eta)
			if test_data:
				print("Epoch {0}: {1} / {2}".format(
					j, self.evaluate(test_data), n_test))
			else:
				print("Epoch {0} complete".format(j))

	def update_mini_batch(self, mini_batch, eta):
		"""Update the network's weights and biases by applying
		gradient descent using backpropagation to a single mini batch.
		The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
		is the learning rate."""
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		for x, y in mini_batch:
			delta_nabla_b, delta_nabla_w = self.backprop(x, y)
			nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
			nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
		self.weights = [w-(eta/len(mini_batch))*nw
						for w, nw in zip(self.weights, nabla_w)]
		self.biases = [b-(eta/len(mini_batch))*nb
					   for b, nb in zip(self.biases, nabla_b)]

	def backprop(self, x, y):
		"""Return a tuple ``(nabla_b, nabla_w)`` representing the
		gradient for the cost function C_x.  ``nabla_b`` and
		``nabla_w`` are layer-by-layer lists of numpy arrays, similar
		to ``self.biases`` and ``self.weights``."""
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		# feedforward
		activation = x
		activations = [x] # list to store all the activations, layer by layer
		zs = [] # list to store all the z vectors, layer by layer
		for b, w in zip(self.biases, self.weights):
			z = np.dot(w, activation)+b
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)
		# backward pass
		delta = self.cost_derivative(activations[-1], y) * \
			sigmoid_prime(zs[-1])
		nabla_b[-1] = delta
		nabla_w[-1] = np.dot(delta, activations[-2].transpose())
		# Note that the variable l in the loop below is used a little
		# differently to the notation in Chapter 2 of the book.  Here,
		# l = 1 means the last layer of neurons, l = 2 is the
		# second-last layer, and so on.  It's a renumbering of the
		# scheme in the book, used here to take advantage of the fact
		# that Python can use negative indices in lists.
		for l in range(2, self.num_layers):
			z = zs[-l]
			sp = sigmoid_prime(z)
			delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
			nabla_b[-l] = delta
			nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
		return (nabla_b, nabla_w)

	def evaluate(self, test_data):
		"""Return the number of test inputs for which the neural
		network outputs the correct result. Note that the neural
		network's output is assumed to be the index of whichever
		neuron in the final layer has the highest activation."""
		test_results = [(np.argmax(self.feedforward(x)), y)
						for (x, y) in test_data]
		return sum(int(x == y) for (x, y) in test_results)

	def cost_derivative(self, output_activations, y):
		"""Return the vector of partial derivatives \partial C_x /
		\partial a for the output activations."""
		return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
	"""The sigmoid function."""
	return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
	"""Derivative of the sigmoid function."""
	return sigmoid(z)*(1-sigmoid(z))
